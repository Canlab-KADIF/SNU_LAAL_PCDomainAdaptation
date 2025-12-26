import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader, build_mixup_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from eval_utils.generate_pseudo_labels import inference_and_generate_pseudo_labes

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--pseudo_thresh', type=float, default=0.1, required=True, help='specify the pseudo label thresh')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


class PseudoLabelGenerator:
    def __init__(self, args, config):
        self.args = args
        self.cfg = config
        self.logger = None
        self.output_dir = None
        self.ckpt_dir = None
        self.num_gpus = 1
        self.use_distributed_training = False
        self.unlabel_set = None
        self.unlabel_loader = None
        self.sampler = None
        self.model = None
        self.unlabel_infos_path = None
        self.tb_log = None

    def initialize_system(self):
        if self.args.launcher == 'none':
            self.use_distributed_training = False
            self.num_gpus = 1
        else:
            self.num_gpus, self.cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % self.args.launcher)(
                self.args.tcp_port, self.args.local_rank, backend='nccl'
            )
            self.use_distributed_training = True

        if self.args.batch_size is None:
            self.args.batch_size = self.cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
        else:
            assert self.args.batch_size % self.num_gpus == 0, "Batch size should match the number of gpus"
            self.args.batch_size = self.args.batch_size // self.num_gpus

        self.args.epochs = self.cfg.OPTIMIZATION.NUM_EPOCHS if self.args.epochs is None else self.args.epochs

        if self.args.fix_random_seed:
            common_utils.set_random_seed(666)

        self.output_dir = Path("#ROOT PATH") / 'output' / self.cfg.EXP_GROUP_PATH / self.cfg.TAG / self.args.extra_tag # ROOT PATH
        self.ckpt_dir = self.output_dir / 'ckpt'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.logger = common_utils.create_logger(log_file, rank=self.cfg.LOCAL_RANK)

        self.logger.info('=== Initiating Pseudo Label Generation Setup ===')
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        self.logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

        if self.use_distributed_training:
            self.logger.info('total_batch_size: %d' % (self.num_gpus * self.args.batch_size))
        for key, val in vars(self.args).items():
            self.logger.info('{:16} {}'.format(key, val))
        log_config_to_file(self.cfg, logger=self.logger)
        if self.cfg.LOCAL_RANK == 0:
            os.system('cp %s %s' % (self.args.cfg_file, self.output_dir))

        self.tb_log = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard')) if self.cfg.LOCAL_RANK == 0 else None

    def setup_unlabeled_dataloader(self):
        self.unlabel_set, self.unlabel_loader, self.sampler = build_dataloader(
            dataset_cfg=self.cfg.UNLABEL_DATA_CONFIG, 
            class_names=self.cfg.CLASS_NAMES, 
            batch_size=self.args.batch_size, 
            dist=self.use_distributed_training, workers=self.args.workers, logger=self.logger, training=False
        )

        self.unlabel_infos_path = self.unlabel_set.root_path / self.cfg.UNLABEL_DATA_CONFIG.INFO_PATH['test'][0]

    def setup_model(self):
        self.model = build_network(model_cfg=self.cfg.MODEL, num_class=len(self.cfg.CLASS_NAMES), dataset=self.unlabel_set)

    def generate_pseudo_labels(self):
        assert self.args.ckpt is not None, "CKPT needs to be provided for generating pseudo labels!"
        if self.args.ckpt is None:
            return 
        self.model.load_params_from_file(filename=self.args.ckpt, logger=self.logger, to_cpu=self.use_distributed_training)
        self.model.cuda()

        inference_and_generate_pseudo_labes(cfg=self.cfg, args=self.args, model=self.model, dataloader=self.unlabel_loader, logger=self.logger, 
                                                           dist_test=self.use_distributed_training, save_to_file=self.args.save_to_file, result_dir=self.output_dir, 
                                                           unlabel_infos_path=self.unlabel_infos_path)

    def run(self):
        self.setup_unlabeled_dataloader()
        self.setup_model()

        with torch.no_grad():
            self.generate_pseudo_labels()

        self.logger.info('=== Completed Generating Pseudo Labels for %s ===' %
                    self.cfg.UNLABEL_DATA_CONFIG.INFO_PATH['test'][0])


if __name__ == '__main__':
    args, cfgs = parse_config()
    generator = PseudoLabelGenerator(args, cfgs)
    generator.initialize_system()
    generator.run()
