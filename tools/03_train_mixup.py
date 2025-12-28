import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import evaluate_multiple_checkpoints

import torch
import torch.nn as nn

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader, build_mixup_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from eval_utils.generate_pseudo_labels import inference_and_generate_pseudo_labes

def load_training_arguments():
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

    parser.add_argument('--pseudo_info_path', type=str, default=None, required=True, help='specify the pseudo label info path')

    arguments = parser.parse_args()

    cfg_from_yaml_file(arguments.cfg_file, cfg)
    cfg.TAG = Path(arguments.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(arguments.cfg_file.split('/')[1:-1])

    if arguments.set_cfgs is not None:
        cfg_from_list(arguments.set_cfgs, cfg)

    return arguments, cfg


class Stage2MixupTrainer:
    def __init__(self, args, config):
        self.args = args
        self.cfg = config
        self.logger = None
        self.experiment_output_path = None
        self.checkpoint_directory = None
        self.num_gpus = 1
        self.use_distributed_training = False
        self.training_dataset = None
        self.training_data_loader = None
        self.training_sampler = None
        self.model = None
        self.model_optimizer = None
        self.initial_epoch = 0
        self.iteration = 0
        self.previous_epoch = -1
        self.learning_rate_scheduler = None
        self.warmup_scheduler = None
        self.test_dataset = None
        self.test_data_loader = None
        self.test_sampler = None

    def initialize_system(self):
        if self.args.launcher != 'none':
            self.num_gpus, self.cfg.LOCAL_RANK = getattr(common_utils, f'init_dist_{self.args.launcher}')(
                self.args.tcp_port, self.args.local_rank, backend='nccl'
            )
            self.use_distributed_training = True
        else:
            self.use_distributed_training = False
            self.num_gpus = 1

        if self.args.batch_size is None:
            self.args.batch_size = self.cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
        else:
            assert self.args.batch_size % self.num_gpus == 0, "Batch size should match the number of gpus"
            self.args.batch_size //= self.num_gpus

        self.args.epochs = self.cfg.OPTIMIZATION.NUM_EPOCHS if self.args.epochs is None else self.args.epochs

        if self.args.fix_random_seed:
            common_utils.set_random_seed(666)

        ROOT_DIR = Path("#ROOT PATH") # ROOT PATH
        self.experiment_output_path = ROOT_DIR / 'output' / self.cfg.EXP_GROUP_PATH / self.cfg.TAG / self.args.extra_tag
        self.checkpoint_directory = self.experiment_output_path / 'ckpt'
        self.experiment_output_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)

        log_file_path = self.experiment_output_path / f"log_train_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        self.logger = common_utils.create_logger(log_file_path, rank=self.cfg.LOCAL_RANK)

        self.logger.info('=== Initiating Stage 2 Mixup Training Setup ===')
        available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')
        self.logger.info(f'CUDA_VISIBLE_DEVICES={available_gpus}')

        if self.use_distributed_training:
            self.logger.info(f'total_batch_size: {self.num_gpus * self.args.batch_size}')
        for key, val in vars(self.args).items():
            self.logger.info('{:16} {}'.format(key, val))
        log_config_to_file(self.cfg, logger=self.logger)
        if self.cfg.LOCAL_RANK == 0:
            os.system(f'cp {self.args.cfg_file} {self.experiment_output_path}')

    def setup_data_loaders(self):
        pseudo_info_path = self.args.pseudo_info_path
        self.training_dataset, self.training_data_loader, self.training_sampler = build_mixup_dataloader(
            dataset_cfg=self.cfg.DATA_CONFIG,
            class_names=self.cfg.CLASS_NAMES,
            batch_size=self.args.batch_size,
            dist=self.use_distributed_training, workers=self.args.workers,
            logger=self.logger,
            training=True, 
            pseudo_info_path=pseudo_info_path
        )

    def setup_model(self):
        self.model = build_network(model_cfg=self.cfg.MODEL, num_class=len(self.cfg.CLASS_NAMES), dataset=self.training_dataset)
        if self.args.sync_bn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.cuda()

    def setup_optimizer(self):
        self.model_optimizer = build_optimizer(self.model, self.cfg.OPTIMIZATION)

    def load_checkpoint(self):
        self.initial_epoch = 0
        self.iteration = 0
        self.previous_epoch = -1
        if self.args.pretrained_model is not None:
            self.model.load_params_from_file(filename=self.args.pretrained_model, to_cpu=self.use_distributed_training, logger=self.logger)

        if self.args.ckpt is not None and self.args.pretrained_model is None:
            self.iteration, self.initial_epoch = self.model.load_params_with_optimizer(self.args.ckpt, to_cpu=self.use_distributed_training, optimizer=self.model_optimizer, logger=self.logger)
            self.previous_epoch = self.initial_epoch + 1
        else:
            ckpt_list = glob.glob(str(self.checkpoint_directory / '*checkpoint_epoch_*.pth'))
            if ckpt_list:
                ckpt_list.sort(key=os.path.getmtime)
                self.iteration, self.initial_epoch = self.model.load_params_with_optimizer(
                    ckpt_list[-1], to_cpu=self.use_distributed_training, optimizer=self.model_optimizer, logger=self.logger
                )
                self.previous_epoch = self.initial_epoch + 1

    def prepare_training(self):
        self.model.train()
        if self.use_distributed_training:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.cfg.LOCAL_RANK % torch.cuda.device_count()])
        self.logger.info(self.model)

        self.learning_rate_scheduler, self.warmup_scheduler = build_scheduler(
            self.model_optimizer, total_iters_each_epoch=len(self.training_data_loader), total_epochs=self.args.epochs,
            last_epoch=self.previous_epoch, optim_cfg=self.cfg.OPTIMIZATION
        )

    def run_training(self):
        self.logger.info(f'*** Starting Stage 2 Mixup Training: {self.cfg.EXP_GROUP_PATH}/{self.cfg.TAG}({self.args.extra_tag}) ***')
        train_model(
            self.model,
            self.model_optimizer,
            self.training_data_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=self.learning_rate_scheduler,
            optim_cfg=self.cfg.OPTIMIZATION,
            start_epoch=self.initial_epoch,
            total_epochs=self.args.epochs,
            start_iter=self.iteration,
            rank=self.cfg.LOCAL_RANK,
            ckpt_save_dir=self.checkpoint_directory,
            train_sampler=self.training_sampler,
            lr_warmup_scheduler=self.warmup_scheduler,
            ckpt_save_interval=self.args.ckpt_save_interval,
            max_ckpt_save_num=self.args.max_ckpt_save_num,
            merge_all_iters_to_one_epoch=self.args.merge_all_iters_to_one_epoch
        )

        if hasattr(self.training_dataset, 'use_shared_memory') and self.training_dataset.use_shared_memory:
            self.training_dataset.clean_shared_memory()

        self.logger.info(f'*** Stage 2 Mixup Training Completed: {self.cfg.EXP_GROUP_PATH}/{self.cfg.TAG}({self.args.extra_tag}) ***\n\n\n')

    def run_evaluation(self):
        self.logger.info(f'*** Starting Evaluation Phase: {self.cfg.EXP_GROUP_PATH}/{self.cfg.TAG}({self.args.extra_tag}) ***')
        self.test_dataset, self.test_data_loader, self.test_sampler = build_dataloader(
            dataset_cfg=self.cfg.DATA_CONFIG,
            class_names=self.cfg.CLASS_NAMES,
            batch_size=self.args.batch_size,
            dist=self.use_distributed_training, workers=self.args.workers, logger=self.logger, training=False
        )
        evaluation_output_path = self.experiment_output_path / 'eval' / 'eval_with_train'
        evaluation_output_path.mkdir(parents=True, exist_ok=True)
        self.args.start_epoch = max(self.args.epochs - self.args.num_epochs_to_eval, 0)

        model_for_eval = self.model.module if self.use_distributed_training else self.model
        evaluate_multiple_checkpoints(
            model_for_eval,
            self.test_data_loader, self.args, evaluation_output_path, self.logger, self.checkpoint_directory,
            use_distributed_testing=self.use_distributed_training, cfg=self.cfg
        )
        self.logger.info(f'*** Evaluation Phase Completed: {self.cfg.EXP_GROUP_PATH}/{self.cfg.TAG}({self.args.extra_tag}) ***')


if __name__ == '__main__':
    arguments, configuration = load_training_arguments()
    trainer = Stage2MixupTrainer(arguments, configuration)
    trainer.initialize_system()
    trainer.setup_data_loaders()
    trainer.setup_model()
    trainer.setup_optimizer()
    trainer.load_checkpoint()
    trainer.prepare_training()
    trainer.run_training()
    trainer.run_evaluation()
