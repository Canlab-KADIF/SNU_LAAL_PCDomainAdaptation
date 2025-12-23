import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def load_evaluation_arguments():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    arguments = parser.parse_args()

    cfg_from_yaml_file(arguments.cfg_file, cfg)
    cfg.TAG = Path(arguments.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(arguments.cfg_file.split('/')[1:-1])

    np.random.seed(1024)

    if arguments.set_cfgs is not None:
        cfg_from_list(arguments.set_cfgs, cfg)

    return arguments, cfg


def evaluate_single_checkpoint(neural_network, test_data_loader, arguments, evaluation_output_path, evaluation_logger, epoch_identifier, use_distributed_testing=False, cfg=None):
    if cfg is None:
        from pcdet.config import cfg
    # Restore model weights from checkpoint file
    neural_network.load_params_from_file(filename=arguments.ckpt, logger=evaluation_logger, to_cpu=use_distributed_testing)
    neural_network.cuda()

    # Run evaluation on test dataset
    eval_utils.eval_one_epoch(
        cfg, neural_network, test_data_loader, epoch_identifier, evaluation_logger, dist_test=use_distributed_testing,
        result_dir=evaluation_output_path, save_to_file=arguments.save_to_file
    )


def find_unevaluated_checkpoint(checkpoint_directory, checkpoint_record_file, arguments, cfg=None):
    if cfg is None:
        from pcdet.config import cfg
    checkpoint_list = glob.glob(os.path.join(checkpoint_directory, '*checkpoint_epoch_*.pth'))
    checkpoint_list.sort(key=os.path.getmtime)
    evaluated_checkpoints = [float(x.strip()) for x in open(checkpoint_record_file, 'r').readlines()]

    for current_checkpoint in checkpoint_list:
        number_list = re.findall('checkpoint_epoch_(.*).pth', current_checkpoint)
        if number_list.__len__() == 0:
            continue

        epoch_identifier = number_list[-1]
        if 'optim' in epoch_identifier:
            continue
        if float(epoch_identifier) not in evaluated_checkpoints and int(float(epoch_identifier)) >= arguments.start_epoch:
            return epoch_identifier, current_checkpoint
    return -1, None


def evaluate_multiple_checkpoints(neural_network, test_data_loader, arguments, evaluation_output_path, evaluation_logger, checkpoint_directory, use_distributed_testing=False, cfg=None):
    if cfg is None:
        from pcdet.config import cfg
    # Create record file to track evaluated checkpoints
    checkpoint_record_file = evaluation_output_path / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(checkpoint_record_file, 'a'):
        pass

    elapsed_time = 0
    is_first_evaluation = True

    while True:
        # Search for checkpoints that haven't been evaluated yet
        current_epoch_id, current_checkpoint = find_unevaluated_checkpoint(checkpoint_directory, checkpoint_record_file, arguments, cfg)
        if current_epoch_id == -1 or int(float(current_epoch_id)) < arguments.start_epoch:
            wait_seconds = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_seconds, elapsed_time * 1.0 / 60, arguments.max_waiting_mins, checkpoint_directory), end='', flush=True)
            time.sleep(wait_seconds)
            elapsed_time += 30
            if elapsed_time > arguments.max_waiting_mins * 60 and (is_first_evaluation is False):
                break
            continue

        elapsed_time = 0
        is_first_evaluation = False

        neural_network.load_params_from_file(filename=current_checkpoint, logger=evaluation_logger, to_cpu=use_distributed_testing)
        neural_network.cuda()

        # Run evaluation on the current checkpoint
        current_result_directory = evaluation_output_path / ('epoch_%s' % current_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        eval_utils.eval_one_epoch(
            cfg, neural_network, test_data_loader, current_epoch_id, evaluation_logger, dist_test=use_distributed_testing,
            result_dir=current_result_directory, save_to_file=arguments.save_to_file
        )

        # Record this epoch in the evaluation log
        with open(checkpoint_record_file, 'a') as f:
            print('%s' % current_epoch_id, file=f)
        evaluation_logger.info('Epoch %s has been evaluated' % current_epoch_id)


class ModelEvaluator:
    def __init__(self, args, config):
        self.args = args
        self.cfg = config
        self.logger = None
        self.experiment_output_path = None
        self.evaluation_output_path = None
        self.checkpoint_directory = None
        self.num_gpus = 1
        self.use_distributed_testing = False
        self.test_dataset = None
        self.test_data_loader = None
        self.test_sampler = None
        self.model = None
        self.epoch_identifier = None

    def initialize_system(self):
        if self.args.launcher != 'none':
            self.num_gpus, self.cfg.LOCAL_RANK = getattr(common_utils, f'init_dist_{self.args.launcher}')(
                self.args.tcp_port, self.args.local_rank, backend='nccl'
            )
            self.use_distributed_testing = True
        else:
            self.use_distributed_testing = False
            self.num_gpus = 1

        if self.args.batch_size is None:
            self.args.batch_size = self.cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
        else:
            assert self.args.batch_size % self.num_gpus == 0, 'Batch size should match the number of gpus'
            self.args.batch_size //= self.num_gpus

        ROOT_DIR = Path("#ROOT PATH") # ROOT PATH
        self.experiment_output_path = ROOT_DIR / 'output' / self.cfg.EXP_GROUP_PATH / self.cfg.TAG / self.args.extra_tag
        self.experiment_output_path.mkdir(parents=True, exist_ok=True)

    def determine_eval_path(self):
        self.evaluation_output_path = self.experiment_output_path / 'eval'

        if not self.args.eval_all:
            number_list = re.findall(r'\d+', self.args.ckpt) if self.args.ckpt is not None else []
            self.epoch_identifier = number_list[-1] if number_list.__len__() > 0 else 'no_number'
            self.evaluation_output_path = self.evaluation_output_path / ('epoch_%s' % self.epoch_identifier) / self.cfg.DATA_CONFIG.DATA_SPLIT['test']
        else:
            self.evaluation_output_path = self.evaluation_output_path / 'eval_all_default'

        if self.args.eval_tag is not None:
            self.evaluation_output_path = self.evaluation_output_path / self.args.eval_tag

        self.evaluation_output_path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        log_file_path = self.evaluation_output_path / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.logger = common_utils.create_logger(log_file_path, rank=self.cfg.LOCAL_RANK)

        self.logger.info('=== Initiating Model Evaluation Setup ===')
        available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')
        self.logger.info(f'CUDA_VISIBLE_DEVICES={available_gpus}')

        if self.use_distributed_testing:
            self.logger.info(f'total_batch_size: {self.num_gpus * self.args.batch_size}')
        for key, val in vars(self.args).items():
            self.logger.info('{:16} {}'.format(key, val))
        log_config_to_file(self.cfg, logger=self.logger)

        self.checkpoint_directory = self.args.ckpt_dir if self.args.ckpt_dir is not None else self.experiment_output_path / 'ckpt'

    def setup_test_dataloader(self):
        self.test_dataset, self.test_data_loader, self.test_sampler = build_dataloader(
            dataset_cfg=self.cfg.DATA_CONFIG,
            class_names=self.cfg.CLASS_NAMES,
            batch_size=self.args.batch_size,
            dist=self.use_distributed_testing, workers=self.args.workers, logger=self.logger, training=False
        )

    def setup_model(self):
        self.model = build_network(model_cfg=self.cfg.MODEL, num_class=len(self.cfg.CLASS_NAMES), dataset=self.test_dataset)

    def run(self):
        with torch.no_grad():
            if self.args.eval_all:
                evaluate_multiple_checkpoints(self.model, self.test_data_loader, self.args, self.evaluation_output_path, self.logger, self.checkpoint_directory, use_distributed_testing=self.use_distributed_testing, cfg=self.cfg)
            else:
                evaluate_single_checkpoint(self.model, self.test_data_loader, self.args, self.evaluation_output_path, self.logger, self.epoch_identifier, use_distributed_testing=self.use_distributed_testing, cfg=self.cfg)


if __name__ == '__main__':
    arguments, configuration = load_evaluation_arguments()
    evaluator = ModelEvaluator(arguments, configuration)
    evaluator.initialize_system()
    evaluator.determine_eval_path()
    evaluator.setup_logging()
    evaluator.setup_test_dataloader()
    evaluator.setup_model()
    evaluator.run()
