import argparse
import math
import sys
import os
from functools import partial

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb

import utils
from datasets import datasets
from models import models


def ddp_setup(rank, world_size, port='12355'):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_reduce(x):
    res  = torch.tensor(x).cuda()
    dist.reduce(res, 0)
    return res.item()

def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    
    dataset = datasets.make(spec['dataset'])

    if dist.get_rank() == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=False,
        num_workers=spec.get('num_workers', 0),
        pin_memory=True,
        sampler=DistributedSampler(dataset, shuffle=(tag == 'train'))
    )
    return loader

def make_test_data_loader():
    test_loader = make_data_loader(config.get('test_dataset'), tag='test')
    return test_loader

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = models.make(checkpoint['model'], load_sd=True)
    return model

@torch.no_grad()
def test(test_loader, model, model_=None, enable_amp=False):
    model.eval()
    if model_ is not None:
        model_.eval()
    loss_fn = nn.CrossEntropyLoss()
    test_loss = utils.Averager()
    test_acc = utils.Accuracy()

    with tqdm(test_loader, leave=False, desc='test', ascii=True) as t:
        for batch in t:
            for k, v in batch.items():
                batch[k] = v.cuda()
            inp = batch['keypoint']
            labels = batch['label'].squeeze(-1)
            with torch.cuda.amp.autocast(enabled=enable_amp):
                logits = model(inp)
                if model_ is not None:
                    logits_ = model_(inp)
                    logits = (logits + logits_) / 2
                loss = loss_fn(logits, labels)
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)
            preds = logits.argmax(dim=1)
            test_loss.add(loss.item())
            test_acc.add(preds, labels)
            preds = None; loss = None
            tqdm.set_postfix(t, {
                'loss': test_loss.item(),
                'test_acc': test_acc.item()})

    return test_loss, test_acc

def main(rank, world_size, config_, save_path, args):
    global config, log
    ddp_setup(rank, world_size, args.port)
    config = config_
    if rank == 0:
        # save_name = save_path.split('/')[-1]
        # wandb.init(project='Skeleton2vec', name=save_name)
        log = utils.set_save_path(save_path, remove=False)
        log = partial(log, filename=f'test_log_{args.port}.txt')
        with open(os.path.join(save_path, f'test_config_{args.port}.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)
    dist.barrier()

    torch.cuda.set_device(rank)
    test_loader = make_test_data_loader()
    model = load_model(config['ckp_path'])
    model = model.cuda()
    model = DDP(model, device_ids=[rank], output_device=rank)
    if config.get('ckp_path_') is not None:
        model_ = load_model(config['ckp_path_'])
        model_ = model_.cuda()
        model_ = DDP(model_, device_ids=[rank], output_device=rank)
    else:
        model_ = None
    if args.compile:
        if rank == 0:
            log('Compiling model...')
        model = torch.compile(model)

    timer = utils.Timer()
    timer.s()

    test_loss, test_acc = test(test_loader, model, model_, args.enable_amp)
    v = ddp_reduce(test_loss.v)
    n = ddp_reduce(test_loss.n)
    correct_num = ddp_reduce(test_acc.correct_num)
    total_num = ddp_reduce(test_acc.total_num)
    log_info = []
    if rank == 0:
        # print(v, n, correct_num, total_num)
        val_loss = (v / n)
        val_acc = (correct_num / total_num)
        log_info.append(f'test: loss={val_loss:.4f}')
        log_info.append(f'test: acc={val_acc:.4f}')
        # wandb.log({'val/loss': val_loss, 'val/acc': val_acc}, epoch)

        t = timer.t()
        time = utils.time_text(t)
        log_info.append(f'time: {time}')
        if rank == 0:
            log(', '.join(log_info))
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--port', default='12355')
    parser.add_argument('--enable_amp', action='store_true', default=False,
                        help='Enabling automatic mixed precision')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Enabling torch.Compile')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    ckp_path = config.get('ckp_path')
    if ckp_path is None:
        raise ValueError('ckp_path is not specified in the config file.')
    save_path = os.path.join(*ckp_path.split('/')[:-1])

    if args.enable_amp:
        print('Enable amp')

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, config, save_path, args), nprocs=world_size)