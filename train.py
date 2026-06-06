import argparse
import os
import time

import torch
from torch import optim, nn, autocast, GradScaler

from common import utils
from common.evaluation import Evaluator
from common.logger import Logger, AverageMeter2
from data.dataset import FSSDataset
from model.net import Net

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def train(epoch, model, dataloader, optimizer, training, scaler=None, use_amp=False):
    train.count = getattr(train, 'count', 0)

    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter2(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)

        if use_amp:
            with autocast(dtype=torch.bfloat16, device_type='cuda'):
                loss, pred1, pred2 = model(
                    batch['query_img'],
                    batch['query_mask'],
                    batch['query_name'],
                    batch['support_imgs'].squeeze(1),
                    batch['support_masks'].squeeze(1),
                )  # , batch['class_name'])

                logit_mask2 = pred2

        else:
            loss, pred1, pred2 = model(
                batch['query_img'],
                batch['query_mask'],
                batch['query_name'],
                batch['support_imgs'].squeeze(1),
                batch['support_masks'].squeeze(1),
            )  # , batch['class_name'])

            logit_mask2 = pred2

        pred_mask = (torch.sigmoid(logit_mask2) > 0.5).float()

        if training:
            optimizer.zero_grad()

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1).clone(), batch)
        average_meter.update('final', area_inter, area_union, batch['class_id'], loss.detach().clone())

        average_meter.write_process(['final'], idx, len(dataloader), epoch,
                                    write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result(['final'],
                               'Training' if training else 'Validation',
                               epoch)
    avg_loss = average_meter.compute_average_loss('final')
    miou, fb_iou = average_meter.compute_iou('final')

    return avg_loss, miou, fb_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', type=str, default="Net")

    parser.add_argument(
        '--sam2_backbone_size',
        type=str,
        default='base',
        choices=['tiny', 'small', 'base', 'large'],
        help='Backbone size of SAM2. Choices: tiny, small, base, large'
    )
    parser.add_argument(
        '--dinov2_backbone_size',
        type=str,
        default='base',
        choices=['small', 'base', 'large'],
        help='Backbone size of DINOv2. Choices: small, base, large'
    )

    parser.add_argument('--logpath', type=str, default='')

    parser.add_argument('--datapath', type=str, default="/home/stone/Documents/Dataset/Datasets_MyNet",
                        help="Path to dataset")
    parser.add_argument(
        '--benchmark',
        type=str, default="pascal",
        choices=['pascal', 'coco', 'fss'],
        help="Benchmark dataset name"
    )
    parser.add_argument('--bsz', type=int, default=20, help="Batch size")
    parser.add_argument('--nworker', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--fold', type=int, default=2, help="Fold index for cross-validation")
    parser.add_argument('--use_aug', action='store_true', help='Use data augmentation')

    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoint")
    parser.add_argument('--loadpath', type=str, default='', help="Path to load model checkpoint")

    parser.add_argument('--epoch', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="Weight decay for optimizer")

    parser.add_argument('--seed', type=int, default=0, help="Random seed")

    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision training with bfloat16')
    parser.add_argument('--amp_dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'],
                        help='Data type for mixed precision training')

    args = parser.parse_args()

    Logger.initialize(args, training=True)
    model = Net(args.sam2_backbone_size, args.dinov2_backbone_size)
    Logger.log_params(model)

    optimizer = optim.AdamW([{"params": model.parameters()}], lr=args.lr, weight_decay=args.weight_decay)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    scaler = None
    if args.use_amp:
        scaler = GradScaler(device='cuda')
        Logger.info(f'Using automatic mixed precision training with {args.amp_dtype}')

        if args.amp_dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
            Logger.warning('Current GPU does not support bfloat16, falling back to float16')
            args.amp_dtype = 'float16'

    model = nn.DataParallel(model)
    model.to(device)

    Evaluator.initialize()
    current_epoch = 0
    if args.resume:
        path = args.loadpath
        checkpoint = torch.load(path)
        current_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.use_amp and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        model.train()

    FSSDataset.initialize(420, datapath=args.datapath, data_augment=args.use_aug)
    dataloader_trn, dataset_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    dataloader_val, dataset_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(current_epoch, args.epoch):

        utils.fix_randseed(None)
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer,
                                               training=True, scaler=scaler, use_amp=args.use_amp)

        utils.fix_randseed(args.seed)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer,
                                                   training=False, scaler=scaler, use_amp=args.use_amp)
        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            if args.use_amp and scaler is not None:
                Logger.save_model_miou(model, epoch, val_miou, optimizer, scheduler=None, scaler=scaler)
            else:
                Logger.save_model_miou(model, epoch, val_miou, optimizer, scheduler=None)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()

    end_time = time.time()
    elapsed = int(end_time - start_time)
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    Logger.info(f'Elapsed: {hours}h {minutes}m {seconds}s')
    Logger.info(f'========= Finished Training , best_val_miou:{best_val_miou:.2f} ==========')
    Logger.tbd_writer.close()
