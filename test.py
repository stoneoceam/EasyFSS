import argparse
import os
import time

import torch
from torch import nn, autocast

from common import utils
from common.evaluation import Evaluator
from common.logger import Logger, AverageMeter2
from common.vis import Visualizer
from data.dataset import FSSDataset
from model.net import Net

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def test(model, dataloader, nshot, use_amp=False):
    average_meter = AverageMeter2(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)

        if use_amp:
            with autocast(dtype=torch.bfloat16, device_type='cuda'):
                pred1, pred2 = model.module.predict_mask_nshot(batch, nshot=nshot)
                logit_mask1 = pred1
                logit_mask2 = pred2

        else:
            pred1, pred2 = model.module.predict_mask_nshot(batch, nshot=nshot)
            logit_mask1 = pred1
            logit_mask2 = pred2

        pred_mask1 = (torch.sigmoid(logit_mask1) > 0.5).float()
        pred_mask2 = (torch.sigmoid(logit_mask2) > 0.5).float()

        area_inter1, area_union1 = Evaluator.classify_prediction(pred_mask1.squeeze(1).clone(), batch)
        area_inter2, area_union2 = Evaluator.classify_prediction(pred_mask2.squeeze(1).clone(), batch)

        average_meter.update('coarse_mask', area_inter1, area_union1, batch['class_id'], loss=None)
        average_meter.update('final', area_inter2, area_union2, batch['class_id'], loss=None)

        average_meter.write_process(['final'], idx, len(dataloader), epoch=-1, write_batch_idx=1)

        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(
                batch['support_imgs'],
                batch['support_masks'],
                batch['query_img'],
                batch['query_mask'],
                pred_mask2,
                batch['class_id'],
                idx,
                batch['query_name'],
                batch['support_names'],
                area_inter2[1].float() / area_union2[1].float()
            )

    # Write evaluation results
    average_meter.write_result(['final'], 'Test', 0)

    miou1, fb_iou1 = average_meter.compute_iou('coarse_mask')
    miou2, fb_iou2 = average_meter.compute_iou('final')

    mious = [miou1, miou2]
    fb_ious = [fb_iou1, fb_iou2]

    return mious, fb_ious


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
        choices=['pascal', 'coco', 'fss', 'lvis', 'pascal_part'],
        help="Benchmark dataset name"
    )
    parser.add_argument('--bsz', type=int, default=1, help="Batch size")
    parser.add_argument('--nworker', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--load', type=str, default='logs/net/pascal-0.log/best.pt')
    parser.add_argument('--fold', type=int, default=0, help="Fold index for cross-validation")
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=1000)

    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--visual_fold_name', type=str, default='')

    parser.add_argument('--seed', type=int, default=0, help="Random seed")

    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision inference with bfloat16')
    parser.add_argument('--amp_dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'],
                        help='Data type for mixed precision inference')

    args = parser.parse_args()

    Logger.initialize(args, training=False)
    model = Net(args.sam2_backbone_size, args.dinov2_backbone_size)
    model.eval()
    Logger.log_params(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    if args.use_amp:
        Logger.info(f'Using automatic mixed precision inference with {args.amp_dtype}')

        if args.amp_dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
            Logger.warning('Current GPU does not support bfloat16, falling back to float16')
            args.amp_dtype = 'float16'

    model = nn.DataParallel(model)
    model.to(device)

    if args.load == '':
        raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load, weights_only=True)['net'])

    Evaluator.initialize()
    Visualizer.initialize(args.visualize, args)

    FSSDataset.initialize(420, datapath=args.datapath)
    dataloader_test, dataset_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold,
                                                                'test', shot=args.nshot, test_num=args.test_num)

    start_time = time.time()
    utils.fix_randseed(args.seed)
    with torch.no_grad():
        test_mious, test_fb_ious = test(model, dataloader_test, args.nshot, use_amp=args.use_amp)
    Logger.info(
        'Fold %d coarse-mask mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_mious[0].item(), test_fb_ious[0].item()))
    Logger.info(
        'Fold %d final-mask mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_mious[1].item(), test_fb_ious[1].item()))

    end_time = time.time()
    elapsed = int(end_time - start_time)
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    Logger.info(f'Elapsed: {hours}h {minutes}m {seconds}s')

    Logger.info('==================== Finished Testing ====================')
