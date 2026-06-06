r""" Logging during training/testing """
import datetime
import logging
import os

import torch
from tensorboardX import SummaryWriter

from common import utils


class AverageMeter2:
    r""" Stores loss, evaluation results for multiple groups """

    def __init__(self, dataset):

        self.benchmark = dataset.benchmark
        self.class_ids_interest = dataset.class_ids
        self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()

        if self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80
        elif self.benchmark == 'fss':
            self.nclass = 1000
        elif self.benchmark == 'pascal_part':
            self.nclass = 100
        elif self.benchmark == 'lvis':
            self.nclass = 1203

        # Initialize a dictionary to hold data for each group
        self.groups = {}

    def _get_group(self, group_name):
        """ Helper function to initialize the group data if not already present """
        if group_name not in self.groups:
            self.groups[group_name] = {
                'intersection_buf': torch.zeros([2, self.nclass]).float().cuda(),
                'union_buf': torch.zeros([2, self.nclass]).float().cuda(),
                'ones': torch.ones_like(torch.zeros([2, self.nclass])).float().cuda(),
                'loss_buf': [],
            }
        return self.groups[group_name]

    def update(self, group_name, inter_b, union_b, class_id, loss):
        group = self._get_group(group_name)
        group['intersection_buf'].index_add_(1, class_id, inter_b.float())
        group['union_buf'].index_add_(1, class_id, union_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        group['loss_buf'].append(loss)

    def compute_iou(self, group_name):
        group = self._get_group(group_name)
        iou = group['intersection_buf'].float() / \
              torch.max(torch.stack([group['union_buf'], group['ones']]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (group['intersection_buf'].index_select(1, self.class_ids_interest).sum(dim=1) /
                  group['union_buf'].index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou

    def compute_average_loss(self, group_name):
        group = self._get_group(group_name)
        return utils.mean(group['loss_buf'])

    def write_result(self, group_name, split, epoch):
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        pre_len = len(msg) - 1

        if isinstance(group_name, str) or len(group_name) == 1:
            if isinstance(group_name, list):
                group_name = group_name[0]
            msg += '─── '
            iou, fb_iou = self.compute_iou(group_name)
            msg += group_name + ' ── '
            group = self._get_group(group_name)
            loss_buf = torch.stack(group['loss_buf'])

            msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'mIoU: %5.2f   ' % iou
            msg += 'FB-IoU: %5.2f   ' % fb_iou

            msg += '***\n'
        elif isinstance(group_name, list):
            group_names = group_name
            max_len = len(max(group_names, key=len))
            for idx, group_name in enumerate(group_names):
                if idx == 0:
                    msg += '┬── '
                elif idx == len(group_names) - 1:
                    msg += ' ' * pre_len + '└── '
                else:
                    msg += ' ' * pre_len + '├── '
                iou, fb_iou = self.compute_iou(group_name)
                group = self._get_group(group_name)
                loss_buf = torch.stack(group['loss_buf'])
                msg += group_name + (max_len - len(group_name)) * ' ' + ' ── '

                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
                msg += 'mIoU: %5.2f   ' % iou
                msg += 'FB-IoU: %5.2f   ' % fb_iou
                msg += '***\n'
        Logger.info(msg)

    def write_process(self, group_name, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx + 1, datalen)
            pre_len = len(msg)

            if isinstance(group_name, str) or len(group_name) == 1:
                if isinstance(group_name, list):
                    group_name = group_name[0]
                iou, fb_iou = self.compute_iou(group_name)
                msg += '─── '
                if epoch != -1:
                    group = self._get_group(group_name)
                    loss_buf = torch.stack(group['loss_buf'])
                    msg += group_name + ' ── '
                    msg += 'L: %6.5f  ' % loss_buf[-1]
                    msg += 'Avg L: %6.5f  ' % loss_buf.mean()
                msg += 'mIoU: %5.2f  |  ' % iou
                msg += 'FB-IoU: %5.2f' % fb_iou

            elif isinstance(group_name, list):
                group_names = group_name
                max_len = len(max(group_names, key=len))
                for idx, group_name in enumerate(group_names):
                    if idx == 0:
                        msg += '┬── '
                    elif idx == len(group_names) - 1:
                        msg += ' ' * pre_len + '└── '
                    else:
                        msg += ' ' * pre_len + '├── '
                    iou, fb_iou = self.compute_iou(group_name)
                    if epoch != -1:
                        group = self._get_group(group_name)
                        loss_buf = torch.stack(group['loss_buf'])
                        msg += group_name + (max_len - len(group_name)) * ' ' + ' ── '
                        msg += 'L: %6.5f  ' % loss_buf[-1]
                        msg += 'Avg L: %6.5f  ' % loss_buf.mean()
                    msg += 'mIoU: %5.2f  |  ' % iou
                    msg += 'FB-IoU: %5.2f' % fb_iou
                    if idx < len(group_names) - 1:
                        msg += '\n'
            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """

    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
        if logpath == '':
            logpath = logtime

        cls.logpath = os.path.join('logs', logpath + '.log')
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info(f'\n:=========== Few-shot Seg. with {args.net_name} ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou, optimizer, scheduler, scaler=None, is_resume=False, is_best=True):

        if scheduler is None and scaler is None:
            state_dict1 = {"net": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
        elif scheduler is not None and scaler is None:
            state_dict1 = {"net": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "epoch": epoch}
        elif scheduler is None and scaler is not None:
            state_dict1 = {"net": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scaler": scaler.state_dict(), "epoch": epoch}
        else:
            state_dict1 = {"net": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch}
        state_dict2 = {"net": model.state_dict()}
        if is_resume:
            torch.save(state_dict1, os.path.join(cls.logpath, f'resume.pt'))
        else:
            if is_best:
                torch.save(state_dict2, os.path.join(cls.logpath, f'best.pt'))
            else:
                torch.save(state_dict2, os.path.join(cls.logpath, f'{epoch}_{val_miou:.2f}.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))

    @classmethod
    def log_params(cls, model):
        learnable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_param = sum(p.numel() for p in model.parameters())
        Logger.info('Learnable # param.: %d' % learnable_param)
        Logger.info('Total # param.: %d' % total_param)
