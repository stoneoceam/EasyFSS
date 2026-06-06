r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle

import PIL.Image as Image
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, test_num=1000):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = os.path.join(datapath, 'COCO2014')
        self.transform = transform
        self.test_num = test_num

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()
        self.class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
                            'trafficlight', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite',
                            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                            'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                            'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
                            'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse', 'remote',
                            'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else self.test_num

    def __getitem__(self, idx):
        query_image, query_cmask, support_images, support_cmasks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame(
            idx)

        data = {'image': query_image, 'mask': Image.fromarray(query_cmask)}
        query_img, query_mask = self.transform(data).values()
        query_mask = query_mask.float()

        support_imgs = []
        support_masks = []

        for (support_img, support_mask) in zip(support_images, support_cmasks):
            data = {'image': support_img, 'mask': Image.fromarray(support_mask)}
            support_img, support_mask = self.transform(data).values()

            support_imgs.append(support_img)
            support_masks.append(support_mask.float())

        support_imgs = torch.stack(support_imgs)
        support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'class_id': torch.tensor(class_sample),
                 'class_name': self.class_names[torch.tensor(class_sample)]}

        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        with open('./data/splits/coco/%s/fold%d.pkl' % (self.split, self.fold), 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png'))
        return mask

    def load_frame(self, idx):

        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]

        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        # 加载图像和掩码
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize
