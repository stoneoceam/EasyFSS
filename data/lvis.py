r""" LVIS-92i few-shot semantic segmentation dataset """
import os
import pickle
from typing import List

import PIL.Image as Image
import numpy as np
import pycocotools.mask as mask_util
import torch
from torch.utils.data import Dataset


def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        return np.zeros((height, width), dtype=bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(bool)


class DatasetLVIS(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, test_num=1000):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 10
        self.benchmark = 'lvis'
        self.shot = shot

        self.anno_path = os.path.join(datapath, "LVIS")
        self.base_path = os.path.join(datapath, "LVIS", 'coco')
        self.transform = transform

        self.nclass, self.class_ids_ori, self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.class_ids_c = {cid: i for i, cid in enumerate(self.class_ids_ori)}
        self.class_ids = sorted(list(self.class_ids_c.values()))

        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 2300

    def __getitem__(self, idx):
        idx %= len(self.class_ids)

        query_image, query_cmask, support_images, support_cmasks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame(
            idx)

        data = {'image': query_image, 'mask': Image.fromarray(query_cmask)}
        query_img, query_mask = self.transform(data).values()
        query_mask = query_mask.float()

        support_imgs = []
        support_masks = []

        for support_img, support_mask in zip(support_images, support_cmasks):
            data = {'image': support_img, 'mask': Image.fromarray(support_mask)}
            support_img, support_mask = self.transform(data).values()
            support_imgs.append(support_img)
            support_masks.append(support_mask.float())

        support_imgs = torch.stack(support_imgs)
        support_masks = torch.stack(support_masks)

        batch = {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_name': query_name,
            'org_query_imsize': org_qry_imsize,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'support_names': support_names,
            'class_id': torch.tensor(self.class_ids_c[class_sample])
        }

        return batch

    def build_img_metadata_classwise(self):
        with open(os.path.join(self.anno_path, 'lvis_train.pkl'), 'rb') as f:
            train_anno = pickle.load(f)
        with open(os.path.join(self.anno_path, 'lvis_val.pkl'), 'rb') as f:
            val_anno = pickle.load(f)

        train_cat_ids = [i for i in train_anno.keys() if len(train_anno[i]) > self.shot]
        val_cat_ids = [i for i in val_anno.keys() if len(val_anno[i]) > self.shot]

        trn_nclass = len(train_cat_ids)
        val_nclass = len(val_cat_ids)

        nclass_val_spilt = val_nclass // self.nfolds

        class_ids_val = [val_cat_ids[self.fold + self.nfolds * v] for v in range(nclass_val_spilt)]
        class_ids_trn = [x for x in train_cat_ids if x not in class_ids_val]

        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val
        nclass = trn_nclass if self.split == 'trn' else val_nclass
        img_metadata_classwise = train_anno if self.split == 'trn' else val_anno

        return nclass, class_ids, img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata.extend(list(self.img_metadata_classwise[k].keys()))
        return sorted(list(set(img_metadata)))

    def get_mask(self, segm, image_size):
        if isinstance(segm, list):
            polygons = [np.asarray(p) for p in segm]
            mask = polygons_to_bitmask(polygons, *image_size[::-1])
        elif isinstance(segm, dict):
            mask = mask_util.decode(segm)
        elif isinstance(segm, np.ndarray):
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(segm.ndim)
            mask = segm
        else:
            raise NotImplementedError

        return torch.as_tensor(mask)

    def merge_masks(self, annotations, image_size):
        masks = [
            self.get_mask(anno['segmentation'], image_size)[None, ...].float()
            for anno in annotations
        ]
        merged_mask = torch.cat(masks, dim=0).sum(0) > 0
        return merged_mask.to(torch.uint8).float().numpy()

    def load_frame(self, idx):
        class_sample = self.class_ids_ori[idx]
        class_img_names = list(self.img_metadata_classwise[class_sample].keys())

        query_name = np.random.choice(class_img_names, 1, replace=False)[0]
        query_info = self.img_metadata_classwise[class_sample][query_name]

        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        org_qry_imsize = query_img.size
        query_mask = self.merge_masks(query_info['annotations'], org_qry_imsize)

        support_names = []
        support_imgs = []
        support_masks = []

        while True:
            support_name = np.random.choice(class_img_names, 1, replace=False)[0]
            if query_name != support_name:
                support_names.append(support_name)

                support_info = self.img_metadata_classwise[class_sample][support_name]
                support_img = Image.open(os.path.join(self.base_path, support_name)).convert('RGB')
                support_mask = self.merge_masks(support_info['annotations'], support_img.size)

                support_imgs.append(support_img)
                support_masks.append(support_mask)

            if len(support_names) == self.shot:
                break

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize
