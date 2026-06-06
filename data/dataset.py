""" Dataloader builder for few-shot semantic segmentation dataset  """
import albumentations as A
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.coco import DatasetCOCO
from data.fss import DatasetFSS
from data.lvis import DatasetLVIS
from data.pascal import DatasetPASCAL
from data.pascal_part import DatasetPASCALPart


class AlbumentationsWrapper:
    def __init__(self):
        self.aug = A.Compose([
            A.ToGray(p=0.2),
            A.Posterize(p=0.2),
            A.Equalize(p=0.2),
            A.Sharpen(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Solarize(p=0.2),
            A.ColorJitter(p=0.2),
        ])

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        image_np = np.array(image)
        mask_np = np.array(mask)

        augmented = self.aug(image=image_np, mask=mask_np)

        return {
            'image': Image.fromarray(augmented['image']),
            'mask': Image.fromarray(augmented['mask'])
        }


class Resize(object):

    def __init__(self, size):
        self.size1 = size

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        image1 = F.resize(image, self.size1, interpolation=F.InterpolationMode.BILINEAR)
        mask1 = F.resize(mask, self.size1, interpolation=F.InterpolationMode.NEAREST)

        return {
            'image1': image1,
            'mask1': mask1,
        }


class ToTensor(object):
    def __call__(self, data):
        output = {}
        for k, v in data.items():
            if isinstance(v, Image.Image):
                if 'mask' in k:
                    output[k] = torch.as_tensor(np.array(v), dtype=torch.long)
                else:
                    output[k] = F.to_tensor(v)
        return output


class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        output = {}
        for k, v in data.items():
            if 'image' in k:
                output[k] = F.normalize(v, self.mean, self.std)
            else:
                output[k] = v
        return output


class FSSDataset:

    @classmethod
    def initialize(cls, size, datapath, data_augment=True):
        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
            'lvis': DatasetLVIS,
            'pascal_part': DatasetPASCALPart,
        }

        cls.size = size

        cls.img_mean = (0.485, 0.456, 0.406)
        cls.img_std = (0.229, 0.224, 0.225)

        cls.datapath = datapath
        cls.data_augment = data_augment

        cls.transform_trn = transforms.Compose([
            AlbumentationsWrapper(),
            Resize((size, size)),
            ToTensor(),
            # Normalize(cls.img_mean, cls.img_std),
        ])

        cls.transform = transforms.Compose([
            Resize((size, size)),
            ToTensor(),
            # Normalize(cls.img_mean, cls.img_std),
        ])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1, test_num=1000):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        if split == 'trn' and cls.data_augment is True:
            transform = cls.transform_trn
        else:
            transform = cls.transform

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=transform,
                                          split=split, shot=shot, test_num=test_num)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader, dataset


if __name__ == '__main__':
    datapath = '/home/stone/Documents/Dataset/Datasets_MyNet'
    benchmark = 'pascal_part'
    bsz = 20
    nworker = 8
    fold = 0
    FSSDataset.initialize(420, datapath=datapath, data_augment=False)
    # dataloader_trn, dataset_trn = FSSDataset.build_dataloader(benchmark, bsz, nworker, fold, 'trn')
    dataloader_val, dataset_val = FSSDataset.build_dataloader(benchmark, bsz, nworker, fold, 'test')

    for idx, batch in enumerate(tqdm(dataloader_val)):
        # print(batch)
        pass
