import argparse
import gc
import os
import sys
from pathlib import Path
from math import ceil

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import nn, autocast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import utils
# ===== 你自己的项目模块 =====
from model.net import Net


# -----------------------------
# 1. 复用你现在的数据预处理思路
# -----------------------------
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None):
        image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
        if mask is not None:
            mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, mask


class ToTensor(object):
    def __call__(self, image, mask=None):
        image = F.to_tensor(image)  # [C,H,W], float32
        if mask is not None:
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)  # [H,W]
        return image, mask


class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return F.normalize(image, self.mean, self.std)


class DemoPreprocessor:
    def __init__(self, size=420, normalize=False):
        self.resize = Resize((size, size))
        self.to_tensor = ToTensor()
        self.normalize = Normalize()
        self.use_normalize = normalize

    def load_image(self, path):
        image = Image.open(path).convert("RGB")
        ori_w, ori_h = image.size
        image, _ = self.resize(image, None)
        image, _ = self.to_tensor(image, None)
        if self.use_normalize:
            image = self.normalize(image)

        return image, ori_h, ori_w

    def load_mask(self, path):
        # 1. 读取为 numpy
        mask = np.array(Image.open(path).convert("L"))

        # 2. 二值化（和你原逻辑一致）
        mask[mask < 128] = 0
        mask[mask >= 128] = 1

        # 3. 转回 PIL（为了复用 resize）
        mask_pil = Image.fromarray(mask.astype(np.uint8))

        # 4. resize（必须和 image 对齐）
        dummy_image = Image.fromarray(
            np.zeros((mask_pil.size[1], mask_pil.size[0], 3), dtype=np.uint8)
        )
        _, mask_pil = self.resize(dummy_image, mask_pil)

        # 5. 转 tensor
        mask = torch.as_tensor(np.array(mask_pil)).float()

        # # 6. 再保证是 0/1（保险）
        # mask = (mask > 0).long()

        return mask


# -----------------------------
# 2. 构造 episode batch
# -----------------------------
def build_demo_batch(
        support_img_paths,
        support_mask_paths,
        query_img_path,
        query_mask_path=None,
        size=420,
        class_id=0,
        normalize=False,
):
    """
    输出的 batch 尽量与你现有 dataloader 保持一致

    support_imgs:  [1, shot, 3, H, W]
    support_masks: [1, shot, H, W]
    query_img:     [1, 3, H, W]
    query_mask:    [1, H, W] 或 None时用全0占位
    """

    assert len(support_img_paths) == len(support_mask_paths), "support 图像和 mask 数量必须一致"
    shot = len(support_img_paths)

    preprocessor = DemoPreprocessor(size=size, normalize=normalize)

    support_imgs = []
    support_masks = []

    for img_path, mask_path in zip(support_img_paths, support_mask_paths):
        img, _, _ = preprocessor.load_image(img_path)
        mask = preprocessor.load_mask(mask_path)
        support_imgs.append(img)
        support_masks.append(mask)

    query_img, ori_h, ori_w = preprocessor.load_image(query_img_path)

    if query_mask_path is not None and query_mask_path != "" and os.path.exists(query_mask_path):
        query_mask = preprocessor.load_mask(query_mask_path)
    else:
        # 没有 gt 时，用全 0 占位，保证接口不炸
        h, w = query_img.shape[-2:]
        query_mask = torch.zeros((h, w)).float()

    support_imgs = torch.stack(support_imgs, dim=0)  # [shot, 3, H, W]
    support_masks = torch.stack(support_masks, dim=0)  # [shot, H, W]

    batch = {
        "support_imgs": support_imgs.unsqueeze(0),  # [1, shot, 3, H, W]
        "support_masks": support_masks.unsqueeze(0),  # [1, shot, H, W]
        "query_img": query_img.unsqueeze(0),  # [1, 3, H, W]
        "query_mask": query_mask.squeeze(0),  # [1, H, W]
        "class_id": torch.tensor([class_id], dtype=torch.long),

        # 这些主要给可视化/日志用
        "query_name": [os.path.basename(query_img_path)],
        "support_names": [[os.path.basename(p) for p in support_img_paths]],
        "ori_h": ori_h, "ori_w": ori_w,
    }

    return batch


def build_demo_batch_multi(
        support_img_paths,
        support_mask_paths,
        query_img_paths,
        query_mask_paths=None,
        size=420,
        class_id=0,
        normalize=False,
):
    """
    Build a batched episode for K-shot support and N query images.

    support_imgs:  [N, shot, 3, H, W]
    support_masks: [N, shot, H, W]
    query_img:     [N, 3, H, W]
    query_mask:    [N, H, W]
    """

    assert len(support_img_paths) == len(support_mask_paths), "support 图像和 mask 数量必须一致"
    assert len(query_img_paths) > 0, "query_img_paths 不能为空"

    query_mask_paths = query_mask_paths or [""] * len(query_img_paths)
    if len(query_mask_paths) != len(query_img_paths):
        raise ValueError("query 图像和 mask 数量必须一致；没有 mask 的位置请传空字符串")

    shot = len(support_img_paths)
    batch_size = len(query_img_paths)
    preprocessor = DemoPreprocessor(size=size, normalize=normalize)

    support_imgs = []
    support_masks = []
    for img_path, mask_path in zip(support_img_paths, support_mask_paths):
        img, _, _ = preprocessor.load_image(img_path)
        mask = preprocessor.load_mask(mask_path)
        support_imgs.append(img)
        support_masks.append(mask)

    support_imgs = torch.stack(support_imgs, dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    support_masks = torch.stack(support_masks, dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    query_imgs = []
    query_masks = []
    ori_h = []
    ori_w = []
    query_names = []

    for query_img_path, query_mask_path in zip(query_img_paths, query_mask_paths):
        query_img, h, w = preprocessor.load_image(query_img_path)
        if query_mask_path and os.path.exists(query_mask_path):
            query_mask = preprocessor.load_mask(query_mask_path)
        else:
            query_mask = torch.zeros(query_img.shape[-2:], dtype=torch.float32)

        query_imgs.append(query_img)
        query_masks.append(query_mask)
        ori_h.append(h)
        ori_w.append(w)
        query_names.append(os.path.basename(query_img_path))

    batch = {
        "support_imgs": support_imgs,
        "support_masks": support_masks,
        "query_img": torch.stack(query_imgs, dim=0),
        "query_mask": torch.stack(query_masks, dim=0),
        "class_id": torch.full((batch_size,), class_id, dtype=torch.long),
        "query_name": query_names,
        "support_names": [[os.path.basename(p) for p in support_img_paths] for _ in range(batch_size)],
        "ori_h": ori_h,
        "ori_w": ori_w,
    }

    return batch


# -----------------------------
# 3. 单样本推理
# -----------------------------
@torch.no_grad()
def infer_single_episode(model, batch, nshot=1, use_amp=True, amp_dtype="bfloat16"):
    device = next(model.parameters()).device
    batch = utils.move_to_device(batch, device)

    use_bf16 = (amp_dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    amp_t = torch.bfloat16 if use_bf16 else torch.float16

    if use_amp and torch.cuda.is_available():
        with autocast(device_type="cuda", dtype=amp_t):
            _, pred = model.module.predict_mask_nshot(batch, nshot=nshot)
    else:
        _, pred = model.module.predict_mask_nshot(batch, nshot=nshot)

    pred_mask = (torch.sigmoid(pred) > 0.5).float()  # [B,1,H,W]

    result = {
        "logit_mask": pred,
        "pred_mask": pred_mask,
    }

    return result


# -----------------------------
# 4. 保存预测 mask
# -----------------------------
def save_mask(mask_tensor, save_path):
    """
    mask_tensor: [1,1,H,W] or [H,W]
    """
    if mask_tensor.dim() == 4:
        mask = mask_tensor[0, 0].detach().cpu().numpy()
    elif mask_tensor.dim() == 2:
        mask = mask_tensor.detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected mask shape: {mask_tensor.shape}")

    mask = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(mask).save(save_path)


def save_mask_resized(mask_tensor, save_path, size):
    if mask_tensor.dim() == 4:
        mask_tensor = mask_tensor[:1]
        mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=size, mode="nearest")
    save_mask(mask_tensor, save_path)


def load_demo_model(
        ckpt_path,
        sam2_backbone_size='base',
        dinov2_backbone_size='base',
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net(sam2_backbone_size, dinov2_backbone_size)
    model.eval()
    model = nn.DataParallel(model).to(device)

    if ckpt_path == '':
        raise Exception('Pretrained model not specified.')

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['net'])

    return model


def run_demo_inference_multi(
        support_img_paths,
        support_mask_paths,
        query_img_paths,
        query_mask_paths,
        ckpt_path,
        save_pred_paths,
        sam2_backbone_size='base',
        dinov2_backbone_size='base',
        size=420,
        class_id=0,
        use_amp=True,
        amp_dtype='bfloat16',
        normalize=False,
        max_query_batch=20,
):
    assert len(support_img_paths) == len(support_mask_paths), "support_imgs 和 support_masks 数量必须一致"
    if not query_img_paths:
        raise ValueError("query_img_paths 不能为空")
    if len(query_img_paths) != len(save_pred_paths):
        raise ValueError("query_img_paths 和 save_pred_paths 数量必须一致")

    query_mask_paths = query_mask_paths or [""] * len(query_img_paths)
    if len(query_mask_paths) != len(query_img_paths):
        raise ValueError("query_img_paths 和 query_mask_paths 数量必须一致")

    nshot = len(support_img_paths)
    model = load_demo_model(
        ckpt_path=ckpt_path,
        sam2_backbone_size=sam2_backbone_size,
        dinov2_backbone_size=dinov2_backbone_size,
    )

    items = []
    total = len(query_img_paths)
    batch_count = ceil(total / max_query_batch)

    try:
        for start in range(0, total, max_query_batch):
            end = min(start + max_query_batch, total)
            batch = build_demo_batch_multi(
                support_img_paths=support_img_paths,
                support_mask_paths=support_mask_paths,
                query_img_paths=query_img_paths[start:end],
                query_mask_paths=query_mask_paths[start:end],
                size=size,
                class_id=class_id,
                normalize=normalize,
            )

            result = infer_single_episode(
                model=model,
                batch=batch,
                nshot=nshot,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            pred_masks = result["pred_mask"]
            for offset, pred_mask in enumerate(pred_masks):
                idx = start + offset
                save_mask_resized(
                    pred_mask.unsqueeze(0),
                    save_pred_paths[idx],
                    size=(batch["ori_h"][offset], batch["ori_w"][offset]),
                )

                item = {
                    "query_name": os.path.basename(query_img_paths[idx]),
                    "pred_mask_path": save_pred_paths[idx],
                }

                items.append(item)

            del batch, result, pred_masks
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return {
        "pred_mask_paths": save_pred_paths,
        "items": items,
        "query_count": total,
        "batch_count": batch_count,
        "max_query_batch": max_query_batch,
    }


def run_demo_inference(
        support_img_paths,
        support_mask_paths,
        query_img_path,
        query_mask_path,
        ckpt_path,
        save_pred_path,
        sam2_backbone_size='base',
        dinov2_backbone_size='base',
        size=420,
        class_id=0,
        use_amp=True,
        amp_dtype='bfloat16',
        normalize=False,
):
    result = run_demo_inference_multi(
        support_img_paths=support_img_paths,
        support_mask_paths=support_mask_paths,
        query_img_paths=[query_img_path],
        query_mask_paths=[query_mask_path],
        ckpt_path=ckpt_path,
        save_pred_paths=[save_pred_path],
        sam2_backbone_size=sam2_backbone_size,
        dinov2_backbone_size=dinov2_backbone_size,
        size=size,
        class_id=class_id,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        normalize=normalize,
    )

    return {
        "pred_mask_path": save_pred_path,
    }


# -----------------------------
# 5. 主函数
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sam2_backbone_size', type=str, default='base',
                        choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--dinov2_backbone_size', type=str, default='base',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--load', type=str, required=True)

    parser.add_argument('--support_imgs', type=str, nargs='+', required=True,
                        help='support image paths')
    parser.add_argument('--support_masks', type=str, nargs='+', required=True,
                        help='support mask paths')
    parser.add_argument('--query_img', type=str, nargs='+', required=True)
    parser.add_argument('--query_mask', type=str, nargs='*', default=None)

    parser.add_argument('--size', type=int, default=420)
    parser.add_argument('--class_id', type=int, default=0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--amp_dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16'])
    parser.add_argument('--normalize', action='store_true', default=False)

    parser.add_argument('--save_pred_path', type=str, nargs='*', default=None)

    args = parser.parse_args()

    query_masks = args.query_mask or [""] * len(args.query_img)
    save_pred_paths = args.save_pred_path or []
    if not save_pred_paths:
        if len(args.query_img) == 1:
            save_pred_paths = ["pred_mask.png"]
        else:
            save_pred_paths = [f"pred_mask_{i}.png" for i in range(len(args.query_img))]

    result = run_demo_inference_multi(
        support_img_paths=args.support_imgs,
        support_mask_paths=args.support_masks,
        query_img_paths=args.query_img,
        query_mask_paths=query_masks,
        ckpt_path=args.load,
        save_pred_paths=save_pred_paths,
        sam2_backbone_size=args.sam2_backbone_size,
        dinov2_backbone_size=args.dinov2_backbone_size,
        size=args.size,
        class_id=args.class_id,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        normalize=args.normalize,
    )

    print("=" * 50)
    print("Demo inference done.")
    for item in result["items"]:
        print(f"{item['query_name']} -> {item['pred_mask_path']}")


if __name__ == "__main__":
    main()
