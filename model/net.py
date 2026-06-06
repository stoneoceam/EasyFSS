import os

import hydra
import torch
import torch.nn.functional as F
from torch import nn

from model.backbone import dinov2_extract
from model.base.conv4d import CenterPivotConv4d, DWConv4d, PWConv4d
from model.base.correlation import Correlation
from sam2.build_sam import build_sam2
from sam2.modeling.position_encoding import PositionEmbeddingSine


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
                :param num_masks:
    """
    inputs, targets = inputs.flatten(1), targets.flatten(1)
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1 + 1e-6)
    return loss.sum() / num_masks


class Adapter(nn.Module):
    def __init__(self):
        super(Adapter, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        )

        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.block(x) * self.alpha
        return x


class Encoder(nn.Module):
    def __init__(self, inch):
        super(Encoder, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        def make_building_block_dwconv(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(DWConv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(PWConv4d(inch, outch))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 48, 64, 128

        # Squeezing building blocks
        self.layer1 = make_building_block(inch, [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])

        # Mixing building blocks
        self.layer2 = make_building_block_dwconv(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.layer3 = make_building_block_dwconv(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        outch1, outch2, outch3 = 16, 64, 128

        self.layer1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                    nn.ReLU())

        self.layer2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(outch2, 1, (3, 3), padding=(1, 1), bias=True))

    def forward(self, x):
        x = self.layer1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.layer2(x)
        return x


class DPHB(nn.Module):
    def __init__(self, inch, output_size):
        super(DPHB, self).__init__()
        self.encoder = Encoder(inch)
        self.decoder = Decoder()
        self.output_size = output_size

    def forward(self, query_feats, support_feats, support_mask):

        support_fg_feats = self.apply_mask_to_feat_list(support_feats, support_mask)
        support_bg_feats = self.apply_mask_to_feat_list(support_feats, 1 - support_mask)

        corr1 = Correlation.multilayer_correlation(query_feats, support_fg_feats)  # [b,l,h,w,h,w]
        corr2 = Correlation.multilayer_correlation(query_feats, support_bg_feats)  # [b,l,h,w,h,w]

        corr = torch.cat((corr1, corr2), dim=1)

        x = self.encoder(corr)  # x:[b,128,30,30,4,4]
        bsz, ch, ha, wa, hb, wb = x.size()
        x = x.view(bsz, ch, ha, wa, -1).mean(dim=-1)
        mask = self.decoder(x)
        mask = F.interpolate(mask, size=self.output_size, mode='bilinear')
        return mask

    def apply_mask_to_feat_list(self, feat_list, mask):

        out_list = []
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [b, 1, h1, w1]
        for feat in feat_list:
            if mask.shape[2:] != feat.shape[2:]:
                mask_resized = F.interpolate(mask, size=feat.shape[2:], mode='bilinear', align_corners=False)
            else:
                mask_resized = mask
            out_list.append(feat * mask_resized)
        return out_list


class SPG(nn.Module):
    def __init__(self, embed_dim, pool_size, image_size):
        super(SPG, self).__init__()

        self.embed_dim = embed_dim
        self.pool_size = pool_size
        self.image_size = image_size

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.branch_local = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.branch_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size * 2),
            nn.Conv2d(32, 64, kernel_size=1)
        )

        self.proj = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=1)
        )

        self.pos_embed = PositionEmbeddingSine(
            num_pos_feats=embed_dim,
            normalize=True
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, query_feat, support_feat, support_mask):
        p_fg, p_bg = self.generate_prior(query_feat, support_feat, support_mask.unsqueeze(1), self.image_size)
        p_fbg = self.func(p_fg, p_bg, self.image_size)

        x = self.stem(p_fbg)  # [b,1,256,256] --> [b,32,128,128]

        x_local = self.branch_local(x)  # [b,32,128,128] --> [b,64,64,64]

        x_global = F.interpolate(self.branch_global(x), size=x_local.shape[2:], mode='bilinear',
                                 align_corners=False)  # [b,32,128,128] --> [b,64,64,64]
        x = self.proj(x_local + x_global)  # [b,64,64,64] --> [b,256,64,64]

        x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))  # [b,256,1,1]

        pos = self.pos_embed(x)
        x = x + pos

        x = x.flatten(2).transpose(1, 2)

        x = x * self.alpha.tanh()
        return x

    def mask_norm(self, mask, cosine_eps=1e-7):
        b, c, h, w = mask.shape

        mask = mask.max(1)[0].view(b, h * w)

        mask = (mask - mask.min(1)[0].unsqueeze(1)) / (
                mask.max(1)[0].unsqueeze(1) - mask.min(1)[0].unsqueeze(1) + cosine_eps)

        mask = mask.view(b, 1, h, w)

        return mask

    def cos_sim(self, query_feat_high, tmp_supp_feat, cosine_eps=1e-7):
        q = query_feat_high.flatten(2).transpose(-2, -1)
        s = tmp_supp_feat.flatten(2).transpose(-2, -1)

        tmp_query = q
        tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s
        tmp_supp = tmp_supp.contiguous()
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = (tmp_supp @ tmp_query) / (tmp_supp_norm @ tmp_query_norm + cosine_eps)
        return similarity

    def generate_prior(self, query_feat, support_feat, support_mask, image_size, normalize=False):
        def Weighted_GAP(supp_feat, mask):
            supp_feat = supp_feat * mask
            feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
            area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
            supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
            return supp_feat

        b, c, h, w = support_feat.shape
        cosine_eps = 1e-7

        feat_size = support_feat.size()[2:]
        temp_support_mask = F.interpolate(support_mask, size=feat_size, mode='bilinear', align_corners=True)

        fg_support_feat = Weighted_GAP(support_feat, temp_support_mask)
        bg_support_feat = Weighted_GAP(support_feat, 1 - temp_support_mask)

        fg_sim = self.cos_sim(query_feat, fg_support_feat, cosine_eps)
        bg_sim = self.cos_sim(query_feat, bg_support_feat, cosine_eps)

        # norm
        fg_sim = self.mask_norm(fg_sim.view(b, 1, h, w))
        bg_sim = self.mask_norm(bg_sim.view(b, 1, h, w))

        fg_sim = F.interpolate(fg_sim, size=image_size, mode='bilinear', align_corners=True)
        bg_sim = F.interpolate(bg_sim, size=image_size, mode='bilinear', align_corners=True)

        return fg_sim, bg_sim

    def func(self, fg_mask, bg_mask, image_size):
        b, c, h, w = fg_mask.shape
        cosine_eps = 1e-7

        ae_mask = fg_mask - bg_mask
        ae_mask = F.relu(ae_mask)

        ae_mask = ae_mask.view(b, -1)
        ae_mask = (ae_mask - ae_mask.min(1)[0].unsqueeze(1)) / (
                ae_mask.max(1)[0].unsqueeze(1) - ae_mask.min(1)[0].unsqueeze(1) + cosine_eps)
        ae_mask = ae_mask.view(b, 1, h, w)
        ae_mask = F.interpolate(ae_mask, size=image_size, mode='bilinear', align_corners=True)

        return ae_mask


class Net(nn.Module):
    def __init__(self,
                 sam2_backbone_size='base',
                 dinov2_backbone_size='base',
                 ):
        super(Net, self).__init__()

        # SAM2 Init
        work_dir = os.getcwd()
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize_config_dir(version_base="1.3.2",
                                    config_dir=work_dir)

        if sam2_backbone_size == 'large':
            config_file = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
            ckpt_path = "checkpoints/sam2.1_hiera_large.pt"
        elif sam2_backbone_size == 'base':
            config_file = "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
            ckpt_path = "checkpoints/sam2.1_hiera_base_plus.pt"
        elif sam2_backbone_size == 'small':
            config_file = "sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
            ckpt_path = "checkpoints/sam2.1_hiera_small.pt"
        elif sam2_backbone_size == 'tiny':
            config_file = "sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
            ckpt_path = "checkpoints/sam2.1_hiera_tiny.pt"
        else:
            raise ValueError('Unsupported sam2 backbone size')

        self.sam2 = build_sam2(
            config_file=config_file,
            ckpt_path=ckpt_path
        )
        del self.sam2.memory_encoder
        del self.sam2.memory_attention
        del self.sam2.obj_ptr_tpos_proj
        del self.sam2.obj_ptr_proj
        for param in self.sam2.parameters():
            param.requires_grad = False

        # DinoV2 Init
        if dinov2_backbone_size == 'large':
            dino_model_name = 'dinov2_vitl14'
        elif dinov2_backbone_size == 'base':
            dino_model_name = 'dinov2_vitb14'
        elif dinov2_backbone_size == 'small':
            dino_model_name = 'dinov2_vits14'
        else:
            raise ValueError('Unsupported dinov2 backbone size')

        # Backbone Feature Extractor
        self.backbone_dino = dinov2_extract.DinoV2FeatureExtractor(dino_model_name, 'dinov2')

        self.DPHB = DPHB(24, (256, 256))

        self.adapter = Adapter()

        self.SPG = SPG(256, 1, (256, 256))

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self,
                query_img,
                query_mask,
                query_name,
                support_img,
                support_mask):
        # dinoV2
        query_image_dino = (query_img - self.mean.to(query_img.device)) / self.std.to(query_img.device)
        support_image_dino = (support_img - self.mean.to(support_img.device)) / self.std.to(support_img.device)

        query_feat_dino, query_feats_dino = self.backbone_dino(query_image_dino)
        support_feat_dino, support_feats_dino = self.backbone_dino(support_image_dino)

        # sam2
        query_image_sam = F.interpolate(query_img, size=(1024, 1024), mode='bilinear', align_corners=True)
        query_image_sam = (query_image_sam - self.mean.to(query_image_sam.device)) / self.std.to(
            query_image_sam.device)

        query_feats_sam = self.sam2.forward_image(query_image_sam)

        coarse_mask = self.DPHB(query_feats_dino, support_feats_dino, support_mask)

        # mask encoding
        sparse_embeddings, dense_embeddings = self.sam2.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=coarse_mask.float(),
            batch_size=support_mask.size(0),
        )

        adapte_dense_embeddings = self.adapter(coarse_mask)

        adapte_sparse_embeddings = self.SPG(query_feat_dino, support_feat_dino, support_mask)

        new_sparse_embeddings = torch.cat([sparse_embeddings, adapte_sparse_embeddings], dim=1)
        new_dense_embeddings = dense_embeddings + adapte_dense_embeddings

        # mask decoding
        mask, iou_pred, sam_tokens_out, object_score_logits = self.sam2.sam_mask_decoder(
            image_embeddings=query_feats_sam['vision_features'],
            image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=new_sparse_embeddings,
            dense_prompt_embeddings=new_dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=query_feats_sam['backbone_fpn'][:-1],
        )

        coarse_mask = F.interpolate(coarse_mask, size=query_mask.size()[1:], mode='bilinear', align_corners=True)
        mask = F.interpolate(mask, size=query_mask.size()[1:], mode='bilinear', align_corners=True)

        # loss
        dice_weight = 1.0
        bce_weight = 1.0

        loss1 = self.compute_objective(coarse_mask, query_mask, bce_weight, dice_weight)
        loss2 = self.compute_objective(mask, query_mask, bce_weight, dice_weight)
        loss = loss1 + loss2

        return loss, coarse_mask, mask

    def predict_mask_nshot(self, batch, nshot):

        all_shots_preds = []

        for s_idx in range(nshot):
            outputs = self(
                batch['query_img'],
                batch['query_mask'],
                batch['query_name'],
                batch['support_imgs'][:, s_idx],
                batch['support_masks'][:, s_idx],
            )

            preds = outputs[1:]
            all_shots_preds.append(preds)

        final_results = []

        for pred_group in zip(*all_shots_preds):
            logit_mask = torch.cat(pred_group, dim=1).mean(1, keepdim=True)
            final_results.append(logit_mask)

        return tuple(final_results)

    def compute_objective(self, logit_mask, gt_mask, bce_weight=1.0, dice_weight=1.0):
        bsz = logit_mask.size(0)
        gt_mask = gt_mask.unsqueeze(1)
        loss1 = F.binary_cross_entropy_with_logits(logit_mask, gt_mask)
        loss2 = dice_loss(logit_mask, gt_mask, bsz)

        return bce_weight * loss1 + dice_weight * loss2

    def train_mode(self):
        self.train()
