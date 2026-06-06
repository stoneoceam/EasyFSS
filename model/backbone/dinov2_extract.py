import torch
import torch.nn as nn


class DinoV2FeatureExtractor(nn.Module):
    def __init__(self, model_name=None, hub_dir=None) -> None:
        super().__init__()
        if hub_dir is None:
            print("No hub_dir specified, using default")
            hub_dir = 'dinov2'
        if model_name is None:
            print("No model_name specified, using default")
            model_name = 'dinov2_vitb14'
        model = torch.hub.load(hub_dir, model_name, source='local', pretrained=False)
        model.load_state_dict(torch.load(f'checkpoints/{model_name}_pretrain.pth', weights_only=False))

        self.dino_encoder = model

        for param in self.dino_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        vitb14
        :param x: [b,3,H,W]
        :return:
            final_feature: [b,768,H/14,W/14]
            features: [[b,768,H/14,W/14]*12]
        """

        features = self.dino_encoder.get_intermediate_layers(x, 12, reshape=True)
        final_feature = features[-1]

        return final_feature, features
