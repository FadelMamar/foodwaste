import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
from torchvision import transforms as T
import timm
from .dataset import id2label
from .config import MainConfig

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=14, tokenH=14, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        return self.classifier(embeddings)


class Dinov2ForSemanticSegmentation(nn.Module):
  def __init__(self, config:MainConfig):
    super().__init__()

    #self.dinov2 = Dinov2Model(config)
    self.model = timm.create_model(
                config.model.model_name, pretrained=True, num_classes=0
            )
    data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)
    self.transform = nn.Sequential(*[t for t in transform.transforms if isinstance(t, (T.Normalize, T.Resize))])
    self.classifier = LinearClassifier(self.model.num_features, config.training.image_size//14, config.training.image_size//14, len(id2label))

    if config.model.freeze_backbone:
        for  param in self.model.parameters():
            param.requires_grad = False

  def forward(self, pixel_values:torch.Tensor):
    # use frozen features, so we exclude the CLS token
    patch_embeddings = self.model.forward_features(pixel_values)[:,1:,:]

    # convert to logits and upsample to the size of the pixel values
    logits = self.classifier(patch_embeddings)
    logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

    return SemanticSegmenterOutput(
        logits=logits,
    )