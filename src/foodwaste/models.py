import torch
import torch.nn as nn
from transformers.modeling_outputs import SemanticSegmenterOutput
from torchvision import transforms as T
import timm
from torchvision.transforms import functional as TF

from .config import MainConfig,id2label

from logging import getLogger
from typing import Optional
import traceback

LOGGER = getLogger(__name__)

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, len(id2label), (1,1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        return self.classifier(embeddings)

class Dinov2ForSemanticSegmentation(nn.Module):

    def __init__(self, config:MainConfig):
        super().__init__()

        self.model = timm.create_model(
                    config.model.model_name, pretrained=True, num_classes=0
                )
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)
        trfs_norm = [t for t in transform.transforms if isinstance(t, T.Normalize)]
        trfs_resize = [t for t in transform.transforms if isinstance(t, T.Resize)]

        assert config.training.image_size % config.model.patch_size == 0, "Image size must be divisible by patch size"
        self.token_w = config.training.image_size // config.model.patch_size
        self.token_h = config.training.image_size // config.model.patch_size

        try:
            self.model.set_input_size((config.training.image_size,config.training.image_size))
            trfs_resize = [T.Resize((config.training.image_size,config.training.image_size))]
            LOGGER.info(f"Set model input size to {config.training.image_size}")
        except:
            LOGGER.warning("Could not set model input size, using default")
        
        trfs = trfs_resize + trfs_norm
        self.transform = nn.Sequential(*trfs)
        self.classifier = LinearClassifier(self.model.num_features, self.token_w, self.token_h)

        if config.model.freeze_backbone:
            for  param in self.model.parameters():
                param.requires_grad = False
        
        self.start_layer = 1
        if config.model.use_cls_token:
            self.start_layer = 0

    def forward_intermediates(self, pixel_values:torch.Tensor,layer:int=12):
        # Check if the model has forward_intermediates method
        if hasattr(self.model, 'forward_intermediates'):
            return self.model.forward_intermediates(pixel_values,indices=[layer])
        else:
            features = self.model.forward_features(pixel_values)
            return features, [features]

    def forward(self, pixel_values:torch.Tensor,mask:Optional[torch.Tensor]=None):
        # use frozen features, so we exclude the CLS token
        patch_embeddings = self.model.forward_features(pixel_values)[:,self.start_layer:,:]
        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)
        return logits

class PatchQuant(torch.nn.Module):
    def __init__(self,image_size:int,patch_size:int,layer:int=12):
        super().__init__()
        self.patch_size = patch_size
        self.quant_filter = torch.nn.Conv2d(1, 1, patch_size, stride=patch_size, bias=False)
        self.quant_filter.weight.data.fill_(1.0 / (patch_size * patch_size))
        self.quant_filter.requires_grad = False
        self.layer = layer
        self.image_size = image_size
        
        assert isinstance(layer, int), f"Received {layer}"
        assert image_size % patch_size == 0, f"Image size {image_size} must be divisible by patch size {patch_size}"

        self.h_patches = int(self.image_size / self.patch_size)
        self.w_patches = self.h_patches
        self.num_patches = self.h_patches * self.w_patches

    
    def forward(self,image:torch.Tensor,model:Dinov2ForSemanticSegmentation,mask:Optional[torch.Tensor]=None)->tuple[torch.Tensor,Optional[torch.Tensor]]:
        
        # process image
        with torch.no_grad():
            out, intermediates = model.forward_intermediates(self.resize(image),layer=self.layer)      
        feats = intermediates[0]
        b,dim,num_patch_h,num_patch_w = feats.shape
        xs = feats.squeeze().view(b,dim, -1).permute(0,2,1).detach()
        
        # process mask
        if mask is not None:
            mask_resized= self.resize(mask.unsqueeze(1))
            ys = self.quant_filter(mask_resized.float()).squeeze().view(b,-1).detach()
            
            # keeping only the patches that have clear positive or negative label
            fg_mask = (ys < 0.01) | (ys > 0.99)
            return xs,(ys*fg_mask).round().long()
        return xs,None
            

    def resize(
        self,
        image:torch.Tensor,
    ) -> torch.Tensor:
        resized_image =  TF.resize(image, (self.h_patches * self.patch_size, self.w_patches * self.patch_size))
        return resized_image

class PatchClassifier(Dinov2ForSemanticSegmentation):

    def __init__(self, config:MainConfig):
        super().__init__(config)
        
        num_features = self.model.num_features
        self.num_labels = len(id2label)
        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_features//2),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(num_features//2, self.num_labels)
        )
        
        try:
            self.patch_quant = PatchQuant(patch_size=config.model.patch_size,
                                          layer=config.model.model_layer,
                                          image_size=config.training.image_size)
            LOGGER.info(f"PatchQuant created successfully: h_patches={self.patch_quant.h_patches}, w_patches={self.patch_quant.w_patches}")
        except Exception:
            LOGGER.error(f"Error creating PatchQuant: {traceback.format_exc()}")
            raise
        
        if config.model.freeze_backbone:
            for  param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, pixel_values:torch.Tensor,mask:Optional[torch.Tensor]=None)->tuple[torch.Tensor,Optional[torch.Tensor]]:
        xs,ys = self.patch_quant(pixel_values,self,mask)
        logits = self.classifier(xs)
        logits = (logits
                  .view(-1,self.patch_quant.h_patches,
                        self.patch_quant.w_patches,
                        self.num_labels)
                  .permute(0,3,1,2)
                  )
        if ys is not None:
            ys = ys.view(-1,self.patch_quant.h_patches,
                          self.patch_quant.w_patches,
                          )
            
        return logits,ys
    
    