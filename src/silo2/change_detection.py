import logging
import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger("urbanfl.silo2.vision")

class SiameseChangeDetector(nn.Module):
    """
    Siamese network using pretrained ResNet backbones for pre/post disaster assessment.
    Optimized for xBD satellite pairing to generate pixel-wise binary change masks.
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        # Use lightweight ResNet18 for rapid inference on standard GPUs
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=weights)
        
        # Extract features (remove fully connected/pooling layers)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        
        # Decoder for generating segmentation masks
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512 * 2, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        )
        logger.info(f"SiameseChangeDetector initialized. Pretrained={pretrained}")

    def forward(self, img_pre: torch.Tensor, img_post: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_pre: Pre-disaster imagery tensor (B, 3, H, W)
            img_post: Post-disaster imagery tensor (B, 3, H, W)
        Returns:
            Logits for change mask (B, 1, H, W) -> passed to BCEWithLogitsLoss
        """
        # Feature extraction (shared weights)
        feat_pre = self.encoder(img_pre)
        feat_post = self.encoder(img_post)
        
        # Concatenate features representing the 'change'
        combined_feat = torch.cat([feat_pre, feat_post], dim=1) # (B, 1024, h, w)
        
        # Decode to original resolution mask
        mask_logits = self.decoder(combined_feat)
        
        # Ensure output is interpolated precisely if dimensions shift
        if mask_logits.shape[-2:] != img_pre.shape[-2:]:
            mask_logits = nn.functional.interpolate(
                mask_logits, size=img_pre.shape[-2:], mode='bilinear', align_corners=False
            )
            
        return mask_logits
