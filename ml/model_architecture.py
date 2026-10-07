import torch
import torch.nn as nn
import timm

class FrequencyBranch(nn.Module):
    """Lightweight CNN to process the 1-channel High-Pass Frequency tensor."""
    def __init__(self):
        super(FrequencyBranch, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
    def forward(self, x):
        return self.conv_layers(x)

class DualStreamDeepfakeDetector(nn.Module):
    """
    Combines an Xception RGB backbone with the High-Pass Frequency branch.
    Replaces the original ImageNet head with a custom Fusion Classifier.
    """
    def __init__(self, model_name='xception', pretrained=True):
        super(DualStreamDeepfakeDetector, self).__init__()
        
        # 1. RGB Spatial Stream (Transfer Learning: Pretrained weights, chopped head)
        self.rgb_backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        self.rgb_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dynamically get the feature size (Xception outputs 2048 features)
        rgb_feature_dim = self.rgb_backbone.num_features 
        
        # 2. High-Frequency Noise Stream
        self.freq_branch = FrequencyBranch()
        freq_feature_dim = 128
        
        # 3. Custom Fusion Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(rgb_feature_dim + freq_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), # Increased dropout slightly for the heavier Xception model
            nn.Linear(512, 1) # Single node for Binary Cross Entropy (Real/Fake)
        )
        
        # Placeholders for XAI (Grad-CAM++)
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, rgb_img, hp_img, return_cam_features=False):
        feat_maps = self.rgb_backbone(rgb_img)
        
        if return_cam_features and feat_maps.requires_grad:
            feat_maps.register_hook(self.save_gradient)
            self.activations = feat_maps
            
        rgb_feats = self.rgb_pool(feat_maps).flatten(1)
        freq_feats = self.freq_branch(hp_img)
        
        combined = torch.cat([rgb_feats, freq_feats], dim=1)
        logits = self.classifier(combined)
        
        return logits