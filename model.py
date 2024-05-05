import torchvision
import torch

from functools import partial
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

def create_model(num_classes):

    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    )

    num_anchors = model.head.classification_head.num_anchors
    
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    return model

if __name__ == '__main__':
    fruit_model = create_model(6)
    print(fruit_model)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in fruit_model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in fruit_model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")