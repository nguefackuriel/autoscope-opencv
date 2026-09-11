"""The Faster R-CNN model, in one place so training and scoring build it alike.

THE ANCHORS ARE THE WHOLE GAME HERE

torchvision's default FPN anchors are (32, 64, 128, 256, 512) px, chosen for
COCO, where objects are a good fraction of the frame. A parasite is about 10 px.
With the stock anchors the smallest box the network can propose is three times
larger than anything it is looking for, every anchor overlaps every parasite at
an IoU near zero, the RPN labels them all background, and training converges to
a model that detects nothing while the loss curve looks healthy.

So the anchors are (4, 8, 16, 32, 64), which straddles the 6 to 40 px range
boxes.py actually produces. If the export's reported box sizes move, these move
with them.
"""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

# One size per FPN level, smallest level first. Matched to the box side
# distribution the export prints (p10 / median / p90).
ANCHOR_SIZES = ((4,), (8,), (16,), (32,), (64,))
ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)


def build_model(num_classes: int = 2, pretrained: bool = True,
                tile: int = 640, detections_per_tile: int = 300):
    """Faster R-CNN with a ResNet50-FPN backbone, re-anchored for small objects.

    num_classes counts the background, so parasite-only detection is 2.

    `detections_per_tile` is raised from the stock 100 because a dense field
    tile can hold far more candidate parasites than a COCO photo holds objects,
    and a cap that bites would silently truncate recall on exactly the heavily
    parasitised slides that matter most.
    """
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        rpn_anchor_generator=AnchorGenerator(sizes=ANCHOR_SIZES,
                                             aspect_ratios=ASPECT_RATIOS),
        # The tiles are already 640 at native resolution. Letting the transform
        # resize them to its default 800 would rescale the parasites and undo
        # the reason for tiling in the first place.
        min_size=tile,
        max_size=tile,
        box_detections_per_img=detections_per_tile,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def describe(model) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Faster R-CNN ResNet50-FPN, anchors {ANCHOR_SIZES}, {n/1e6:.1f}M trainable"
