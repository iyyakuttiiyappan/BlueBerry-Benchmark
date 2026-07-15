| task           | metric          | higher_is_better   | specialist_method              | specialist_display_name               |   specialist_score |
|:---------------|:----------------|:-------------------|:-------------------------------|:--------------------------------------|-------------------:|
| detection      | map50           | True               | fasterrcnn_resnet50_fpn_thr005 | Faster R-CNN ResNet-50 FPN (thr=0.05) |           0.416088 |
| segmentation   | miou_foreground | True               | fpn_convnextv2_tiny_tta        | FPN ConvNeXtV2-T + Flip TTA           |           0.487271 |
| counting       | mae             | False              | count_efficientnetv2_s         | EfficientNetV2-S Regression           |          13.8584   |
| classification | macro_f1        | True               | resnet50                       | ResNet-50                             |           0.725447 |