# Protocol Audit

- Split seed: `42`.
- Training seed: `42`.
- All rows use the locked train/validation/test split stored in `outputs/fresh_benchmark/01_annotations/image_manifest.csv`.
- Important remaining differences: specialist detectors use torchvision two-stage/RetinaNet heads at 768 px, while unified CenterDet variants use a ConvNeXtV2 shared backbone at 512 px; the best specialist segmentation result uses flip TTA.
