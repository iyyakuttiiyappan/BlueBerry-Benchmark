# Fresh Benchmark Qualitative and Failure Analysis

## Dataset Statistics

- Images: 514
- Instances/crops: 24336
- Mean berries per image: 58.75
- Median berries per image: 31
- Maximum berries in one image: 346

## Best-model Qualitative Summaries

- Classification: ResNet-50 macro F1 0.7254; 291 test crop errors.
- Counting: BerryMTL-HiTile-QualityDet (ours) MAE 9.1845; worst absolute error 66.00.
- Detection: Faster R-CNN ResNet-50 FPN (thr=0.05) mAP50 0.4161; mean image-level FP+FN 61.34.
- Segmentation: FPN ConvNeXtV2-T + Flip TTA foreground mIoU 0.4873; worst image foreground IoU 0.5088.

## Figure Folders

- Dataset statistics: `outputs\fresh_benchmark_514\07_cross_task_analysis\figures\dataset_statistics`
- Qualitative examples: `outputs\fresh_benchmark_514\07_cross_task_analysis\figures\qualitative`
- Failure analysis: `outputs\fresh_benchmark_514\07_cross_task_analysis\figures\failure_analysis`

## Table Folder

- Tables: `outputs\fresh_benchmark_514\07_cross_task_analysis\tables`