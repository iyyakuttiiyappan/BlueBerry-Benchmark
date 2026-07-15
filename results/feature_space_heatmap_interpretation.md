# Feature-Space and Heatmap Interpretation

Source unified run: `outputs\fresh_benchmark_514\07_cross_task_analysis\ours\runs\20260705_142348_berrymtl_specialist_adapter_fusion_uncertainty_seed42`

The unified feature-space heatmap is a representation-similarity plot, not an accuracy table. Higher off-diagonal CKA means two task heads preserve similar geometry from the shared encoder. This is the visual argument for the unified model: the four outputs are not four unrelated pipelines; they reuse berry-aware evidence before specializing.

The specialist reference heatmap is intentionally diagonal because four individual models have no train-time shared task representation. That is useful as a contrast, but it should be described as a structural reference, not as empirical CKA between specialist backbones.

Key CKA values from the unified checkpoint: segmentation-counting=0.687, detection-classification=0.180, detection-counting=0.730.

## Why counting can work when detection is weaker

Detection is an instance-level decision: every berry needs a sufficiently accurate box, class score, and non-suppressed peak. Dense clusters, occlusion, adjacent berries, and class-score/localization mismatch can hurt mAP even when the model sees the berry mass.

Counting is a global/dense task: the model can integrate distributed foreground and density evidence over a cluster. It does not need every berry to survive NMS as a separate high-confidence box. Therefore, a model can underperform on class-sensitive detection mAP while still producing a strong count, especially in crowded blueberry clusters.

Use the montage as follows:

1. Unified detection heatmaps show where the model forms discrete berry peaks.
2. Unified density maps show broader count evidence over berry-heavy regions.
3. Unified segmentation and specialist masks show that dense foreground support remains strong.
4. When density/foreground covers the cluster but detections are sparse or misclassified, the count can remain close to the true count while mAP drops.
