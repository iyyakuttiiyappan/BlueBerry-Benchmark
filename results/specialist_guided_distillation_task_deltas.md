| method                                         | display_name                        | task           | metric          |   score |   specialist_score |   delta_vs_specialist_percent |
|:-----------------------------------------------|:------------------------------------|:---------------|:----------------|--------:|-------------------:|------------------------------:|
| integrated_specialists                         | Integrated best specialists         | detection      | mAP50           |  0.4161 |             0.4161 |                        0      |
| integrated_specialists                         | Integrated best specialists         | segmentation   | foreground mIoU |  0.4873 |             0.4873 |                        0      |
| integrated_specialists                         | Integrated best specialists         | counting       | MAE             | 13.8584 |            13.8584 |                        0      |
| integrated_specialists                         | Integrated best specialists         | classification | macro F1        |  0.7254 |             0.7254 |                        0      |
| berrymtl_centerdet_hitile_quality              | BerryMTL-HiTile-QualityDet          | detection      | mAP50           |  0.3364 |             0.4161 |                      -19.1485 |
| berrymtl_centerdet_hitile_quality              | BerryMTL-HiTile-QualityDet          | segmentation   | foreground mIoU |  0.4608 |             0.4873 |                       -5.4415 |
| berrymtl_centerdet_hitile_quality              | BerryMTL-HiTile-QualityDet          | counting       | MAE             |  9.1845 |            13.8584 |                       33.7257 |
| berrymtl_centerdet_hitile_quality              | BerryMTL-HiTile-QualityDet          | classification | macro F1        |  0.678  |             0.7254 |                       -6.5471 |
| berrymtl_teacher_aligned_det                   | BerryMTL-TeacherAlignedDet          | detection      | mAP50           |  0.3688 |             0.4161 |                      -11.3652 |
| berrymtl_teacher_aligned_det                   | BerryMTL-TeacherAlignedDet          | segmentation   | foreground mIoU |  0.4691 |             0.4873 |                       -3.7304 |
| berrymtl_teacher_aligned_det                   | BerryMTL-TeacherAlignedDet          | counting       | MAE             |  9.9356 |            13.8584 |                       28.3061 |
| berrymtl_teacher_aligned_det                   | BerryMTL-TeacherAlignedDet          | classification | macro F1        |  0.6647 |             0.7254 |                       -8.3741 |
| berrymtl_specialist_guided_distill             | BerryMTL-SpecialistGuidedDistill    | detection      | mAP50           |  0.3539 |             0.4161 |                      -14.9515 |
| berrymtl_specialist_guided_distill             | BerryMTL-SpecialistGuidedDistill    | segmentation   | foreground mIoU |  0.4615 |             0.4873 |                       -5.2871 |
| berrymtl_specialist_guided_distill             | BerryMTL-SpecialistGuidedDistill    | counting       | MAE             |  9.2724 |            13.8584 |                       33.0919 |
| berrymtl_specialist_guided_distill             | BerryMTL-SpecialistGuidedDistill    | classification | macro F1        |  0.6726 |             0.7254 |                       -7.2854 |
| berrymtl_specialist_adapter_fusion             | BerryMTL-SpecialistAdapterFusion    | detection      | mAP50           |  0.3718 |             0.4161 |                      -10.6352 |
| berrymtl_specialist_adapter_fusion             | BerryMTL-SpecialistAdapterFusion    | segmentation   | foreground mIoU |  0.4726 |             0.4873 |                       -3.0089 |
| berrymtl_specialist_adapter_fusion             | BerryMTL-SpecialistAdapterFusion    | counting       | MAE             |  9.4754 |            13.8584 |                       31.627  |
| berrymtl_specialist_adapter_fusion             | BerryMTL-SpecialistAdapterFusion    | classification | macro F1        |  0.6707 |             0.7254 |                       -7.5498 |
| berrymtl_specialist_adapter_fusion_uncertainty | BerryMTL-SpecialistAdapterFusion-UW | detection      | mAP50           |  0.3714 |             0.4161 |                      -10.7417 |
| berrymtl_specialist_adapter_fusion_uncertainty | BerryMTL-SpecialistAdapterFusion-UW | segmentation   | foreground mIoU |  0.4731 |             0.4873 |                       -2.905  |
| berrymtl_specialist_adapter_fusion_uncertainty | BerryMTL-SpecialistAdapterFusion-UW | counting       | MAE             |  9.4611 |            13.8584 |                       31.7297 |
| berrymtl_specialist_adapter_fusion_uncertainty | BerryMTL-SpecialistAdapterFusion-UW | classification | macro F1        |  0.6751 |             0.7254 |                       -6.941  |