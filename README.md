# ArmorDetector

`ArmorDetector` 当前保持原有对外消息契约不变，但内部已经收敛成一条明确的四阶段链路：

1. `CameraFrameSync` 提供同步后的图像帧
2. `OpenVINO` 模型直接输出颜色、编号和四角点
3. 可选传统灯条细化只负责角点修正
4. `PnP` 负责位姿估计并填充检测结果

## Runtime Role

- 输入: `CameraFrameSync<Info>::SyncedFrame`
- 输出: `armor_detector/armors_result`、`armor_detector/metrics`
- 模型:
  - `model/yolov5.xml` + `model/yolov5.bin`

## Public Contract

- `armors_result` 仍然发布 `ArmorDetectionsMessage`
- `metrics` 仍然发布 `ArmorDetectorMetrics`
- 下游 `ArmorTracker`、录像器、truth publisher 不需要跟着这次内部清理一起改

## Internal Notes

- 不再需要独立 `NumberClassifier`
- `PnPSolver` 现在直接吃编译期 `CameraInfo`
- detector 解码阶段不再使用并行数组和散落的输出列号

## Debug

- 打开预览: 设置 `armor_detector.debug.preview: true`
- 可选项:
  - `armor_detector.debug.show_binary`
  - `armor_detector.yolo.use_roi`
  - `armor_detector.yolo.use_traditional_refine`
