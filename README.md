# ArmorDetector

`ArmorDetector` 现在使用 `YOLOv5 + OpenVINO` 作为主检测链路，并保留传统灯条几何筛选作为可选细化步骤。

## Runtime Role

- 输入: 相机图像
- 输出: `armor_detector/results`、`armor_detector/metrics`
- 模型:
  - `model/yolov5.xml` + `model/yolov5.bin`
  - `model/mlp.onnx`，内容已替换为 `sp_vision` 使用的分类模型

## Debug

- 打开预览: 设置 `armor_detector.debug.preview: true`
- 可选项:
  - `armor_detector.debug.show_binary`
  - `armor_detector.yolo.use_roi`
  - `armor_detector.yolo.use_traditional_refine`
