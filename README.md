# ArmorDetector

`ArmorDetector` 从 `CameraFrameSync` 读取同步后的图像和 IMU，使用 OpenVINO 模型检测装甲板四角点，再根据相机内参求出装甲板在相机坐标系下的位姿。模块输出会保留当前同步帧引用，供 `ArmorTracker` 在同进程回调里立刻使用。

## 数据流

1. 读取 `CameraFrameSync<Info>::SyncedFrame`。
2. 将图像缩放到模型输入尺寸，并从 BGR 转成 RGB。
3. 执行 OpenVINO 推理，解码颜色、编号、置信度和四角点。
4. 过滤低置信度、颜色不匹配、编号无效或几何异常的候选。
5. 对有效候选执行 PnP，发布 `armor_detector/armors_frame`。

## 输入输出

输入：

- `CameraFrameSync<Info>::SyncedFrame`
- `host/robot_game_ref`，仅在 `referee_auto_detect_color: true` 时订阅

输出：

- `armor_detector/armors_frame`：本帧检测结果、图像时间戳、当前同步帧引用

`armors_frame` 里的图像和 IMU 指针只在本次回调期间有效。消费模块必须在回调内完成读取，不能跨帧保存这些指针。

## 模型

默认模型文件：

- `model/armor_detector_640x512.onnx`
- `model/armor_number_mlp.onnx`

主检测模型输入为 `640x512` RGB 图像，输出为 `20160x22` 候选矩阵。候选字段包含 objectness、四角点、颜色和编号类别。解码后会把网络输入坐标映射回原始相机图像坐标。

数字 refine 模型使用 OpenCV DNN CPU 后处理。它只在检测候选已经成立后运行，用数字 ROI 的分类结果覆盖 detector 编号；置信度不够或尺寸类型冲突时不会覆盖。

Hailo 路线现在只通过固定 `model_name` 选择。当前临时接受的 `6` 个名字是：

- `int8-head-l`
- `int8-grid-l`
- `int16-head-l`
- `int8-head`
- `int8-grid`
- `int16-head`

这些变体都已经映射到模块目录下的稳定工件文件名：

- `model/skd_int8_head_l.hef`
- `model/skd_int8_grid_l.hef`
- `model/szu_int16_head_l.hef`
- `model/skd_int8_head.hef`
- `model/skd_int8_grid.hef`
- `model/szu_int16_head.hef`

其中：

- `int8-head*`：SKD 六输出 host-tail 语义
- `int8-grid*`：SKD 单输出 `21x6720` 语义
- `int16-head*`：SZU 三头 `conv47/54/60` 语义

`int16-box*` 当前不在接受集合里，不建议配置进当前 detector。

## 结果内容

单个装甲板结果包含：

- 颜色、编号、尺寸类型和置信度
- 图像包围盒、中心点和四个角点
- PnP 是否成功
- PnP 平均重投影误差
- OpenCV 相机坐标系下的装甲板位姿

相机坐标系沿用 OpenCV 约定：`x` 向右，`y` 向下，`z` 向前。这里的位姿只描述装甲板相对相机的位置和朝向，不包含 IMU 姿态融合。

## 配置

- `detect_color`：`0` 只保留红色，`1` 只保留蓝色，其他值不过滤颜色。
- `referee_auto_detect_color`：按裁判系统 robot_id 自动切换敌方颜色。
- `referee_domain`：裁判系统摘要包所在 topic domain，默认 `host`。
- `referee_topic`：裁判系统摘要包 topic 名，默认 `robot_game_ref`。
- `network.min_confidence`：最终结果置信度门限。
- `network.logit_threshold`：网络 objectness 原始 logit 门限。
- `network.nms_threshold`：OpenCV NMS IoU 门限。
- `network.bbox_expand`：NMS 前包围盒扩张比例。
- `network.max_detections`：NMS 后最多保留候选数量。
- `network.enable_quad_check`：是否检查四边形面积和基本形状。
- `network.min_quad_area_px`：四边形最小面积，单位 `px^2`。
- `network.model_name`：固定 detector 模型名；当前接受 `int8-head-l / int8-grid-l / int16-head-l / int8-head / int8-grid / int16-head`。
- `number_refine.enabled`：是否启用数字 refine。
- `number_refine.detector_min_confidence`：允许进入数字 refine 的 detector 最低置信度。
- `number_refine.classifier_min_confidence`：允许覆盖 detector 编号的分类器最低置信度。
- `number_refine.enforce_type_compatibility`：尺寸类型明确冲突时禁止覆盖编号。

## 预览

`preview.enabled: true` 时启动实时预览。预览只显示本模块当前帧结果叠加，不录像、不写数据文件。

- `preview.output_mode: window` 使用 OpenCV 窗口。
- `preview.output_mode: raw` / `web` / `http` / `bmp` 使用 BMP web 推流。
- `preview.web_bind_address` 默认 `0.0.0.0`。
- `preview.web_port` 默认 `8080`。
- `preview.web_stream_name` 默认 `armor_detector`，直接取流路径为 `/stream/armor_detector`。
- `preview.max_fps` 默认 `30.0`；小于等于 `0` 表示不限频。

## 使用要求

- `Info` 里的图像尺寸、`step`、编码、内参和畸变参数必须与实际相机输出一致。
- 当前主检测模型固定使用 `640x512` 输入；原始图像可以是其他尺寸。
- 原始视频、同步数据和回放包由相机或采集模块保存，不在 `ArmorDetector` 中落盘。
