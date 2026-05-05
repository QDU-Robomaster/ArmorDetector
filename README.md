# ArmorDetector

`ArmorDetector` 从 `CameraFrameSync` 获取同步后的图像帧，调用 OpenVINO
检测装甲板，并使用相机内参计算每个装甲板在相机坐标系下的位姿。

## 数据流

1. 读取 `CameraFrameSync<Info>::SyncedFrame`
2. 将图像缩放到模型输入尺寸并执行 OpenVINO 推理
3. 解码候选装甲板，过滤低置信度或几何异常的结果
4. 对有效结果执行 PnP
5. 发布检测结果、原始帧引用和运行指标

## 输入输出

输入:

- `CameraFrameSync<Info>::SyncedFrame`

输出:

- `armor_detector/armors_result`: 当前帧装甲板检测结果
- `armor_detector/armors_frame`: 检测结果和当前同步帧引用
- `armor_detector/metrics`: 每帧检测数量、过滤数量、耗时等指标

模型文件:

- `model/armor_keypoint_640x512_bgr.xml`
- `model/armor_keypoint_640x512_bgr.bin`

## 结果内容

每个装甲板结果包含:

- 颜色、编号、大小类型和置信度
- 图像中的包围框、中心点和四个角点
- PnP 是否成功
- PnP 平均重投影误差
- 相机坐标系下的位姿

`armors_frame` 中的原始图像和 IMU 指针只在同进程回调期间有效，下游模块应在回调内同步消费。

## 配置

- `detect_color`: `0` 只保留红色，`1` 只保留蓝色，其他值不过滤颜色
- `network.score_threshold`: 网络候选置信度门限
- `network.min_confidence`: 最终结果置信度门限
- `network.enable_quad_check`: 是否启用四边形面积检查
- `network.min_quad_area_px`: 四边形最小面积，单位为像素平方
- `network.openvino_device`: `AUTO_DETECT`、`CPU`、`GPU`、`NPU`、`AUTO:*` 或 `MULTI:*`
- `network.openvino_performance_mode`: `LATENCY`、`THROUGHPUT` 或 `CUMULATIVE_THROUGHPUT`

默认设备策略为 `AUTO_DETECT + LATENCY`，按 `NPU -> GPU -> CPU` 顺序选择可用设备。
CI 使用 `CPU + LATENCY`，保证没有 GPU/NPU 的环境也能构建。

## 边界

- 模块不创建窗口，也不负责绘制预览。
- 原始视频、可视化和落盘应由独立模块订阅 topic 后处理。
- 相机参数来自模板参数 `Info`，必须与实际图像尺寸、编码、内参和畸变参数一致。
- 当前模型输入尺寸固定为 `640x512`。
