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
- 可选 `VisionPreview::RuntimeParam`，只用于本模块实时预览

输出:

- `armor_detector/armors_result`: 当前帧装甲板检测结果
- `armor_detector/armors_frame`: 检测结果和当前同步帧引用
- `armor_detector/metrics`: 每帧检测数量、过滤数量、耗时等指标

模型文件:

- `model/armor_keypoint_640x512_bgr.xml`
- `model/armor_keypoint_640x512_bgr.bin`
- `model/armor_keypoint_640x512_bgr.onnx`

OpenVINO 默认使用 `.xml/.bin` 文件；需要接入其他推理后端时使用 `.onnx` 文件。

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
- `referee_auto_detect_color`: 开启后订阅裁判系统摘要包，按本机 `robot_id`
  动态切换敌方颜色；未收到有效 ID 时仍使用 `detect_color`
- `referee_domain`: 裁判系统摘要包所在 topic domain，默认 `host`
- `referee_topic`: 裁判系统摘要包 topic 名，默认 `robot_game_ref`
- `network.score_threshold`: 网络候选置信度门限
- `network.min_confidence`: 最终结果置信度门限
- `network.enable_quad_check`: 是否启用四边形面积检查
- `network.min_quad_area_px`: 四边形最小面积，单位为像素平方
- `network.openvino_device`: `AUTO_DETECT`、`CPU`、`GPU`、`NPU`、`AUTO:*` 或 `MULTI:*`
- `network.openvino_performance_mode`: `LATENCY`、`THROUGHPUT` 或 `CUMULATIVE_THROUGHPUT`
- `depth_correction.enabled`: PnP 深度修正总开关，默认关闭；关闭时发布原始 PnP 位姿
- `depth_correction.camera_normalized_features`: 是否使用相机内参归一化像素特征，默认开启；开启后四边形高度使用 `h/fy`，宽度使用 `w/fx`，重投影误差使用 `err/((fx+fy)/2)`，中心使用 `(x-cx)/fx` 和 `(y-cy)/fy`
- `depth_correction.coeffs`: 线性深度修正系数，顺序为常数项、`pose_z`、四边形高度特征、四边形宽度特征、PnP 重投影误差特征、四边形中心 x 特征、四边形中心 y 特征
- `depth_correction.max_abs_correction_m`: 单次 z 修正绝对值上限，单位 m
- `depth_correction.min_quad_height_px`: 四边形平均高度低于该值时跳过修正
- `depth_correction.height_fusion_enabled`: 是否启用物理高度一致性融合，默认关闭
- `depth_correction.height_fusion_weight`: 向 `armor_height_m * fy / quad_height_px` 靠拢的比例，建议范围 `0~1`
- `depth_correction.height_fusion_max_abs_m`: 高度融合单次 z 修正上限，单位 m
- `depth_correction.armor_height_m`: 装甲板灯条有效高度，默认 `0.056m`
- `preview.enabled`: detector 实时预览总开关；`false` 时不启动预览线程
- `preview.preview_window_name`: OpenCV 窗口名
- `preview.preview_scale`: 显示缩放比例，只影响窗口画面
- `preview.preview_wait_key_ms`: OpenCV 窗口事件轮询时间，单位 ms
- `preview.queue_capacity`: 预览队列长度，超过上限时丢弃旧帧
- `preview.output_mode`: 预览输出模式；`window` 使用 OpenCV 窗口，`mjpeg` 使用 HTTP MJPEG 推流
- `preview.web_bind_address`: MJPEG 监听地址，默认 `0.0.0.0`
- `preview.web_port`: MJPEG 监听端口，默认 `8080`
- `preview.web_jpeg_quality`: MJPEG JPEG 编码质量，默认 `80`
- `preview.web_stream_name`: MJPEG stream 名，默认 `armor_detector`

默认设备策略为 `AUTO_DETECT + LATENCY`，按 `NPU -> GPU -> CPU` 顺序选择可用设备。
CI 使用 `CPU + LATENCY`，保证没有 GPU/NPU 的环境也能构建。

## 边界

- 模块只在 `preview.enabled: true` 时启动实时预览。
- 预览只显示当前 detector overlay，不订阅 topic、不录像、不写 TSV。
- 深度修正只改相机坐标系下的 z，不改角点、旋转或 x/y。
- 线性深度修正系数需要用对应相机、模型、场景数据重新标定；不要把裸像素系数跨相机复用。
- 物理高度融合复用同一组检测角点，只能作为弱一致性约束，不能替代新的独立深度观测。
- 原始视频、同步数据和回放包落盘由相机/同步模块负责，不放在 detector 里。
- 相机参数来自模板参数 `Info`，必须与实际图像尺寸、编码、内参和畸变参数一致。
- 当前模型输入尺寸固定为 `640x512`。
