# ArmorDetector

`ArmorDetector` 当前保持原结果话题不变，同时额外输出 tracker 专用的帧包接口。内部链路已经收敛成三阶段：

1. `CameraFrameSync` 提供同步后的图像帧
2. `OpenVINO` 模型直接输出颜色、编号和四角点
3. `PnP` 负责位姿估计并填充检测结果

## 运行角色

- 输入: `CameraFrameSync<Info>::SyncedFrame`
- 输出: `armor_detector/armors_result`、`armor_detector/armors_frame`、`armor_detector/metrics`
- 模型:
  - `model/armor_keypoint_640x512_quantized.xml` + `model/armor_keypoint_640x512_quantized.bin`

## 对外接口

- `armors_result` 仍然发布 `ArmorDetectionsMessage`
- `armors_frame` 发布 `ArmorDetectionsFrameMessage<Info>`
  - `source_frame.image_frame`: 原始共享图像槽位指针
  - `source_frame.imu`: 同步帧里的姿态/运动数据指针
  - `results`: 当前帧识别结果
- `metrics` 仍然发布 `ArmorDetectorMetrics`
- 仍然只消费 `armors_result` 的录像器、truth publisher 不需要跟着这次接口清理一起改

## 内部约束

- 不再需要独立 `NumberClassifier`
- `PnPSolver` 现在直接吃编译期 `CameraInfo`
- detector 解码阶段不再使用并行数组和散落的输出列号
- 当前只保留 640x512 dense-grid keypoint 模型，输入直接拉伸到模型尺寸，输出按 `[6720,21]` 解码
- dense-grid 路径固定按模型声明角点顺序转换为左上、右上、右下、左下，并使用 IPPE PnP
- 候选抑制复现 dense-grid 源语义：按 confidence 排序取前 128 个候选，丢弃与已保留候选有任意 bbox 交叠的框
- detector 发布 `armors_frame` 时不再额外复制一份 imu/pose 缓冲
- `ArmorDetectorResult` 只保留下游需要的语义、2D 几何、PnP 位姿和 `pnp_valid / pnp_reprojection_error_px`

## 配置

- Detector 不再直接创建窗口或绘制预览。
- 实时预览、原始视频和数据落盘由独立 `VisionPreview` 模块订阅 topic 后完成。
- 算法配置只保留生产路径必需项:
  - `detect_color`: `0` 红色、`1` 蓝色、其他值不限制颜色
  - `network.use_roi / roi_x / roi_y / roi_width / roi_height`: 可选 ROI 裁剪
  - `network.score_threshold`: 网络候选置信度门限
  - `network.min_confidence`: 语义过滤后的最终置信度门限
  - `network.enable_quad_check / min_quad_area_px`: 四边形合法性过滤
  - `network.openvino_device`: `AUTO_DETECT`、`CPU`、`GPU`、`NPU`、`AUTO:*` 或 `MULTI:*`
  - `network.openvino_performance_mode`: `LATENCY`、`THROUGHPUT` 或 `CUMULATIVE_THROUGHPUT`
- 默认值使用 `AUTO_DETECT + LATENCY`，按 `NPU -> GPU -> CPU` 顺序选择设备。
- CI 固定使用 `CPU + LATENCY`，保证没有 GPU/NPU 的 runner 也能编译。
- 初始化会打印 OpenVINO 可见设备、请求设备、实际设备和性能模式；只有 CPU 可见时打印 warning。
