# ArmorDetector

`ArmorDetector` 当前保持原结果话题不变，同时额外输出 tracker 专用的帧包接口。内部链路已经收敛成四阶段：

1. `CameraFrameSync` 提供同步后的图像帧
2. `OpenVINO` 模型直接输出颜色、编号和四角点
3. 可选传统灯条细化只负责角点修正
4. `PnP` 负责位姿估计并填充检测结果

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
- dense-grid 路径默认信任模型声明角点顺序，关闭传统细化，并使用 IPPE-only PnP；传统细化和更重的 PnP 仍可显式打开做诊断
- 候选抑制复现 dense-grid 源语义：按 confidence 排序取前 128 个候选，丢弃与已保留候选有任意 bbox 交叠的框
- detector 发布 `armors_frame` 时不再额外复制一份 imu/pose 缓冲
- `ArmorDetectorResult` 保留诊断字段：
  - `raw_points / refined` 用于区分网络原始角点和传统细化后的角点
  - `pnp_valid / pnp_reprojection_error_px` 用于判断当前 PnP 解本身是否可信

## 调试与预览

- Detector 不再直接创建窗口或绘制预览。
- 实时预览、原始视频和数据落盘由独立 `VisionPreview` 模块订阅 topic 后完成。
- 可选算法项:
  - `armor_detector.network.openvino_device`: `AUTO_DETECT`、`CPU`、`GPU`、`NPU`、`AUTO:*` 或 `MULTI:*`
  - `armor_detector.network.openvino_performance_mode`: `LATENCY`、`THROUGHPUT` 或 `CUMULATIVE_THROUGHPUT`
  - `armor_detector.network.input_scale`: 默认 `255.0`，用于 OpenVINO IR 的 `/255` 输入归一化
  - `armor_detector.network.direct_point_order`: `DECLARED_ORDER` 或 `CANONICAL_SORT`
  - `armor_detector.network.pnp_strategy`: `IPPE_ONLY` 或 `ROBUST`
  - `armor_detector.network.use_roi`
  - `armor_detector.network.use_traditional_refine`
  - `armor_detector.traditional.refine_mode`: `PAIR_ROI` 或 `SPLIT_ROI_WEIGHTED`
- 部署约束:
  - 默认值使用 `AUTO_DETECT + LATENCY`，按 `NPU -> GPU -> CPU` 顺序选择设备。
  - CI 固定使用 `CPU + LATENCY`，保证没有 GPU/NPU 的 runner 也能编译。
  - 11/12 代 NUC 实机会自动落到 `GPU + LATENCY`；也可以显式配置 `GPU`。
  - 初始化会打印 OpenVINO 可见设备、请求设备、实际设备和性能模式；只有 CPU 可见时打印 warning。
  - `AUTO`/`MULTI` 可用于实验，但当前 detector 是同步单请求链路；真正利用 CPU/GPU/NPU 并行需要后续 async 多请求流水线。
- 环境变量只用于运行期诊断，不作为模块主配置接口：
  - `ARMOR_DETECTOR_AUDIT_EVERY_FRAME`
  - `ARMOR_DETECTOR_AUDIT_ZERO_FRAMES`
  - `XR_ARMOR_DETECTOR_DISABLE_TRADITIONAL_REFINE`
  - `XR_ARMOR_DETECTOR_INPUT_SCALE`
  - `XR_ARMOR_DETECTOR_DIRECT_POINT_ORDER`
  - `XR_ARMOR_DETECTOR_CORNER_REFINE_MODE`
  - `XR_ARMOR_DETECTOR_PNP_STRATEGY`
  - `XR_ARMOR_DETECTOR_DUMP_REFINE_FAILS`
