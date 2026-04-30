# ArmorDetector

`ArmorDetector` 当前保持 legacy 结果话题不变，同时额外输出 tracker 专用的帧包接口。内部链路已经收敛成四阶段：

1. `CameraFrameSync` 提供同步后的图像帧
2. `OpenVINO` 模型直接输出颜色、编号和四角点
3. 可选传统灯条细化只负责角点修正
4. `PnP` 负责位姿估计并填充检测结果

## 运行角色

- 输入: `CameraFrameSync<Info>::SyncedFrame`
- 输出: `armor_detector/armors_result`、`armor_detector/armors_frame`、`armor_detector/metrics`
- 模型:
  - `model/yolov5.xml` + `model/yolov5.bin`

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
- detector 发布 `armors_frame` 时不再额外复制一份 imu/pose 缓冲
- `ArmorDetectorResult` 保留诊断字段：
  - `raw_points / refined` 用于区分网络原始角点和传统细化后的角点
  - `pnp_valid / pnp_reprojection_error_px` 用于判断当前 PnP 解本身是否可信

## 调试与预览

- Detector 不再直接创建窗口或绘制预览。
- 实时预览、原始视频和数据落盘由独立 `VisionPreview` 模块订阅 topic 后完成。
- 可选算法项:
  - `armor_detector.yolo.use_roi`
  - `armor_detector.yolo.use_traditional_refine`
- 环境变量只用于运行期诊断，不作为模块主配置接口：
  - `ARMOR_DETECTOR_AUDIT_EVERY_FRAME`
  - `ARMOR_DETECTOR_AUDIT_ZERO_FRAMES`
  - `XR_ARMOR_DETECTOR_DISABLE_TRADITIONAL_REFINE`
  - `XR_ARMOR_DETECTOR_CENTER_LETTERBOX`
  - `XR_ARMOR_DETECTOR_YOLO_LETTERBOX`
  - `XR_ARMOR_DETECTOR_DUMP_REFINE_FAILS`
