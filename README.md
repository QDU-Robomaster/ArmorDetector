# ArmorDetector

`ArmorDetector` 从 `CameraFrameSync` 读取同步后的图像和 IMU，使用 HailoRT 模型检测装甲板四角点，再根据原生传感器标定求出装甲板在相机坐标系下的位姿。模块输出按值携带共享图像句柄、IMU 和检测结果；逐帧几何始终从共享图像读取。

## 数据流

1. 普通 Topic 回调借用 `const CameraFrameSync<Layout>::SyncedFrame*`；有空闲 `InferSlot` 时复制共享所有权，并直接预处理到该槽的 Hailo 输入绑定。三个固定 `InferSlot` 由两个设备内请求位和一个已预处理候补位组成；三者都忙时立即丢弃最新输入，不等待。
2. inference dispatcher 最多提交两个 Hailo 异步请求，第三个 `InferSlot` 只保留下一张已预处理输入，避免设备完成后等待下一次图像回调；完成结果严格按提交顺序交给 output fusion。
3. 单个 output worker 等待一个空闲 `PostSlot`，把 Hailo 输出整理为持久的 `CV_32F` 矩阵，再把同步帧和逐帧上下文移动到 `PostSlot`。移动完成后立即释放原 `InferSlot`，随后在同一线程执行 `DecodeOutput`、NMS、语义过滤、PnP 和发布；没有第三个 postprocess 线程或队列。
4. 发布边界把图像所有权从 `PostSlot` 移进临时 `DetectedFrame`，把四角点、中心和包围盒映射到原生传感器坐标，使用原生 K/D 执行 PnP，并通过普通 Topic 同步发布 `const DetectedFrame*`。已经接纳的帧不会在内部阶段丢弃。

## 输入输出

输入：

- `<camera image topic>_synced`：`const CameraFrameSync<Layout>::SyncedFrame*`
- `host/robot_game_ref`，仅在 `referee_auto_detect_color: true` 时订阅

输出：

- `armor_detector/armors_frame`：`const DetectedFrame<Layout>*`

Topic 中的 `DetectedFrame*` 只在本次同步回调期间有效。异步消费者必须在回调返回前复制 `DetectedFrame`；其中的 `SharedFrame` 会保留 CameraBase 槽位，直到最后一份阶段对象释放。`FrameGeometry` 不重复存放，消费者只读取当前帧 `SharedFrame` 中的不可变参数。

下游权威帧时间是 `SyncedFrame::imu.timestamp_us`，对应 MCU `FRAME_TRIGGER` 的陀螺仪时间。detector benchmark、结果 Topic 时间戳和后续链路都使用该时间；`ImageFrame::timestamp_us` 只作为明确命名的相机设备诊断时间保留。

## 模型

Hailo 路线现在只通过固定 `network.model` 枚举选择。当前支持的 `8` 个枚举值是：

- `ArmorDetectorModel::INT8_HEAD_L`
- `ArmorDetectorModel::INT8_GRID_L`
- `ArmorDetectorModel::INT16_HEAD_L`
- `ArmorDetectorModel::INT16_FAST_L`
- `ArmorDetectorModel::INT8_HEAD`
- `ArmorDetectorModel::INT8_GRID`
- `ArmorDetectorModel::INT16_HEAD`
- `ArmorDetectorModel::INT16_FAST`

这些变体都已经映射到模块目录下的稳定工件文件名：

- `model/skd_int8_head_l.hef`
- `model/skd_int8_grid_l.hef`
- `model/szu_int16_head_l.hef`
- `model/int16_fast_l.hef`
- `model/skd_int8_head.hef`
- `model/skd_int8_grid.hef`
- `model/szu_int16_head.hef`
- `model/int16_fast.hef`

当前 clean-tree 已按最新已验证候选更新过以下稳定工件：

- `model/skd_int8_head.hef`
  - refreshed to the measured-better bare `26T` public `1ctx` candidate
- `model/skd_int8_grid_l.hef`
  - refreshed to `int8_grid_l_force2limits.hef`
- `model/szu_int16_head_l.hef`
  - refreshed to promoted `int16_quality_l_perfmax_autores.hef`
- `model/szu_int16_head.hef`
  - refreshed to measured bare `int16_quality_h8_perfmax_autores.hef`

`infer/` 目录负责不同模型的适配：

- `infer/ArmorDetectorModelRegistry.hpp`：模型枚举到稳定 HEF 的映射
- `infer/ArmorDetectorInt8Model.hpp`：`int8` 线输出语义适配
- `infer/ArmorDetectorInt16Model.hpp`：`int16` 线输出语义适配
- `infer/ArmorDetectorModelAdapter.hpp`：统一的模型适配入口

其中：

- `INT8_HEAD*`：`int8` 六输出 host-tail 语义
- `INT8_GRID*`：`int8` 单输出 `21x6720` 语义
- `INT16_HEAD*`：`int16` 三头 `conv47/54/60` 语义，当前对外 canonical 名为 `int16-quality*`
- `INT16_FAST*`：`int16` 三头 `conv47/54/60` 语义，same-HAR fast 路线

## 结果内容

单个装甲板结果包含：

- 颜色、编号、尺寸类型和置信度
- 图像包围盒、中心点和四个角点
- PnP 是否成功
- PnP 平均重投影误差
- OpenCV 相机坐标系下的装甲板位姿

相机坐标系沿用 OpenCV 约定：`x` 向右，`y` 向下，`z` 向前。这里的位姿只描述装甲板相对相机的位置和朝向，不包含 IMU 姿态融合。

发布的包围盒、中心和四角点使用原生传感器像素坐标；`center_norm`、网络解码、NMS、阈值、结果 TSV 和 detector preview 仍使用当前帧坐标。这保证 resize、wide decimation 或 ROI 不改变检测语义。

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
- `network.model`：固定 detector 模型枚举；当前接受 `ArmorDetectorModel::INT8_HEAD_L / INT8_GRID_L / INT16_HEAD_L / INT16_FAST_L / INT8_HEAD / INT8_GRID / INT16_HEAD / INT16_FAST`。

如果配置层需要写显式表达式，直接用：

```yaml
network:
  model: {expr: ArmorDetectorModel::INT8_GRID_L}
```

## 预览

`preview.enabled: true` 时启动实时预览。预览只显示本模块当前帧结果叠加，不录像、不写数据文件。

- `preview.output_mode: window` 使用 OpenCV 窗口。
- `preview.output_mode: raw` / `web` / `http` / `bmp` 使用 BMP web 推流。
- `preview.web_bind_address` 默认 `0.0.0.0`。
- `preview.web_port` 默认 `8080`。
- `preview.web_stream_name` 默认 `armor_detector`，直接取流路径为 `/stream/armor_detector`。
- `preview.max_fps` 默认 `30.0`；小于等于 `0` 表示不限频。

## 使用要求

- 模板参数只描述图像缓冲区最大宽高、`step` 和固定编码。原生标定由 `CameraFrameSync::Calibration()` 在构造期复制，ROI、下采样和翻转由每帧 `FrameGeometry` 描述。
- 当前主检测模型固定使用 `640x512` 输入；原始图像可以是其他尺寸。
- 原始视频、同步数据和回放包由相机或采集模块保存，不在 `ArmorDetector` 中落盘。
