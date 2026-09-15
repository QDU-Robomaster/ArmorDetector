# ArmorDetector

## Static assembly source line

This source line uses explicit C++ constructor dependencies and ordered instance
arguments. Inspect the current primary header with `xrobot_mod_parser --path .`;
its declarations, not old manifest/config examples, define the interface.
Historical HardwareContainer/ApplicationManager examples below apply only to the
older dynamic source tags. Device/protocol descriptions remain relevant.
See the XRobot [migration guide](https://github.com/xrobot-org/XRobot/blob/dev/MIGRATION.md).
Compilation is not hardware validation; retain version-specific board evidence.


`ArmorDetector` 从 `CameraFrameSync` 读取同步后的图像和 IMU，使用与所选模型绑定的 HailoRT 或 OpenVINO 后端检测装甲板四角点，再根据原生传感器标定求出装甲板在相机坐标系下的位姿。模块输出按值携带共享图像句柄、IMU 和检测结果；逐帧几何始终从共享图像读取。

## 数据流

1. 普通 Topic 回调借用 `const CameraFrameSync<Layout>::SyncedFrame*`；有空闲 `InferSlot` 时复制共享所有权，并直接预处理到该槽的输入缓冲区。三个固定 `InferSlot` 由两个设备内请求位和一个已预处理候补位组成；三者都忙时立即丢弃最新输入，不等待。
2. inference dispatcher 最多提交两个 Hailo 异步请求，第三个 `InferSlot` 只保留下一张已预处理输入，避免设备完成后等待下一次图像回调；完成结果严格按提交顺序交给 output fusion。OpenVINO 使用同一流水线的同步推理分支，每个槽独立持有请求和输出 tensor，不把旧版本的单请求输出视图跨帧复用。
3. 单个 output worker 等待一个空闲 `PostSlot`，把后端输出整理为在该槽处理期间有效的 `CV_32F` 矩阵，再把同步帧和逐帧上下文移动到 `PostSlot`。移动完成后立即释放原 `InferSlot`，随后在同一线程执行 `DecodeOutput`、NMS、语义过滤、PnP 和发布；没有第三个 postprocess 线程或队列。
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

`network.model` 固定绑定模型工件、输入输出语义和所需后端，不提供任意组合的独立 backend 开关。当前包含 8 个 Hailo 模型和 1 个 OpenVINO 模型。原有枚举值 `0..7` 与默认值 `INT16_HEAD_L` 不变；缺少所选模型的后端时明确报错，不切换模型。

Hailo 路线的 8 个枚举值是：

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

- `infer/ArmorDetectorModelRegistry.hpp`：模型枚举到所需后端和稳定工件的映射
- `infer/ArmorDetectorInt8Model.hpp`：`int8` 线输出语义适配
- `infer/ArmorDetectorInt16Model.hpp`：`int16` 线输出语义适配
- `infer/ArmorDetectorModelAdapter.hpp`：统一的模型适配入口

其中：

- `INT8_HEAD*`：`int8` 六输出 host-tail 语义
- `INT8_GRID*`：`int8` 单输出 `21x6720` 语义
- `INT16_HEAD*`：`int16` 三头 `conv47/54/60` 语义，当前对外 canonical 名为 `int16-quality*`
- `INT16_FAST*`：`int16` 三头 `conv47/54/60` 语义，same-HAR fast 路线

### OpenVINO：NUC / x86 Linux / Webots

新增 `ArmorDetectorModel::OPENVINO_640X512`，绑定 OpenVINO 和 `model/armor_detector_640x512.onnx`。工件原样恢复自 2026-05-24 的 `6ee074b60cf42d93b40baf70e4e4e642a68ded0e`，Git blob 为 `08830a05f40f4607d9f9b23b955df50eb3544730`；没有恢复旧 number-refiner 或已退役的 ORT/CUDA 实验分支。

在现有实例配置的 `network` 段中选择：

```yaml
network:
  model: {expr: ArmorDetectorModel::OPENVINO_640X512}
```

模型直接接受 RGB `uint8 [512,640,3]`，预处理仍是拉伸到 `640x512` 后 BGR→RGB；不要额外做 NCHW/归一化。输出为 `float32 [1,20160,22]`，沿用旧版颜色、编号和 `[0,3,2,1]` 角点顺序，并对原始 objectness logit 使用 `network.logit_threshold` 后再输出 sigmoid 置信度。共用 22 字段解码适配器不代表把此 ONNX 重新量化为 INT16；既有 Hailo 模型的门限行为保持不变。

CMake 分别通过 `find_package(HailoRT QUIET CONFIG)` 和 `find_package(OpenVINO QUIET COMPONENTS Runtime)` 自动发现 SDK。可以只有其中一个、两者都有或均无；均无时仍可构建，但任何模型选择都会在初始化时报错且不启动 detector 流水线。安装 OpenVINO 不会自动替换默认 Hailo 模型，无 Hailo 开发机应显式选择上面的新枚举。OpenVINO 的 Runtime CMake 目录可通过 `OpenVINO_DIR` 或 `CMAKE_PREFIX_PATH` 提供。

OpenVINO 默认在其实际枚举出的设备中按 NPU、GPU、CPU 选择；`XR_ARMOR_OPENVINO_DEVICE=CPU` 可显式固定 CPU。指定设备或模型初始化失败会报错，不改用其他设备或后端。`XR_ARMOR_HEF_PATH` 仅影响 Hailo 模型，不影响此 ONNX。

OpenVINO 默认使用现有同步 worker 分支，保留输入槽、输出槽、共享帧所有权、PnP 和发布协议。`ARMOR_DETECTOR_INFERENCE_MODE=sync` 可显式指定；对此后端请求 `async` 会明确报错。Hailo 的默认异步模式及可选同步模式不变。每个 OpenVINO 槽持有独立请求和输出，下一帧推理不会覆盖仍在后处理中的上一帧结果；模型重配置后旧槽不能再提交。

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
- `network.model`：固定 detector 模型枚举；当前接受 `ArmorDetectorModel::INT8_HEAD_L / INT8_GRID_L / INT16_HEAD_L / INT16_FAST_L / INT8_HEAD / INT8_GRID / INT16_HEAD / INT16_FAST / OPENVINO_640X512`。

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

## 测试

宿主 XRobot 工程启用 `BUILD_TESTING=ON` 后，模块注册 `armor_detector_publish_geometry_test`、`armor_detector_pipeline_test`、`armor_detector_input_view_test` 和 `armor_detector_backend_test`。后端测试覆盖模型绑定、缺失 SDK、枚举稳定性；有 OpenVINO 时还会用所附 ONNX 在 CPU 上做真实推理、独立输出对照、跨槽并发读取、输入校验与重配置失效检查，并追加 `armor_detector_unavailable_device_test` 验证显式错误设备不会回退。新增测试的检查在 Release 构建中同样执行。

无 SDK 构建可用 `-DCMAKE_DISABLE_FIND_PACKAGE_OpenVINO=ON -DCMAKE_DISABLE_FIND_PACKAGE_HailoRT=ON` 验证。CI 保留完整模板实例化及本模块拥有的测试，并增加无 SDK 配置；相机回放、真实模块图和 Webots 运行验收属于宿主集成测试，不向模块 CI 注入模拟相机或执行器。
