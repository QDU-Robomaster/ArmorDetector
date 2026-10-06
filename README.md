# ArmorDetector

装甲板检测模块：HailoRT / OpenVINO 模型推理与 PnP 位姿估计 / Armor detection Module with HailoRT or OpenVINO model inference and PnP pose estimation

## 1. 模块作用 / Purpose

构造时，ArmorDetector 加载 `cfg.network.model` 指定的模型，创建推理线程 `armor-infer` 与输出线程 `armor-output`（线程名在 Linux 下设置），并订阅 CameraFrameSync 发布的同步帧。输入图像可以是 BGR8、RGB8、BGRA8、RGBA8 或 MONO8，统一转换为 BGR 后拉伸到 `640x512`，预处理到推理槽的输入缓冲区。模型输出四角点、颜色和编号，经过 NMS、颜色与编号过滤、尺寸类型判定后，使用原生传感器标定做 PnP，得到装甲板在相机坐标系下的位姿，最后发布到 Topic `armors_frame`。

处理流水线由三个 `InferSlot` 和两个 `PostSlot` 组成：

1. 同步帧回调在有空闲 `InferSlot` 时复制共享图像所有权并完成预处理；三个 `InferSlot` 均被占用时丢弃新到的帧。
2. 推理线程提交推理。HailoRT 模型异步推理，最多同时提交两个请求，第三个 `InferSlot` 保存下一帧已预处理的输入，完成结果按提交顺序交给输出线程。OpenVINO 模型同步推理，每个槽持有独立的请求和输出 tensor。
3. 输出线程等待一个空闲的 `PostSlot`，把后端输出整理为 `CV_32F` 矩阵，移交同步帧与逐帧上下文后释放 `InferSlot`，并在同一线程内完成解码、NMS、过滤、PnP 和发布。

`cfg.preview.enabled` 为 `true` 时启动 VisionPreview，在当前帧上叠加检测结果：`output_mode` 为 `window` 时使用 OpenCV 窗口，为 `raw`、`web`、`http` 或 `bmp` 时使用 BMP 网页推流，取流路径为 `/stream/<web_stream_name>`。默认关闭预览。

`OnMonitor()` 输出预处理、推理线程、后处理和结果填充四个阶段的耗时统计（次数、平均值、最小值、最大值，单位 μs）。

Upon construction, ArmorDetector loads the model given by `cfg.network.model`, creates the inference thread `armor-infer` and the output thread `armor-output` (the thread names are set on Linux), and subscribes to the synchronized frames published by CameraFrameSync. Input images in BGR8, RGB8, BGRA8, RGBA8 or MONO8 are converted to BGR, stretched to `640x512` and preprocessed into the input buffer of an inference slot. The model outputs four corner points, color and number; after NMS, color and number filtering and size-type classification, PnP with the native sensor calibration gives the armor pose in the camera frame, which is published on the Topic `armors_frame`.

The processing pipeline consists of three `InferSlot`s and two `PostSlot`s:

1. The synchronized-frame callback copies the shared image ownership and preprocesses the frame when an `InferSlot` is free; a frame arriving while all three `InferSlot`s are occupied is dropped.
2. The inference thread submits the inference. HailoRT models run asynchronously with at most two requests in flight, the third `InferSlot` holds the preprocessed input of the next frame, and completions are handed to the output thread in submission order. OpenVINO models run synchronously, and each slot owns its request and output tensor.
3. The output thread waits for a free `PostSlot`, arranges the backend output into a `CV_32F` matrix, moves the synchronized frame and the per-frame context over, releases the `InferSlot`, and then decodes, runs NMS, filters, solves PnP and publishes on the same thread.

When `cfg.preview.enabled` is `true`, VisionPreview starts and overlays the detection results on the current frame: `output_mode` `window` uses an OpenCV window, and `raw`, `web`, `http` or `bmp` use a BMP web stream at `/stream/<web_stream_name>`. Preview is off by default.

`OnMonitor()` logs duration statistics (count, average, minimum, maximum, in μs) for the preprocess, inference thread, postprocess and result stages.

## 2. 模型与推理 / Models and Inference

`network.model` 绑定模型文件、输出语义和所需后端。提供 8 个 HailoRT 模型和 1 个 OpenVINO 模型，默认 `INT16_HEAD_L`。所选模型需要的后端未构建时，初始化记录错误日志并结束，流水线保持未启动。

| 枚举 `ArmorDetectorModel::` | 值 | 后端 | 文件 |
| --- | --- | --- | --- |
| `INT8_HEAD_L` | 0 | HailoRT | `model/skd_int8_head_l.hef` |
| `INT8_GRID_L` | 1 | HailoRT | `model/skd_int8_grid_l.hef` |
| `INT16_HEAD_L` | 2 | HailoRT | `model/szu_int16_head_l.hef` |
| `INT8_HEAD` | 3 | HailoRT | `model/skd_int8_head.hef` |
| `INT8_GRID` | 4 | HailoRT | `model/skd_int8_grid.hef` |
| `INT16_HEAD` | 5 | HailoRT | `model/szu_int16_head.hef` |
| `INT16_FAST_L` | 6 | HailoRT | `model/int16_fast_l.hef` |
| `INT16_FAST` | 7 | HailoRT | `model/int16_fast.hef` |
| `OPENVINO_640X512` | 8 | OpenVINO | `model/armor_detector_640x512.onnx` |

模型输出语义由 `infer/` 目录下的适配器描述：

- `INT8_HEAD*`：`int8` 六输出 host-tail 语义。
- `INT8_GRID*`：`int8` 单输出 `21x6720` 语义。
- `INT16_HEAD*`：`int16` 三头 `conv47/54/60` 语义（日志中的模型名为 `int16-quality*`）。
- `INT16_FAST*`：`int16` 三头 `conv47/54/60` 语义，fast 版本。

环境变量 `XR_ARMOR_HEF_PATH` 指定 HailoRT 模型的 HEF 路径，覆盖上表中的默认文件。

`OPENVINO_640X512` 使用 `model/armor_detector_640x512.onnx`，模型直接接受 RGB `uint8 [512,640,3]` 输入，模块把图像拉伸到 `640x512` 并由 BGR 转为 RGB 后送入。输出为 `float32 [1,20160,22]`，颜色、编号和 `[0,3,2,1]` 角点顺序与 `INT16_*` 模型使用同一解码适配器，置信度为 sigmoid 后的值。`network.logit_threshold` 与原始 objectness logit 比较。

OpenVINO 在枚举到的设备中按 NPU、GPU、CPU 的顺序选择；环境变量 `XR_ARMOR_OPENVINO_DEVICE`（例如 `CPU`）指定设备。指定的设备或模型初始化失败时，记录错误日志并结束初始化。

HailoRT 默认异步推理，OpenVINO 使用同步推理。环境变量 `ARMOR_DETECTOR_INFERENCE_MODE` 取 `sync` 时 HailoRT 也同步推理；取 `async` 仅适用于 HailoRT，取 `sync` 与 `async` 以外的值或对 OpenVINO 取 `async` 时，记录错误日志，流水线保持未启动。环境变量 `ARMOR_DETECTOR_INFERENCE_CPU=<n>` 把推理线程绑定到 CPU `n`（Linux）。

调试用环境变量：`ARMOR_DETECTOR_DUMP_TSV` 指定输出文件，导出最终结果 TSV；`ARMOR_DETECTOR_DUMP_OUTPUT_F32` 与 `ARMOR_DETECTOR_DUMP_OUTPUT_FRAME_INDEX` 导出指定帧的网络输出矩阵。

`network.model` selects the model file, the output semantics and the required backend. There are 8 HailoRT models and 1 OpenVINO model, with `INT16_HEAD_L` as the default. When the backend required by the selected model is not built, initialization logs an error and ends, and the pipeline stays stopped.

| Enum `ArmorDetectorModel::` | Value | Backend | File |
| --- | --- | --- | --- |
| `INT8_HEAD_L` | 0 | HailoRT | `model/skd_int8_head_l.hef` |
| `INT8_GRID_L` | 1 | HailoRT | `model/skd_int8_grid_l.hef` |
| `INT16_HEAD_L` | 2 | HailoRT | `model/szu_int16_head_l.hef` |
| `INT8_HEAD` | 3 | HailoRT | `model/skd_int8_head.hef` |
| `INT8_GRID` | 4 | HailoRT | `model/skd_int8_grid.hef` |
| `INT16_HEAD` | 5 | HailoRT | `model/szu_int16_head.hef` |
| `INT16_FAST_L` | 6 | HailoRT | `model/int16_fast_l.hef` |
| `INT16_FAST` | 7 | HailoRT | `model/int16_fast.hef` |
| `OPENVINO_640X512` | 8 | OpenVINO | `model/armor_detector_640x512.onnx` |

The output semantics of the models are described by the adapters in the `infer/` directory:

- `INT8_HEAD*`: `int8` six-output host-tail semantics.
- `INT8_GRID*`: `int8` single-output `21x6720` semantics.
- `INT16_HEAD*`: `int16` three-head `conv47/54/60` semantics (the model name in the logs is `int16-quality*`).
- `INT16_FAST*`: `int16` three-head `conv47/54/60` semantics, fast version.

The environment variable `XR_ARMOR_HEF_PATH` sets the HEF path of the HailoRT models and overrides the default files in the table.

`OPENVINO_640X512` uses `model/armor_detector_640x512.onnx`. The model takes RGB `uint8 [512,640,3]` directly; the Module stretches the image to `640x512` and converts BGR to RGB before feeding it. The output is `float32 [1,20160,22]`; the color, number and `[0,3,2,1]` corner order go through the same decoding adapter as the `INT16_*` models, and the confidence is the value after the sigmoid. `network.logit_threshold` is compared with the raw objectness logit.

OpenVINO selects among the enumerated devices in the order NPU, GPU, CPU; the environment variable `XR_ARMOR_OPENVINO_DEVICE` (for example `CPU`) sets the device. When the given device or the model fails to initialize, an error is logged and initialization ends.

HailoRT runs asynchronously by default and OpenVINO runs synchronously. The environment variable `ARMOR_DETECTOR_INFERENCE_MODE` set to `sync` also makes HailoRT synchronous; `async` applies to HailoRT only, and any value other than `sync` and `async`, or `async` with OpenVINO, logs an error and leaves the pipeline stopped. The environment variable `ARMOR_DETECTOR_INFERENCE_CPU=<n>` binds the inference thread to CPU `n` (Linux).

Debug environment variables: `ARMOR_DETECTOR_DUMP_TSV` gives the output file for the final-result TSV export; `ARMOR_DETECTOR_DUMP_OUTPUT_F32` and `ARMOR_DETECTOR_DUMP_OUTPUT_FRAME_INDEX` export the network output matrix of the given frame.

## 3. 时间戳与坐标 / Timestamp and Coordinates

时间戳：`SyncedFrame::imu.timestamp_us` 是 MCU `FRAME_TRIGGER` 对应的陀螺仪时间，作为检测统计、发布时间戳和后续链路使用的帧时间。相机设备时间 `ImageFrame::timestamp_us` 记录为诊断信息。

坐标：发布的包围盒、中心和四角点使用原生传感器像素坐标，角点顺序为左上、右上、右下、左下。`center_norm`、网络解码、NMS、阈值、结果 TSV 和预览使用当前帧坐标，因此缩放、下采样或 ROI 变化时检测语义保持一致。ROI、下采样和翻转由每帧 `FrameGeometry` 描述。PnP 使用原生角点与原生标定，位姿 `pose` 位于 OpenCV 相机坐标系：`x` 向右，`y` 向下，`z` 向前。

Timestamp: `SyncedFrame::imu.timestamp_us` is the gyroscope time of the MCU `FRAME_TRIGGER` and serves as the frame time for the detection statistics, the publish timestamp and the downstream chain. The camera device time `ImageFrame::timestamp_us` is recorded as diagnostic information.

Coordinates: the published bounding box, center and four corner points use native sensor pixel coordinates, with the corners ordered top-left, top-right, bottom-right, bottom-left. `center_norm`, network decoding, NMS, thresholds, the result TSV and the preview use current-frame coordinates, so the detection semantics stay the same when scaling, decimation or the ROI change. The ROI, decimation and flipping are described by the per-frame `FrameGeometry`. PnP uses the native corners and the native calibration, and the pose `pose` is in the OpenCV camera frame: `x` right, `y` down, `z` forward.

## 4. 构造接口 / Constructor

```cpp
template <CameraTypes::FrameLayout FrameLayoutV>
class ArmorDetector;

ArmorDetector(Sync& sync, Config cfg = DefaultConfig());  // 节选 / excerpt
```

模板参数：

- `FrameLayoutV`：帧布局，与上游 CameraFrameSync 和相机的帧布局相同。原生标定由 `CameraFrameSync::Calibration()` 在构造时复制，原始图像可以是 `640x512` 以外的尺寸。

依赖：

- `sync`：`CameraFrameSync<FrameLayoutV>&`，提供同步帧、IMU 和原生标定。

配置参数（`cfg`，类型 `Config`；`DefaultConfig()` 给出以下全部默认值）：

- `detect_color`：保留的目标颜色，`0` 为红色，`1` 为蓝色，其他值关闭颜色过滤，默认 `1`。
- `network.model`：模型枚举，默认 `ArmorDetectorModel::INT16_HEAD_L`。
- `network.min_confidence`：最终置信度门限，默认 `0.1`。
- `network.enable_quad_check`：是否检查四边形凸性和面积，默认 `true`。
- `network.min_quad_area_px`：四边形最小面积，单位 px²，默认 `16.0`。
- `network.logit_threshold`：objectness 预过滤门限，默认 `0.619`；`OPENVINO_640X512` 与原始 logit 比较，`INT16_*` 与 sigmoid 后的置信度比较，`INT8_*` 使用 `min_confidence` 预过滤。
- `network.nms_threshold`：NMS 的 IoU 门限，默认 `0.45`。
- `network.bbox_expand`：NMS 前包围盒的扩张比例，默认 `0.1`。
- `network.max_detections`：NMS 后保留的候选数量上限，默认 `128`。
- `referee_auto_detect_color`：按裁判系统的 robot_id 切换敌方颜色，默认 `false`。robot_id 为 1 至 99（红方）时保留蓝色，为 101 至 199（蓝方）时保留红色；收到有效 robot_id 之前使用 `detect_color`。启用时，构造函数等待 Topic `referee_topic` 出现。
- `referee_domain`：裁判系统摘要包所在的 Topic 域，默认 `"host"`。
- `referee_topic`：裁判系统摘要包的 Topic 名，默认 `"robot_game_ref"`。
- `preview`：`VisionPreview::RuntimeParam`，默认关闭，`preview_window_name` 为 `"armor_detector_preview"`，`preview_scale` 为 `0.5`，`web_stream_name` 为 `"armor_detector"`，其余字段取 VisionPreview 的默认值，字段见 VisionPreview。
- `lightbar_keypoints.length_mm` / `lightbar_keypoints.end_offset_px`：灯条端点关键点标定，默认 `56.0` / `0.0`，即不改变角点。网络给出的两个端点关键点按"灯条上相距 `length_mm` 的两点、每端再向外偏 `end_offset_px` 像素"建模；发布前每根灯条绕中点缩放到 PnP 模型的 56 mm 灯条长度，灯条中点、方位和两灯条间距不变。标定方法：在比赛曝光下把装甲板放在至少 3 个经激光测距的距离（2–8 m），拟合检测到的灯条像素长度 = a·f·L/Z + b（L = 56 mm，f 为焦距，Z 为距离），取 `length_mm` = a·L、`end_offset_px` = b/2；更换曝光或镜头后重新标定。

流水线启动后，`SetConfig()` 记录错误日志并返回。

Template parameter:

- `FrameLayoutV`: the frame layout, identical to that of the upstream CameraFrameSync and camera. The native calibration is copied from `CameraFrameSync::Calibration()` upon construction, and the raw image may have a size other than `640x512`.

Dependency:

- `sync`: `CameraFrameSync<FrameLayoutV>&`, providing the synchronized frames, the IMU data and the native calibration.

Configuration parameters (`cfg`, of type `Config`; `DefaultConfig()` gives all defaults below):

- `detect_color`: the target color to keep, `0` for red, `1` for blue, any other value disables color filtering, default `1`.
- `network.model`: the model enum, default `ArmorDetectorModel::INT16_HEAD_L`.
- `network.min_confidence`: the final confidence threshold, default `0.1`.
- `network.enable_quad_check`: whether to check quadrilateral convexity and area, default `true`.
- `network.min_quad_area_px`: the minimum quadrilateral area in px², default `16.0`.
- `network.logit_threshold`: the objectness pre-filter threshold, default `0.619`; `OPENVINO_640X512` compares it with the raw logit, `INT16_*` with the confidence after the sigmoid, and `INT8_*` pre-filter with `min_confidence`.
- `network.nms_threshold`: the NMS IoU threshold, default `0.45`.
- `network.bbox_expand`: the bounding-box expansion ratio before NMS, default `0.1`.
- `network.max_detections`: the maximum number of candidates kept after NMS, default `128`.
- `referee_auto_detect_color`: switches the enemy color by the referee robot_id, default `false`. A robot_id from 1 to 99 (red side) keeps blue, and from 101 to 199 (blue side) keeps red; `detect_color` applies until a valid robot_id arrives. When enabled, the constructor waits for the Topic `referee_topic` to appear.
- `referee_domain`: the Topic domain of the referee summary packet, default `"host"`.
- `referee_topic`: the Topic name of the referee summary packet, default `"robot_game_ref"`.
- `preview`: `VisionPreview::RuntimeParam`, disabled by default, `preview_window_name` is `"armor_detector_preview"`, `preview_scale` is `0.5` and `web_stream_name` is `"armor_detector"`, the other fields take the defaults of VisionPreview; see VisionPreview for the fields.
- `lightbar_keypoints.length_mm` / `lightbar_keypoints.end_offset_px`: light-bar end keypoint calibration, default `56.0` / `0.0`, which leaves the corners unchanged. The two end keypoints of a bar are modeled as two points `length_mm` apart on the bar, each shifted outward by `end_offset_px` pixels; before publishing, each bar is scaled about its midpoint to the 56 mm light-bar length of the PnP model, so the bar midpoints, the bearing and the bar separation stay unchanged. Calibration: under match exposure, place an armor at three or more laser-measured distances (2–8 m) and fit detected bar length in px = a·f·L/Z + b (L = 56 mm, f the focal length, Z the distance), then set `length_mm` = a·L and `end_offset_px` = b/2; recalibrate after changing the exposure or the lens.

After the pipeline has started, `SetConfig()` logs an error and returns.

## 5. Topic

| Topic | 方向 | 类型 | 说明 |
| --- | --- | --- | --- |
| `sync.SyncedFrameTopicName()`（默认 `<相机图像 Topic>_synced`） | 订阅 | `const CameraFrameSync<FrameLayoutV>::SyncedFrame*` | 同步后的图像与 IMU |
| `referee_topic`（域 `referee_domain`，默认 `host` / `robot_game_ref`） | 订阅 | 裁判系统摘要包 | 首字节为 robot_id；仅 `referee_auto_detect_color` 为 `true` 时订阅 |
| `armors_frame`（域 `armor_detector`） | 发布 | `const DetectedFrame<FrameLayoutV>*` | 一帧的检测结果 |

`DetectedFrame` 包含 `sequence`（CameraFrameSync 的帧序号）、`image`（共享图像所有权句柄）、`imu`（与图像对齐的 IMU 样本）和 `detections`（`std::vector<ArmorDetectorResult>`）。Topic 中的指针只在同步回调期间有效；异步订阅者在回调返回前复制 `DetectedFrame`，其中的 `SharedFrame` 持有 CameraBase 的图像槽位，直到最后一份副本释放。帧几何 `FrameGeometry` 从 `image` 指向的帧读取。ArmorTracker 订阅 `armor_detector` 域的 `armors_frame`。

`ArmorDetectorResult` 的字段：

- `color`、`number`、`type`、`priority`、`confidence`：颜色、编号、尺寸类型（`SMALL` / `LARGE`）、按编号给出的默认优先级和置信度。
- `box`、`center`、`points`：包围盒、中心和四角点，原生传感器像素坐标。
- `center_norm`、`distance_to_image_center`：按当前帧宽高归一化的中心，以及中心到相机主点的像素距离。
- `pnp_valid`、`pnp_reprojection_error_px`、`pose`：PnP 是否成功、平均重投影误差（px）和装甲板在 OpenCV 相机坐标系下的位姿。

| Topic | Direction | Type | Meaning |
| --- | --- | --- | --- |
| `sync.SyncedFrameTopicName()` (default `<camera image Topic>_synced`) | Subscribe | `const CameraFrameSync<FrameLayoutV>::SyncedFrame*` | Synchronized image and IMU |
| `referee_topic` (domain `referee_domain`, default `host` / `robot_game_ref`) | Subscribe | Referee summary packet | The first byte is the robot_id; subscribed only when `referee_auto_detect_color` is `true` |
| `armors_frame` (domain `armor_detector`) | Publish | `const DetectedFrame<FrameLayoutV>*` | Detection results of one frame |

`DetectedFrame` contains `sequence` (the CameraFrameSync frame sequence number), `image` (the shared image ownership handle), `imu` (the IMU sample aligned with the image) and `detections` (`std::vector<ArmorDetectorResult>`). The pointer in the Topic is valid during the synchronous callback only; an asynchronous subscriber copies the `DetectedFrame` before the callback returns, and the `SharedFrame` inside holds the CameraBase image slot until the last copy is released. The frame geometry `FrameGeometry` is read from the frame that `image` points to. ArmorTracker subscribes to `armors_frame` in the `armor_detector` domain.

Fields of `ArmorDetectorResult`:

- `color`, `number`, `type`, `priority`, `confidence`: color, number, size type (`SMALL` / `LARGE`), the default priority derived from the number, and the confidence.
- `box`, `center`, `points`: bounding box, center and four corner points in native sensor pixel coordinates.
- `center_norm`, `distance_to_image_center`: the center normalized by the current-frame width and height, and the pixel distance from the center to the camera principal point.
- `pnp_valid`, `pnp_reprojection_error_px`, `pose`: whether PnP succeeded, the mean reprojection error in px, and the armor pose in the OpenCV camera frame.

## 6. 配置示例 / Configuration Example

`xrobot instance add QDU-Robomaster/ArmorDetector --template-arg <FrameLayout>` 写入的实例，`sync` 填写为 CameraFrameSync 实例的 id，`cfg` 为 `DefaultConfig()` 表达式，`Config` 的默认值见第 4 节。`FrameLayout` 是常量，须与相机输出的帧布局一致：

An instance written by `xrobot instance add QDU-Robomaster/ArmorDetector --template-arg <FrameLayout>`; `sync` is set to the id of a CameraFrameSync instance, and `cfg` is the `DefaultConfig()` expression, with the defaults of `Config` listed in section 4. `FrameLayout` is a constant and has to match the frame layout of the camera output:

```yaml
constexpr_namespace: AutoAimRunConfig
constexpr_includes:
  - CameraBase.hpp
constexprs:
  FrameLayout:
    type: CameraTypes::FrameLayout
    value: '{.width = 720, .height = 540, .step = 2160, .encoding = CameraTypes::Encoding::BGR8}'
modules:
  - module: QDU-Robomaster/ArmorDetector
    id: armor_detector
    template_args:
      - AutoAimRunConfig::FrameLayout
    args:
      - sync: camera_frame_sync
      - cfg: ArmorDetector<AutoAimRunConfig::FrameLayout>::DefaultConfig()
```

`camera_frame_sync` 是 `QDU-Robomaster/CameraFrameSync` 实例的 id，列在本实例之前，两个实例的 `template_args` 相同。`referee_auto_detect_color` 为 `true` 时，Topic `robot_game_ref` 由其他实例（例如 SharedTopic）提供。

`camera_frame_sync` is the id of a `QDU-Robomaster/CameraFrameSync` instance, listed before this instance, and both instances share the same `template_args`. With `referee_auto_detect_color` set to `true`, the Topic `robot_game_ref` is provided by another instance (for example SharedTopic).

## 7. 依赖与硬件 / Dependencies and Hardware

依赖：

- `QDU-Robomaster/CameraFrameSync`：同步帧输入与原生标定。
- `QDU-Robomaster/VisionPreview`：检测结果预览。
- `xrobot-org/DurationStatistics`：阶段耗时统计。
- `QDU-Robomaster/CameraBase`：帧布局、帧几何与共享图像类型。
- LibXR。
- 外部库：OpenCV 与 Eigen；HailoRT（`find_package(HailoRT QUIET CONFIG)`）和 OpenVINO Runtime（`find_package(OpenVINO QUIET COMPONENTS Runtime)`，通过 `OpenVINO_DIR` 或 `CMAKE_PREFIX_PATH` 提供）按需安装，可只安装其中一个。

硬件：

- HailoRT 模型运行在 Hailo 加速器上。
- OpenVINO 模型运行在 NPU、GPU 或 CPU 上，适用于 NUC、x86 Linux 和 Webots 仿真；这些环境使用 `network.model: ArmorDetectorModel::OPENVINO_640X512`。

Dependencies:

- `QDU-Robomaster/CameraFrameSync`: synchronized frame input and native calibration.
- `QDU-Robomaster/VisionPreview`: detection result preview.
- `xrobot-org/DurationStatistics`: stage duration statistics.
- `QDU-Robomaster/CameraBase`: frame layout, frame geometry and shared image types.
- LibXR.
- External libraries: OpenCV and Eigen; HailoRT (`find_package(HailoRT QUIET CONFIG)`) and the OpenVINO Runtime (`find_package(OpenVINO QUIET COMPONENTS Runtime)`, provided through `OpenVINO_DIR` or `CMAKE_PREFIX_PATH`) are installed as needed, and either one alone is sufficient.

Hardware:

- HailoRT models run on a Hailo accelerator.
- OpenVINO models run on an NPU, GPU or CPU, which suits NUC, x86 Linux and Webots simulation; these environments use `network.model: ArmorDetectorModel::OPENVINO_640X512`.
