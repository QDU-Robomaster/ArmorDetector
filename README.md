# ArmorDetector

`ArmorDetector` 从 `CameraFrameSync` 读取同步后的图像和 IMU，使用与所选模型绑定的 HailoRT
或 OpenVINO 后端检测装甲板四角点，再根据原生传感器标定求出装甲板在相机坐标系下的位姿。
模块输出按值携带共享图像句柄、IMU 和检测结果；逐帧几何始终从共享图像读取。

## 数据流

1. 普通 Topic 回调借用 `const CameraFrameSync<FrameLayoutV>::SyncedFrame*`；有空闲
   `InferSlot` 时复制共享所有权，并直接预处理到该槽的输入缓冲区。三个固定 `InferSlot`
   由两个设备内请求位和一个已预处理候补位组成；三者都忙时立即丢弃最新输入，不等待。
2. 推理线程（`armor-infer`）最多提交两个 Hailo 异步请求，第三个 `InferSlot` 只保留下一张
   已预处理输入，避免设备完成后等待下一次图像回调；完成结果严格按提交顺序交给 output
   fusion。OpenVINO 使用同一流水线的同步推理分支，每个槽独立持有请求和输出 tensor。
3. 单个 output worker 等待两个 `PostSlot` 之一空闲，把后端输出整理为在该槽处理期间有效的
   `CV_32F` 矩阵，再把同步帧和逐帧上下文移动到 `PostSlot`。移动完成后立即释放原
   `InferSlot`，随后在同一线程执行 `DecodeOutput`、NMS、语义过滤、PnP 和发布；没有第三个
   postprocess 线程或队列。
4. 发布边界把图像所有权从 `PostSlot` 移进临时 `DetectedFrame`，把四角点、中心和包围盒
   映射到原生传感器坐标，使用原生 K/D 执行 PnP，并通过普通 Topic 同步发布
   `const DetectedFrame*`。已经接纳的帧不会在内部阶段丢弃。

输入图像可以是 BGR8、RGB8、BGRA8、RGBA8 或 MONO8，统一转换为 BGR 后进入网络预处理。

## 输入输出

输入：

- `sync.SyncedFrameTopicName()`（默认 `<相机图像 topic>_synced`）：
  `const CameraFrameSync<FrameLayoutV>::SyncedFrame*`
- `referee_domain` 域的 `referee_topic`（默认 `host` / `robot_game_ref`），仅在
  `referee_auto_detect_color: true` 时订阅；构造时会等待该 topic 出现

输出：

- `armor_detector` 域的 `armors_frame`：`const DetectedFrame<FrameLayoutV>*`

Topic 中的 `DetectedFrame*` 只在本次同步回调期间有效。异步消费者必须在回调返回前复制
`DetectedFrame`；其中的 `SharedFrame` 会保留 CameraBase 槽位，直到最后一份阶段对象释放。
`FrameGeometry` 不重复存放，消费者只读取当前帧 `SharedFrame` 中的不可变参数。
`DetectedFrame` 包含 `sequence`（CameraFrameSync 帧序号）、`image`、`imu` 和
`detections`（`std::vector<ArmorDetectorResult>`）。

下游权威帧时间是 `SyncedFrame::imu.timestamp_us`，对应 MCU `FRAME_TRIGGER` 的陀螺仪时间。
detector 统计、结果和后续链路都使用该时间；`ImageFrame::timestamp_us` 只作为相机设备
诊断时间保留。

## 模型

`network.model` 固定绑定模型工件、输入输出语义和所需后端，不提供任意组合的独立 backend
开关。当前包含 8 个 Hailo 模型和 1 个 OpenVINO 模型，默认 `INT16_HEAD_L`。缺少所选模型的
后端时初始化明确报错并且不启动流水线，不切换模型。

| 枚举 `ArmorDetectorModel::` | 值 | 后端 | 工件 |
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

`infer/` 目录负责不同模型的适配：

- `infer/ArmorDetectorModelRegistry.hpp`：模型枚举到所需后端和稳定工件的映射
- `infer/ArmorDetectorInt8Model.hpp`：`int8` 线输出语义适配
- `infer/ArmorDetectorInt16Model.hpp`：`int16` 线输出语义适配
- `infer/ArmorDetectorModelAdapter.hpp`：统一的模型适配入口

其中：

- `INT8_HEAD*`：`int8` 六输出 host-tail 语义
- `INT8_GRID*`：`int8` 单输出 `21x6720` 语义
- `INT16_HEAD*`：`int16` 三头 `conv47/54/60` 语义（对外名 `int16-quality*`）
- `INT16_FAST*`：`int16` 三头 `conv47/54/60` 语义，fast 路线

`XR_ARMOR_HEF_PATH` 环境变量可覆盖 Hailo 模型的 HEF 路径，不影响 ONNX。

### OpenVINO：NUC / x86 Linux / Webots

`OPENVINO_640X512` 绑定 OpenVINO 和 `model/armor_detector_640x512.onnx`。模型直接接受 RGB
`uint8 [512,640,3]`，预处理是拉伸到 `640x512` 后 BGR→RGB；不要额外做 NCHW/归一化。输出为
`float32 [1,20160,22]`，沿用颜色、编号和 `[0,3,2,1]` 角点顺序，并对原始 objectness logit
使用 `network.logit_threshold` 后再输出 sigmoid 置信度。共用 22 字段解码适配器不代表把此
ONNX 重新量化为 INT16；Hailo 模型的门限行为不受影响。

OpenVINO 默认在实际枚举出的设备中按 NPU、GPU、CPU 选择；`XR_ARMOR_OPENVINO_DEVICE=CPU`
可显式固定设备。指定设备或模型初始化失败会报错，不改用其他设备或后端。

## 推理模式

HailoRT 默认异步推理，OpenVINO 只支持同步推理。环境变量
`ARMOR_DETECTOR_INFERENCE_MODE=sync` 可强制同步；对 OpenVINO 请求 `async`，或给出
`sync` / `async` 以外的值，都会明确报错并不启动流水线。每个 OpenVINO 槽持有独立请求和输出，下一帧推理不会覆盖仍在
后处理中的上一帧结果。`ARMOR_DETECTOR_INFERENCE_CPU=<n>` 可把推理线程绑定到指定 CPU。

调试用环境变量：`ARMOR_DETECTOR_DUMP_TSV`（导出最终结果 TSV）、
`ARMOR_DETECTOR_DUMP_OUTPUT_F32` / `ARMOR_DETECTOR_DUMP_OUTPUT_FRAME_INDEX`（导出一帧网络输出
矩阵）。

## 结果内容

单个装甲板结果 `ArmorDetectorResult` 包含：

- 颜色、编号、尺寸类型、默认优先级和置信度
- 图像包围盒、中心点和四个角点（左上、右上、右下、左下）
- `center_norm` 和中心到主点的像素距离
- PnP 是否成功
- PnP 平均重投影误差
- OpenCV 相机坐标系下的装甲板位姿

相机坐标系沿用 OpenCV 约定：`x` 向右，`y` 向下，`z` 向前。这里的位姿只描述装甲板相对
相机的位置和朝向，不包含 IMU 姿态融合。

发布的包围盒、中心和四角点使用原生传感器像素坐标；`center_norm`、网络解码、NMS、阈值、
结果 TSV 和 detector preview 仍使用当前帧坐标。这保证 resize、wide decimation 或 ROI 不
改变检测语义。

## 预览与监控

`preview.enabled: true` 时启动实时预览（VisionPreview）。预览只显示本模块当前帧结果叠加，
不录像、不写数据文件。`preview.output_mode: window` 使用 OpenCV 窗口；`raw` / `web` /
`http` / `bmp` 使用 BMP web 推流，取流路径为 `/stream/<web_stream_name>`。默认预览关闭。

`OnMonitor()` 打印前处理、推理线程、后处理和结果填充的累计耗时统计（次数、平均、最小、
最大，单位微秒）。

## 依赖

- `QDU-Robomaster/CameraFrameSync`：同步帧输入和原生标定。
- `QDU-Robomaster/VisionPreview`：检测结果预览。
- `xrobot-org/DurationStatistics`：阶段耗时统计。
- `QDU-Robomaster/CameraBase`：帧布局、geometry 与共享图像类型。
- 外部：OpenCV（必需）；HailoRT（`find_package(HailoRT QUIET CONFIG)`）和 OpenVINO Runtime
  （`find_package(OpenVINO QUIET COMPONENTS Runtime)`，可通过 `OpenVINO_DIR` 或
  `CMAKE_PREFIX_PATH` 提供）均为可选；Eigen。

可以只有其中一个后端、两者都有或均无；均无时仍可构建，但任何模型选择都会在初始化时报错且
不启动 detector 流水线。安装 OpenVINO 不会自动替换默认 Hailo 模型，无 Hailo 的机器应显式
选择 `OPENVINO_640X512`。

## 构造接口

```cpp
template <CameraTypes::FrameLayout FrameLayoutV>
class ArmorDetector;

ArmorDetector(
    Sync& sync,
    Config cfg = DefaultConfig());
```

模板参数：

- `FrameLayoutV`：帧布局，必须与上游 CameraFrameSync 和相机相同。原生标定由
  `CameraFrameSync::Calibration()` 在构造期复制，ROI、下采样和翻转由每帧 `FrameGeometry`
  描述。主检测模型固定使用 `640x512` 输入，原始图像可以是其他尺寸。

依赖：

- `sync`：`CameraFrameSync<FrameLayoutV>&`，即前面的 CameraFrameSync 实例。

配置 `cfg`（`Config`，`DefaultConfig()` 即全部默认值）：

- `detect_color`：`0` 只保留红色，`1` 只保留蓝色（默认），其他值不过滤颜色。
- `network.model`：模型枚举，默认 `ArmorDetectorModel::INT16_HEAD_L`，见上表。
- `network.min_confidence`：最终结果置信度门限，默认 `0.1`。
- `network.enable_quad_check`：是否检查四边形凸性和面积，默认 `true`。
- `network.min_quad_area_px`：四边形最小面积，单位 `px^2`，默认 `16.0`。
- `network.logit_threshold`：网络 objectness 原始 logit 门限，默认 `0.619`。
- `network.nms_threshold`：OpenCV NMS IoU 门限，默认 `0.45`。
- `network.bbox_expand`：NMS 前包围盒扩张比例，默认 `0.1`。
- `network.max_detections`：NMS 后最多保留候选数量，默认 `128`。
- `referee_auto_detect_color`：按裁判系统 robot_id 自动切换敌方颜色（1–99 为红方则打蓝，
  101–199 为蓝方则打红），默认 `false`；收到有效 robot_id 前使用 `detect_color`。
- `referee_domain`：裁判系统摘要包所在 topic domain，默认 `"host"`。
- `referee_topic`：裁判系统摘要包 topic 名，默认 `"robot_game_ref"`。
- `preview`：`VisionPreview::RuntimeParam`，默认关闭，字段见 VisionPreview。
- `number_refine`：仅为兼容保留，运行时忽略。

模型在构造时加载；流水线启动后 `SetConfig()` 会被拒绝。

## 使用

```sh
xrobot module add QDU-Robomaster/ArmorDetector
xrobot setup
xrobot instance add QDU-Robomaster/ArmorDetector
```

`xrobot instance add` 在 `User/xrobot.yaml` 中写入一个实例，依赖项留空，默认值按源码写出；
把 `sync` 填为前面 CameraFrameSync 实例的 id。帧布局用 constexpr 定义，必须与相机输出一致：

```yaml
constexpr_includes:
  - CameraBase.hpp
constexprs:
  FrameLayout:
    type: CameraTypes::FrameLayout
    value: '{.width = 640, .height = 480, .step = 1920, .encoding = CameraTypes::Encoding::BGR8}'
modules:
  - module: QDU-Robomaster/ArmorDetector
    id: armordetector_0
    template_args:
      - ProjectConstexpr::FrameLayout
    args:
      - sync: cameraframesync_0
      - cfg: ArmorDetector<ProjectConstexpr::FrameLayout>::DefaultConfig()
```

`cameraframesync_0` 是 CameraFrameSync 实例的 id，必须在 `modules:` 中列在本实例之前，
并使用同一个 `template_args`。本模块不直接使用 BSP 对象，不需要额外的 `XR_REGISTER`。
ArmorTracker 订阅本模块的 `armor_detector` / `armors_frame` 输出，它必须列在本实例之后。

`cfg` 也可以写成 YAML map（字段名同上，字符串写成 C++ 字符串字面量，枚举写成 C++ 表达式）。
Webots 仿真中选 OpenVINO 模型并打开 Web 预览的示例：

```yaml
cfg:
  detect_color: 2
  network:
    model: 'ArmorDetectorModel::OPENVINO_640X512'
    min_confidence: 0.1
    enable_quad_check: true
    min_quad_area_px: 16.0
    logit_threshold: 0.619
    nms_threshold: 0.45
    bbox_expand: 0.1
    max_detections: 128
  referee_auto_detect_color: false
  referee_domain: '"host"'
  referee_topic: '"robot_game_ref"'
  preview:
    enabled: true
    preview_window_name: '"armor_detector_preview"'
    preview_scale: 0.5
    preview_wait_key_ms: 1
    queue_capacity: 1
    output_mode: '"web"'
    web_bind_address: '"0.0.0.0"'
    web_port: 8080
    web_stream_name: '"armor_detector"'
    max_fps: 30.0
  number_refine: '{}'
```

填好后再次运行 `xrobot setup`，生成 `User/xrobot_main.hpp`。

`xrobot module show .`（在本仓库中）或 `xrobot module show Modules/QDU-Robomaster/ArmorDetector`
（在 BSP 中）打印当前的构造函数。

## 测试

宿主 XRobot 工程启用 `BUILD_TESTING=ON` 后，模块注册 `armor_detector_publish_geometry_test`、
`armor_detector_pipeline_test`、`armor_detector_input_view_test` 和
`armor_detector_backend_test`，用 `ctest` 运行。后端测试覆盖模型绑定、缺失 SDK、枚举稳定性；
有 OpenVINO 时还会用所附 ONNX 在 CPU 上做真实推理、独立输出对照、跨槽并发读取、输入校验与
重配置失效检查，并追加 `armor_detector_unavailable_device_test` 验证显式错误设备不会回退。

无 SDK 构建可用 `-DCMAKE_DISABLE_FIND_PACKAGE_OpenVINO=ON -DCMAKE_DISABLE_FIND_PACKAGE_HailoRT=ON`
验证。相机回放、真实模块图和 Webots 运行验收属于宿主集成测试。
