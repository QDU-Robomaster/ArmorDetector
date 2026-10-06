# ArmorDetector

装甲板检测：v4 模型直接吃原始 Bayer，解码、NMS、编号分类、颜色过滤，发布检测帧 / Armor detection that runs the v4 model on raw Bayer, decodes, applies NMS, classifies numbers, filters colours and publishes detected frames

## 1. 模块作用 / Purpose

ArmorDetector 订阅 `<相机名>_synced`，把 640×512 原始 BayerRG8 帧直接送入 v4 检测模型，得到每块装甲板的颜色、大小、置信度和四个角点，再用 num-v1 分类器给出编号，过滤掉不是对方颜色和不是装甲板的检测，把角点换算到原生像素后发布 `DetectedFrame{synced, armors}` 到 `<相机名>_detected`。收进的每一帧都发布一次，没有装甲板时 `armors` 为空。

ArmorDetector subscribes to `<camera>_synced`, feeds the 640×512 raw BayerRG8 frame straight into the v4 detector, which gives each armor's colour, size, confidence and four corners, classifies its number with num-v1, drops detections that are not the opponent's colour or not an armor, converts the corners to native pixels and publishes `DetectedFrame{synced, armors}` on `<camera>_detected`. Every accepted frame is published once; `armors` is empty when nothing is found.

## 2. 模型 / Models

模型来自私有仓库 QDU-Robomaster/armor-models 的 Release，不进入本仓库。部署时把该仓库克隆到 BSP 根目录，用其中的脚本下载：

The models come from the Releases of the private repository QDU-Robomaster/armor-models and never enter this repository. For deployment that repository is cloned into the BSP root and its script downloads them:

```bash
armor-models/scripts/fetch_model.sh det-v4.0 armor-models/model_private
armor-models/scripts/fetch_model.sh num-v1.0 armor-models/model_private
```

| 文件 / File | 后端 / Backend |
| --- | --- |
| `armor_det_v4.hef`、`armor_det_v4_near.hef` | Hailo-8L（HailoRT） |
| `armor_det_v4.onnx` | OpenVINO（CPU、GPU、NPU） |
| `armor_num_v1.onnx`、`armor_num_v1_slim.onnx` | OpenCV DNN（CPU） |

后端由检测模型的扩展名决定。启动时按文件名核对 SHA256，文件缺失、不认识或校验不符即致命退出，并提示下载命令。

The backend follows the detector model's extension. At start-up each file's SHA256 is checked against the known value for its name; a missing, unknown or mismatching file is fatal and the log names the download command.

## 3. 处理流程 / Processing

```
synced 回调 → 一格信箱（满则丢新帧，不阻塞上游）
提交线程   → 取一个空闲推理槽，拷入 Bayer 字节，提交推理，放进在途队列
后处理线程 → 按提交顺序等推理完成 → 解码 → NMS → 颜色过滤 → 编号 → 原生坐标 → 发布
```

- 解码：两个尺度（stride 8、16）合在一起，`sigmoid(obj) > min_confidence` 的格子取颜色、大小的 argmax，角点为 `偏移 × 4 × stride + 格心`，格心 `((j + 0.5)·stride − 0.5, (i + 0.5)·stride − 0.5)`；角点不做多格加权。
- NMS：四个角点的外接框，`cv::dnn::NMSBoxes`，阈值 `nms_iou`，最多 50 个；之后按分数做中心包含去重：一个检测的中心落在已保留检测的四边形里，或它的四边形包含已保留检测的中心，就视为同一块板的重复框丢弃。
- 编号：灰度图（`COLOR_BayerBG2GRAY`）按四个角点矫正成 36×40，减均值、除以（标准差 + 4），9 类取 argmax；`negative` 的检测丢弃。
- 角点顺序为左上、左下、右下、右上（灯条四端点），按帧几何换算到原生像素。

- Decoding: both scales (stride 8 and 16) together; cells with `sigmoid(obj) > min_confidence` take the argmax of colour and size, corners are `offset × 4 × stride + cell centre` with the centre at `((j + 0.5)·stride − 0.5, (i + 0.5)·stride − 0.5)`; corners are not averaged over cells.
- NMS: the corners' bounding boxes, `cv::dnn::NMSBoxes`, threshold `nms_iou`, at most 50; then centre-containment suppression in score order: a detection whose centre lies in a kept quad, or whose quad contains a kept centre, is a duplicate on the same plate and is dropped.
- Number: the grey image (`COLOR_BayerBG2GRAY`) is rectified to 36×40 by the four corners, mean-subtracted and divided by (standard deviation + 4); argmax over 9 classes; `negative` detections are dropped.
- Corners are ordered top-left, bottom-left, bottom-right, top-right (light-bar ends) and converted to native pixels with the frame geometry.

`inflight` 是同时在推理的帧数。各后端在真实输入上的实测（640×512，含解码）：

`inflight` is the number of frames in inference at once. Measured on real input (640×512, decoding included):

| 后端 / Backend | inflight 1：延迟 p50 / 吞吐 | inflight 2：吞吐 |
| --- | --- | --- |
| Hailo-8L（PI-HAILO-13T） | 4.19 ms / 235 fps | 235 fps |
| OpenVINO NPU（Meteor Lake） | 2.43 ms / 404 fps | 490 fps |
| OpenVINO GPU（Meteor Lake 核显） | 7.01 ms / 143 fps | 141 fps |
| OpenVINO CPU（Meteor Lake） | 14.5 ms / 68 fps | 81 fps |

Hailo 的吞吐在 inflight 1 时已到硬件上限，用 1；OpenVINO NPU 用 2。

Hailo already reaches its hardware limit with inflight 1, so it uses 1; the OpenVINO NPU uses 2.

## 4. 颜色 / Colour

`target_color` 为 `RED`、`BLUE` 或 `FROM_REFEREE`。`FROM_REFEREE` 订阅 `host` 域的裁判系统摘要包 `robot_game_ref`，按首字节的本机 robot_id 取对方颜色（1–99 为红方，101–199 为蓝方）；收到第一包之前不发布任何装甲板。

`target_color` is `RED`, `BLUE` or `FROM_REFEREE`. `FROM_REFEREE` subscribes to the referee summary `robot_game_ref` in the `host` domain and takes the opponent of the robot_id in its first byte (1–99 red, 101–199 blue); no armor is published before the first packet.

## 5. 配置示例 / Configuration Example

```yaml
modules:
  - module: QDU-Robomaster/ArmorDetector
    id: detector
    args:
      - settings:
          camera_name: "gimbal"
          model_dir: "armor-models/model_private"
          detector_model: "armor_det_v4.hef"
          number_model: "armor_num_v1.onnx"
          openvino_device: ""
          min_confidence: 0.4F
          nms_iou: 0.3F
          target_color: TargetColor::FROM_REFEREE
          inflight: 1
```

哨兵使用 `detector_model: "armor_det_v4.onnx"`、`openvino_device: "NPU"`、`inflight: 2`。

The sentry uses `detector_model: "armor_det_v4.onnx"`, `openvino_device: "NPU"` and `inflight: 2`.

## 6. 统计 / Statistics

`OnMonitor` 打印发布帧数、信箱满丢弃数、装甲板数、`negative` 数、推理失败数，以及四段本地耗时（DurationStatistics）：拷入（preprocess）、等推理结果（inference）、解码到编号（postprocess）、发布（result）。

`OnMonitor` prints published frames, mailbox drops, armors, `negative` detections, inference failures and four local durations (DurationStatistics): copy-in (preprocess), waiting for the result (inference), decoding to numbers (postprocess) and publishing (result).

## 7. 测试 / Tests

- `tests/decoder_test.cpp`：用构造的张量检查解码（门限、格心、两个尺度、NMS、中心包含去重、量化 HWC 视图），SHA-256 标准向量，编号分类器的输入小图。不需要模型文件。
- `tests/model_test.cpp`：设置 `ARMOR_MODELS_DIR`（模型文件）与 `ARMOR_DETECTOR_GOLDEN_DIR`（Python 参考生成的帧与 `gold.txt`）后运行，否则跳过。与 Python 参考（ONNX Runtime + `rm_model.decode` + num-v1）逐项对比，再把整条模块跑一遍。

- `tests/decoder_test.cpp`: decoding on constructed tensors (threshold, cell centres, both scales, NMS, centre containment, the quantised HWC view), SHA-256 test vectors and the number classifier's patch. No model files needed.
- `tests/model_test.cpp`: runs when `ARMOR_MODELS_DIR` (model files) and `ARMOR_DETECTOR_GOLDEN_DIR` (frames and `gold.txt` from the Python reference) are set, and is skipped otherwise. It compares item by item with the Python reference (ONNX Runtime + `rm_model.decode` + num-v1) and then runs the whole Module.

## 8. 依赖 / Dependencies

CameraBase、AutoAimTypes、DurationStatistics、LibXR、OpenCV（core、imgproc、dnn）；可选 HailoRT 4.24、OpenVINO 2025。

CameraBase, AutoAimTypes, DurationStatistics, LibXR, OpenCV (core, imgproc, dnn); optionally HailoRT 4.24 and OpenVINO 2025.
