#pragma once

/**
 * @file ArmorDetectorDetail.hpp
 * @brief ArmorDetector 内部使用的模型常量和轻量几何/语义工具。
 */

/**
 * @brief ArmorDetector 内部实现命名空间。
 *
 * 这些工具不构成跨模块 ABI；对外请使用 ArmorDetectorTypes.hpp 中的结果结构。
 */
namespace armor_detector_detail
{

/**
 * @brief Detector model artifact selected by runtime config.
 */
enum class DetectorModelVariant : uint8_t
{
  QDU_RESIZE512_BGR = 0, ///< Current QDU dense-grid OpenVINO IR.
  SZU_U8_RGB_HWC = 1,   ///< SHtech/SZU UINT8 HWC ONNX, RGB input.
};

/**
 * @brief Model input tensor memory contract used by OpenVINO Infer().
 */
enum class NetworkInputTensorLayout : uint8_t
{
  NHWC_BATCHED = 0, ///< Tensor shape [1,H,W,3].
  HWC = 1,          ///< Tensor shape [H,W,3].
};

/**
 * @brief Model input color contract after BuildNetworkInput().
 */
enum class NetworkInputColor : uint8_t
{
  BGR = 0,
  RGB = 1,
};

/**
 * @brief Decoder contract selected from the loaded model artifact.
 */
enum class NetworkOutputLayout : uint8_t
{
  QDU_GRID21 = 0, ///< [1,21,N] dense-grid relative keypoint output.
  SHTECH22 = 1,  ///< [1,20160,22] absolute SHtech/SZU output.
  UNKNOWN = 2,
};

/**
 * @brief dense-grid keypoint detector 输出网格宽度。
 */
constexpr int direct_keypoint_grid_width = 512;

/**
 * @brief dense-grid keypoint detector 输出网格高度。
 */
constexpr int direct_keypoint_grid_height = 384;

/**
 * @brief detector 模型日志名称。
 */
inline constexpr const char* detector_model_name =
    "direct_keypoint_dense_grid";

/**
 * @brief SHtech/SZU UINT8 model log name.
 */
inline constexpr const char* szu_u8_model_name = "szu_u8_rgb_hwc_shtech22";

/**
 * @brief dense-grid keypoint detector 的输出候选数量。
 */
constexpr int direct_keypoint_candidate_count = 4032;

/**
 * @brief dense-grid keypoint detector 的输出列数。
 */
constexpr int direct_keypoint_output_width = 21;

/**
 * @brief dense-grid keypoint detector 置信度排序后参与交叠抑制的最大候选数。
 */
constexpr int direct_keypoint_keep_topk = 128;

/**
 * @brief SHtech/SZU 512x640 absolute-output model constants.
 */
constexpr int shtech_input_width = 640;
constexpr int shtech_input_height = 512;
constexpr int shtech_candidate_count = 20160;
constexpr int shtech_output_width = 22;
constexpr double shtech_default_logit_threshold = 0.619;
constexpr double shtech_default_confidence_threshold = 0.65;
constexpr double shtech_default_nms_threshold = 0.45;
constexpr double shtech_default_bbox_expand = 0.1;
constexpr int shtech_default_max_detections = 128;

/**
 * @brief Parse runtime model variant text.
 * @param text Config string; empty value keeps the current QDU model.
 * @return Known model variant.
 */
inline DetectorModelVariant ParseDetectorModelVariant(const char* text)
{
  if (text == nullptr || text[0] == '\0')
  {
    return DetectorModelVariant::QDU_RESIZE512_BGR;
  }
  const std::string value(text);
  if (value == "szu_u8_rgb_hwc" || value == "szu_u8_640_rgb_hwc" ||
      value == "szu_u8" || value == "shtech22")
  {
    return DetectorModelVariant::SZU_U8_RGB_HWC;
  }
  return DetectorModelVariant::QDU_RESIZE512_BGR;
}

/**
 * @brief Stable name for logs and diagnostics.
 * @param variant Model variant.
 * @return String literal model variant name.
 */
inline const char* DetectorModelVariantName(DetectorModelVariant variant)
{
  switch (variant)
  {
    case DetectorModelVariant::SZU_U8_RGB_HWC:
      return "szu_u8_rgb_hwc";
    case DetectorModelVariant::QDU_RESIZE512_BGR:
    default:
      return "qdu_resize512_bgr";
  }
}

/**
 * @brief Stable name for decoder logs and diagnostics.
 */
inline const char* NetworkOutputLayoutName(NetworkOutputLayout layout)
{
  switch (layout)
  {
    case NetworkOutputLayout::SHTECH22:
      return "shtech22";
    case NetworkOutputLayout::QDU_GRID21:
      return "qdu_grid21";
    case NetworkOutputLayout::UNKNOWN:
    default:
      return "unknown";
  }
}

/**
 * @brief 同步帧 worker 单次等待超时，单位 ms。
 */
constexpr uint32_t sync_frame_wait_timeout_ms = 100;

/**
 * @brief 周期性指标日志输出帧间隔。
 */
constexpr uint32_t metrics_log_period = 30;

/**
 * @brief detector 网络张量尺寸。
 */
struct NetworkInputShape
{
  int width{0};   ///< 网络张量宽度，单位 px。
  int height{0};  ///< 网络张量高度，单位 px。
};

/**
 * @brief 判断 dense-grid 网络张量尺寸是否满足当前 decoder 假设。
 * @param shape 输入宽高。
 * @return 宽高均为正且是最大检测 stride 32 的整数倍时返回 true。
 */
inline bool IsValidNetworkInputShape(const NetworkInputShape& shape)
{
  return shape.width > 0 && shape.height > 0 &&
         shape.width % 32 == 0 && shape.height % 32 == 0;
}

/**
 * @brief 计算 dense-grid keypoint detector 在给定网格尺寸下的候选数量。
 *
 * 当前模型有 stride 8/16/32 三个检测层，每个网格单元一个候选。
 *
 * @param shape 输入宽高。
 * @return 输出候选数量；尺寸非法时返回 0。
 */
inline int DirectKeypointCandidateCount(const NetworkInputShape& shape)
{
  if (!IsValidNetworkInputShape(shape))
  {
    return 0;
  }

  return (shape.width / 8) * (shape.height / 8) +
         (shape.width / 16) * (shape.height / 16) +
         (shape.width / 32) * (shape.height / 32);
}

/**
 * @brief 输出网格坐标到原始图像坐标的映射。
 */
struct NetworkInputMapping
{
  double x_scale{1.0}; ///< 网络张量 x 坐标还原到源图像的比例。
  double y_scale{1.0}; ///< 网络张量 y 坐标还原到源图像的比例。
  int x_offset{0};     ///< 输出网格到网络张量的 x 偏移。
  int y_offset{0};     ///< 输出网格到网络张量的 y 偏移。

  /**
   * @brief 将输出网格上的点还原到原始图像平面。
   * @param x 输出网格坐标 x。
   * @param y 输出网格坐标 y。
   * @return 源图像像素坐标。
   */
  [[nodiscard]] cv::Point2f MapToSource(float x, float y) const
  {
    return {static_cast<float>(
                static_cast<double>(x + static_cast<float>(x_offset)) * x_scale),
            static_cast<float>(
                static_cast<double>(y + static_cast<float>(y_offset)) * y_scale)};
  }
};

/**
 * @brief 当前模型输出网格尺寸。
 * @return 输出网格宽高。
 */
inline NetworkInputShape DirectKeypointGridShape()
{
  return {direct_keypoint_grid_width, direct_keypoint_grid_height};
}

/**
 * @brief 计算输出网格到网络张量坐标的偏移。
 * @param input_shape 当前网络张量尺寸。
 * @param grid_shape 当前输出网格尺寸。
 * @return x/y 方向偏移；尺寸不匹配时按 0 兜底。
 */
inline cv::Point NetworkGridOffset(const NetworkInputShape& input_shape,
                                   const NetworkInputShape& grid_shape)
{
  if (input_shape.width < grid_shape.width ||
      input_shape.height < grid_shape.height)
  {
    return {};
  }
  return {(input_shape.width - grid_shape.width) / 2,
          (input_shape.height - grid_shape.height) / 2};
}

/**
 * @brief dense-grid keypoint 输出的字段布局。
 */
struct DirectKeypointOutputLayout
{
  static constexpr int point_begin = 0;       ///< 角点偏移起始列。
  static constexpr int objectness_index = 8;  ///< 目标置信度列。
  static constexpr int number_begin = 9;      ///< 编号分类起始列，闭区间。
  static constexpr int number_end = 17;       ///< 编号分类结束列，开区间。
  static constexpr int color_begin = 17;      ///< 颜色分类起始列，闭区间。
  static constexpr int color_end = 19;        ///< 颜色分类结束列，开区间。
  static constexpr int size_begin = 19;       ///< 尺寸分类起始列，闭区间。
  static constexpr int size_end = 21;         ///< 尺寸分类结束列，开区间。
};

/**
 * @brief dense-grid 输出行对应的网格单元。
 */
struct DirectKeypointGridCell
{
  int center_x{0}; ///< 网格中心 x，单位为输出网格像素。
  int center_y{0}; ///< 网格中心 y，单位为输出网格像素。
  int stride{8};   ///< 当前检测层 stride。
};

/**
 * @brief 将 dense-grid 输出行号转换为网格中心和 stride。
 * @param shape 当前输出网格尺寸。
 * @param row 输出行号，范围为 [0, DirectKeypointCandidateCount(shape))。
 * @return 对应网格单元；越界时返回最后一层的兜底单元。
 */
inline DirectKeypointGridCell DirectKeypointGridCellForRow(
    const NetworkInputShape& shape, int row)
{
  if (row < 0 || !IsValidNetworkInputShape(shape))
  {
    return {};
  }

  const int stride8_cols = shape.width / 8;
  const int stride8_count = stride8_cols * (shape.height / 8);
  const int stride16_cols = shape.width / 16;
  const int stride16_count = stride16_cols * (shape.height / 16);

  if (row < stride8_count)
  {
    return {(row % stride8_cols) * 8, (row / stride8_cols) * 8, 8};
  }

  row -= stride8_count;
  if (row < stride16_count)
  {
    return {(row % stride16_cols) * 16, (row / stride16_cols) * 16, 16};
  }

  row -= stride16_count;
  const int stride32_cols = shape.width / 32;
  return {(row % stride32_cols) * 32, (row / stride32_cols) * 32, 32};
}

/**
 * @brief dense-grid 模型输出矩阵的轻量视图。
 *
 * 当前模型输出为 `[21,N]`，行是字段，列是候选；N 由输出网格尺寸决定。
 */
class DirectKeypointOutputView
{
 public:
  DirectKeypointOutputView(const cv::Mat& output,
                           const NetworkInputShape& grid_shape)
      : output_(output),
        candidate_count_(DirectKeypointCandidateCount(grid_shape))
  {
    if (output_.type() != CV_32F || output_.dims != 2)
    {
      return;
    }

    if (output_.rows == direct_keypoint_output_width &&
        output_.cols == candidate_count_)
    {
      valid_ = true;
    }
  }

  [[nodiscard]] bool Valid() const { return valid_; }

  [[nodiscard]] int CandidateCount() const
  {
    return valid_ ? candidate_count_ : 0;
  }

  [[nodiscard]] float At(int row, int field) const
  {
    return output_.at<float>(field, row);
  }

 private:
  const cv::Mat& output_;
  int candidate_count_{0};
  bool valid_{false};
};

/**
 * @brief SHtech/SZU 20160x22 absolute detector output view.
 *
 * Most OpenVINO ONNX runs expose [1,20160,22]. The transposed 22x20160 form
 * is accepted as a defensive check because some dump tools normalize outputs.
 */
class Shtech22OutputView
{
 public:
  explicit Shtech22OutputView(const cv::Mat& output) : output_(output)
  {
    if (output_.type() != CV_32F || output_.dims != 2)
    {
      return;
    }
    if (output_.rows == shtech_candidate_count &&
        output_.cols == shtech_output_width)
    {
      valid_ = true;
      transposed_ = false;
      return;
    }
    if (output_.rows == shtech_output_width &&
        output_.cols == shtech_candidate_count)
    {
      valid_ = true;
      transposed_ = true;
    }
  }

  [[nodiscard]] bool Valid() const { return valid_; }

  [[nodiscard]] int CandidateCount() const
  {
    return valid_ ? shtech_candidate_count : 0;
  }

  [[nodiscard]] float At(int row, int field) const
  {
    return transposed_ ? output_.at<float>(field, row)
                       : output_.at<float>(row, field);
  }

 private:
  const cv::Mat& output_;
  bool valid_{false};
  bool transposed_{false};
};

/**
 * @brief 在输出视图某一候选的指定字段范围内取 argmax。
 * @param output 输出视图。
 * @param row 行号。
 * @param begin 起始列，闭区间。
 * @param end 结束列，开区间。
 * @return 最大值列号减去 begin 后的类别 id。
 */
inline int ArgMaxRowRange(const DirectKeypointOutputView& output, int row,
                          int begin, int end)
{
  int best = begin;
  float best_value = output.At(row, begin);
  for (int index = begin + 1; index < end; ++index)
  {
    const float value = output.At(row, index);
    if (value > best_value)
    {
      best_value = value;
      best = index;
    }
  }
  return best - begin;
}

/**
 * @brief Argmax over one SHtech/SZU candidate row.
 */
inline int ArgMaxShtechRange(const Shtech22OutputView& output, int row,
                             int begin, int end)
{
  int best = begin;
  float best_value = output.At(row, begin);
  for (int index = begin + 1; index < end; ++index)
  {
    const float value = output.At(row, index);
    if (value > best_value)
    {
      best_value = value;
      best = index;
    }
  }
  return best - begin;
}

/**
 * @brief Numerically stable sigmoid for SHtech objectness logits.
 */
inline float Sigmoid(float value)
{
  return 1.0F / (1.0F + std::exp(-value));
}

/**
 * @brief 判断点坐标是否都是有限数。
 * @param point 待检查点。
 * @return x/y 均有限时返回 true。
 */
inline bool FinitePoint(const cv::Point2f& point)
{
  return std::isfinite(point.x) && std::isfinite(point.y);
}

/**
 * @brief 计算四边形面积。
 * @param points 顺序排列的四边形角点。
 * @return 绝对面积，单位 px^2。
 */
inline double QuadArea(const std::array<cv::Point2f, 4>& points)
{
  double area = 0.0;
  for (std::size_t i = 0; i < points.size(); ++i)
  {
    const auto& a = points[i];
    const auto& b = points[(i + 1U) % points.size()];
    area += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
  }
  return std::abs(area) * 0.5;
}

/**
 * @brief 判断四个点是否构成非退化凸四边形。
 * @param points 顺序排列的四边形角点。
 * @return 凸且非共线时返回 true。
 */
inline bool IsConvexQuad(const std::array<cv::Point2f, 4>& points)
{
  int sign = 0;
  for (std::size_t i = 0; i < points.size(); ++i)
  {
    const cv::Point2f a = points[i];
    const cv::Point2f b = points[(i + 1U) % points.size()];
    const cv::Point2f c = points[(i + 2U) % points.size()];
    const cv::Point2f ab = b - a;
    const cv::Point2f bc = c - b;
    const double cross = static_cast<double>(ab.x) * bc.y -
                         static_cast<double>(ab.y) * bc.x;
    if (std::abs(cross) < 1e-6)
    {
      continue;
    }
    const int current_sign = cross > 0.0 ? 1 : -1;
    if (sign == 0)
    {
      sign = current_sign;
      continue;
    }
    if (sign != current_sign)
    {
      return false;
    }
  }
  return sign != 0;
}

/**
 * @brief 检查角点是否可用于后续 PnP。
 * @param points 顺序排列的四边形角点。
 * @param min_area 最小面积门限，单位 px^2。
 * @return 点有限、凸且面积足够时返回 true。
 */
inline bool IsUsableQuad(const std::array<cv::Point2f, 4>& points,
                         double min_area)
{
  for (const auto& point : points)
  {
    if (!FinitePoint(point))
    {
      return false;
    }
  }
  return IsConvexQuad(points) && QuadArea(points) >= min_area;
}

/**
 * @brief 将 CameraTypes::Encoding 转成 OpenCV Mat 类型。
 * @param encoding 相机图像编码。
 * @return OpenCV CV_8UC* 类型；不支持时返回 -1。
 */
inline int CvTypeFromEncoding(CameraTypes::Encoding encoding)
{
  switch (encoding)
  {
    case CameraTypes::Encoding::RGB8:
    case CameraTypes::Encoding::BGR8:
      return CV_8UC3;
    case CameraTypes::Encoding::RGBA8:
    case CameraTypes::Encoding::BGRA8:
      return CV_8UC4;
    case CameraTypes::Encoding::MONO8:
      return CV_8UC1;
    default:
      return -1;
  }
}

/**
 * @brief 按相机编码把图像转换成 BGR 通道顺序。
 * @param input 输入图像。
 * @param encoding 输入图像编码。
 * @return BGR 图像；原本就是 BGR/MONO 时返回共享数据视图。
 */
inline cv::Mat ConvertToBgrWithEncoding(const cv::Mat& input,
                                        CameraTypes::Encoding encoding)
{
  switch (encoding)
  {
    case CameraTypes::Encoding::RGB8:
    {
      cv::Mat output;
      cv::cvtColor(input, output, cv::COLOR_RGB2BGR);
      return output;
    }
    case CameraTypes::Encoding::BGRA8:
    {
      cv::Mat output;
      cv::cvtColor(input, output, cv::COLOR_BGRA2BGR);
      return output;
    }
    case CameraTypes::Encoding::RGBA8:
    {
      cv::Mat output;
      cv::cvtColor(input, output, cv::COLOR_RGBA2BGR);
      return output;
    }
    default:
      return input;
  }
}

/**
 * @brief 将配置中的颜色编号转为 ArmorColor。
 * @param detect_color 0=red，1=blue，其他=不限制。
 * @return 对应目标颜色；不限制时返回 UNKNOWN。
 */
inline ArmorColor detect_color_from_config(int detect_color)
{
  if (detect_color == 0)
  {
    return ArmorColor::RED;
  }
  if (detect_color == 1)
  {
    return ArmorColor::BLUE;
  }
  return ArmorColor::UNKNOWN;
}

/**
 * @brief 将 dense-grid keypoint 模型的颜色类别 id 转为 ArmorColor。
 * @param color_id 模型颜色类别 id。
 * @return detector 统一颜色枚举。
 */
inline ArmorColor color_from_direct_keypoint_id(int color_id)
{
  if (color_id == 0)
  {
    return ArmorColor::RED;
  }
  if (color_id == 1)
  {
    return ArmorColor::BLUE;
  }
  return ArmorColor::UNKNOWN;
}

/**
 * @brief 将 dense-grid keypoint 模型的 8 类编号 id 转为 ArmorNumber。
 * @param class_id 模型原始 class id，范围通常为 [0, 7]。
 * @return detector 统一编号枚举。
 */
inline ArmorNumber number_from_direct_keypoint_class_id(int class_id)
{
  switch (class_id)
  {
    case 0:
      return ArmorNumber::GUARD;
    case 1:
      return ArmorNumber::ONE;
    case 2:
      return ArmorNumber::TWO;
    case 3:
      return ArmorNumber::THREE;
    case 4:
      return ArmorNumber::FOUR;
    case 5:
      return ArmorNumber::FIVE;
    case 6:
      return ArmorNumber::OUTPOST;
    case 7:
      return ArmorNumber::BASE;
    default:
      return ArmorNumber::UNKNOWN;
  }
}

/**
 * @brief 将网络角点顺序映射到 detector/PnP 统一顺序。
 *
 * detector 统一使用左上、右上、右下、左下。
 *
 * @param keypoints 网络输出的四点。
 * @return 左上、右上、右下、左下顺序。
 */
inline std::array<cv::Point2f, 4> direct_keypoint_declared_to_canonical(
    const std::array<cv::Point2f, 4>& keypoints)
{
  return {keypoints[0], keypoints[3], keypoints[2], keypoints[1]};
}

/**
 * @brief 将 OpenCV PnP rvec/tvec 转成 LibXR Transform。
 * @param rvec OpenCV 旋转向量。
 * @param tvec OpenCV 平移向量，单位 m。
 * @return 相机坐标系下的 LibXR 位姿。
 */
inline LibXR::Transform<double> make_pose(const cv::Mat& rvec, const cv::Mat& tvec)
{
  cv::Mat rotation_cv;
  cv::Rodrigues(rvec, rotation_cv);

  Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      rotation_matrix(row, col) = rotation_cv.at<double>(row, col);
    }
  }

  return LibXR::Transform<double>(
      LibXR::Quaternion<double>(rotation_matrix),
      LibXR::Position<double>(tvec.at<double>(0), tvec.at<double>(1),
                              tvec.at<double>(2)));
}

/**
 * @brief 计算四边形中心。
 * @param points 顺序四角点。
 * @return 四点均值。
 */
inline cv::Point2f quad_center(const std::array<cv::Point2f, 4>& points)
{
  return (points[0] + points[1] + points[2] + points[3]) * 0.25F;
}

/**
 * @brief 根据四个浮点角点构造像素包围盒。
 * @param points 四角点。
 * @return 至少 1x1 的整数像素包围盒。
 */
inline cv::Rect bounding_rect_from_points(
    const std::array<cv::Point2f, 4>& points)
{
  float min_x = points[0].x;
  float max_x = points[0].x;
  float min_y = points[0].y;
  float max_y = points[0].y;
  for (const auto& point : points)
  {
    min_x = std::min(min_x, point.x);
    max_x = std::max(max_x, point.x);
    min_y = std::min(min_y, point.y);
    max_y = std::max(max_y, point.y);
  }

  return {static_cast<int>(min_x), static_cast<int>(min_y),
          std::max(1, static_cast<int>(max_x - min_x)),
          std::max(1, static_cast<int>(max_y - min_y))};
}

/**
 * @brief Bounding box expanded symmetrically by a fraction of its width/height.
 */
inline cv::Rect expanded_bounding_rect_from_points(
    const std::array<cv::Point2f, 4>& points, double expand)
{
  float min_x = points[0].x;
  float max_x = points[0].x;
  float min_y = points[0].y;
  float max_y = points[0].y;
  for (const auto& point : points)
  {
    min_x = std::min(min_x, point.x);
    max_x = std::max(max_x, point.x);
    min_y = std::min(min_y, point.y);
    max_y = std::max(max_y, point.y);
  }

  const float width = std::max(1.0F, max_x - min_x);
  const float height = std::max(1.0F, max_y - min_y);
  const float dx = static_cast<float>(expand) * width;
  const float dy = static_cast<float>(expand) * height;
  return {static_cast<int>(min_x - dx), static_cast<int>(min_y - dy),
          std::max(1, static_cast<int>(width + 2.0F * dx)),
          std::max(1, static_cast<int>(height + 2.0F * dy))};
}

/**
 * @brief SHtech/SZU color mapping.
 *
 * Raw 0/1 map to the public blue/red enum. Raw 2/3 are not accepted by the
 * production detector path and must be rejected upstream of candidate publish.
 */
inline ArmorColor color_from_shtech_id(int color_id)
{
  if (color_id == 0)
  {
    return ArmorColor::BLUE;
  }
  if (color_id == 1)
  {
    return ArmorColor::RED;
  }
  return ArmorColor::UNKNOWN;
}

/**
 * @brief SHtech/SZU 9-class tag mapping into ArmorNumber.
 */
inline ArmorNumber number_from_shtech_class_id(int class_id)
{
  int tag = class_id;
  if (tag == 7 || tag == 8)
  {
    tag = 9;
  }
  else if (tag == 0)
  {
    tag = 7;
  }
  else if (tag == 6)
  {
    tag = 8;
  }

  int qdu_number = tag;
  if (tag >= 1 && tag <= 7)
  {
    qdu_number = tag - 1;
  }

  switch (qdu_number)
  {
    case 0:
      return ArmorNumber::ONE;
    case 1:
      return ArmorNumber::TWO;
    case 2:
      return ArmorNumber::THREE;
    case 3:
      return ArmorNumber::FOUR;
    case 4:
      return ArmorNumber::FIVE;
    case 5:
      return ArmorNumber::OUTPOST;
    case 6:
      return ArmorNumber::GUARD;
    case 7:
      return ArmorNumber::BASE;
    default:
      return ArmorNumber::UNKNOWN;
  }
}

/**
 * @brief SHtech/SZU detector type rule used by the validated artifact matrix.
 */
inline ArmorType shtech_type_from_number(ArmorNumber number)
{
  if (number == ArmorNumber::ONE || number == ArmorNumber::TWO)
  {
    return ArmorType::LARGE;
  }
  if (ArmorNumberIsKnown(number))
  {
    return ArmorType::SMALL;
  }
  return ArmorType::INVALID;
}

}  // namespace armor_detector_detail

namespace detail = armor_detector_detail;
