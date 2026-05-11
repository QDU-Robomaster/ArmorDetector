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
 * @brief 当前 detector 模型输入宽度，单位 px。
 */
constexpr int model_input_width = 640;

/**
 * @brief 当前 detector 模型输入高度，单位 px。
 */
constexpr int model_input_height = 512;

/**
 * @brief 当前 detector 输出候选数量。
 */
constexpr int model_candidate_count = 20160;

/**
 * @brief 当前 detector 单候选字段数量。
 */
constexpr int model_output_width = 22;

/**
 * @brief 当前 detector objectness 原始 logit 默认门限。
 */
constexpr double default_logit_threshold = 0.619;

/**
 * @brief NMS 使用的 bbox 扩张比例默认值。
 */
constexpr double default_bbox_expand = 0.1;

/**
 * @brief NMS 后最多保留候选数量默认值。
 */
constexpr int default_max_detections = 128;

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
 * @brief 判断网络张量尺寸是否满足当前 detector 模型约定。
 * @param shape 输入宽高。
 * @return 宽高等于模型固定输入尺寸时返回 true。
 */
inline bool IsValidNetworkInputShape(const NetworkInputShape& shape)
{
  return shape.width == model_input_width && shape.height == model_input_height;
}

/**
 * @brief 网络输入坐标到原始图像坐标的映射。
 */
struct NetworkInputMapping
{
  double x_scale{1.0}; ///< 网络张量 x 坐标还原到源图像的比例。
  double y_scale{1.0}; ///< 网络张量 y 坐标还原到源图像的比例。

  /**
   * @brief 将网络张量上的点还原到原始图像平面。
   * @param x 网络张量坐标 x。
   * @param y 网络张量坐标 y。
   * @return 源图像像素坐标。
   */
  [[nodiscard]] cv::Point2f MapToSource(float x, float y) const
  {
    return {static_cast<float>(static_cast<double>(x) * x_scale),
            static_cast<float>(static_cast<double>(y) * y_scale)};
  }
};

/**
 * @brief 当前 detector 20160x22 输出矩阵视图。
 *
 * OpenVINO exposes [1,20160,22] as a 2D [20160,22] Mat in this wrapper. The
 * transposed [22,20160] form is accepted to keep the view tolerant of output
 * adapters that flatten dimensions differently.
 */
class ModelOutputView
{
 public:
  explicit ModelOutputView(const cv::Mat& output) : output_(output)
  {
    if (output_.type() != CV_32F || output_.dims != 2)
    {
      return;
    }
    if (output_.rows == model_candidate_count &&
        output_.cols == model_output_width)
    {
      valid_ = true;
      transposed_ = false;
      return;
    }
    if (output_.rows == model_output_width &&
        output_.cols == model_candidate_count)
    {
      valid_ = true;
      transposed_ = true;
    }
  }

  [[nodiscard]] bool Valid() const { return valid_; }

  [[nodiscard]] int CandidateCount() const
  {
    return valid_ ? model_candidate_count : 0;
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
inline int ArgMaxOutputRange(const ModelOutputView& output, int row,
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
 * @brief Numerically stable sigmoid for objectness logits.
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
  for (std::size_t index = 0; index < points.size(); ++index)
  {
    const auto& current = points[index];
    const auto& next = points[(index + 1U) % points.size()];
    area += static_cast<double>(current.x) * static_cast<double>(next.y) -
            static_cast<double>(next.x) * static_cast<double>(current.y);
  }
  return std::abs(area) * 0.5;
}

/**
 * @brief 判断四边形是否凸。
 * @param points 顺序排列的四边形角点。
 * @return 四点构成严格同向凸四边形时返回 true。
 */
inline bool IsConvexQuad(const std::array<cv::Point2f, 4>& points)
{
  int sign = 0;
  for (std::size_t index = 0; index < points.size(); ++index)
  {
    const cv::Point2f ab = points[(index + 1U) % points.size()] - points[index];
    const cv::Point2f bc =
        points[(index + 2U) % points.size()] - points[(index + 1U) % points.size()];
    const double cross =
        static_cast<double>(ab.x) * static_cast<double>(bc.y) -
        static_cast<double>(ab.y) * static_cast<double>(bc.x);
    if (std::abs(cross) < 1e-6)
    {
      return false;
    }

    const int current_sign = cross > 0.0 ? 1 : -1;
    if (sign == 0)
    {
      sign = current_sign;
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
 * @brief 将网络角点顺序映射到 detector/PnP 统一顺序。
 *
 * detector 统一使用左上、右上、右下、左下。
 *
 * @param keypoints 网络输出的四点。
 * @return 左上、右上、右下、左下顺序。
 */
inline std::array<cv::Point2f, 4> model_points_to_canonical(
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
 * @brief 当前 detector 模型颜色类别映射。
 *
 * Raw 0/1 map to the public blue/red enum. Raw 2/3 are not accepted by the
 * production detector path and must be rejected upstream of candidate publish.
 */
inline ArmorColor color_from_model_id(int color_id)
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
 * @brief 当前 detector 模型 9-class tag 映射到 ArmorNumber。
 */
inline ArmorNumber number_from_model_class_id(int class_id)
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

  int public_number = tag;
  if (tag >= 1 && tag <= 7)
  {
    public_number = tag - 1;
  }

  switch (public_number)
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
 * @brief 当前 detector 模型输出编号对应的尺寸先验。
 */
inline ArmorType type_from_model_number(ArmorNumber number)
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
