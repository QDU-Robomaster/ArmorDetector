#pragma once

// 仅供 ArmorDetector.hpp 在类声明之后包含。
namespace armor_detector_detail
{
// 只保留与模块主体相关的低层工具，避免 detector 主逻辑里充满魔法数字。
constexpr double deg2rad = CV_PI / 180.0;
constexpr int yolo_input_size = 640;
constexpr int shtech_szu0526_input_width = 640;
constexpr int shtech_szu0526_input_height = 512;
constexpr double shtech_szu0526_nms_box_padding_ratio = 0.10;
constexpr uint32_t sync_frame_wait_timeout_ms = 100;
constexpr uint32_t metrics_log_period = 30;
constexpr size_t sync_frame_thread_stack_size = 1024U * 128U;

struct NetworkInputShape
{
  int width{yolo_input_size};
  int height{yolo_input_size};
};

enum class DetectorProfile : uint8_t
{
  SP_YOLOV5 = 0,
  SHTECH_SZU0526 = 1,
};

enum class ResizeMode : uint8_t
{
  PROPORTIONAL,
  STRETCH,
};

struct DetectorProfileSpec
{
  const char* name{"sp_yolov5"};
  NetworkInputShape input_shape{};
  ResizeMode resize_mode{ResizeMode::PROPORTIONAL};
  double nms_box_padding_ratio{0.0};
};

inline DetectorProfileSpec ProfileSpecFor(DetectorProfile profile)
{
  switch (profile)
  {
    case DetectorProfile::SP_YOLOV5:
      return {};
    case DetectorProfile::SHTECH_SZU0526:
      return {"shtech_szu0526",
              {shtech_szu0526_input_width, shtech_szu0526_input_height},
              ResizeMode::STRETCH,
              shtech_szu0526_nms_box_padding_ratio};
    default:
      return {"unknown", {}, ResizeMode::PROPORTIONAL, 0.0};
  }
}

inline const char* DetectorProfileName(DetectorProfile profile)
{
  return ProfileSpecFor(profile).name;
}

struct NetworkInputMapping
{
  double x_scale{1.0};
  double y_scale{1.0};
  cv::Point2f input_offset{};

  [[nodiscard]] cv::Point2f MapToSource(float x, float y) const
  {
    return {
        static_cast<float>((static_cast<double>(x) - input_offset.x) * x_scale),
        static_cast<float>((static_cast<double>(y) - input_offset.y) * y_scale)};
  }
};

inline uint32_t to_log_u32(uint64_t value)
{
  return value > std::numeric_limits<uint32_t>::max()
             ? std::numeric_limits<uint32_t>::max()
             : static_cast<uint32_t>(value);
}

inline uint32_t scaled_log_u32(double value, double scale)
{
  if (!std::isfinite(value) || value <= 0.0)
  {
    return 0U;
  }

  const double scaled = value * scale;
  if (scaled >= static_cast<double>(std::numeric_limits<uint32_t>::max()))
  {
    return std::numeric_limits<uint32_t>::max();
  }
  return static_cast<uint32_t>(std::lround(scaled));
}

struct OutputLayout
{
  static constexpr int objectness_index = 8;
  static constexpr int color_begin = 9;
  static constexpr int color_end = 13;
  static constexpr int number_begin = 13;
  static constexpr int number_end = 22;
};

inline int ArgMaxRowRange(const cv::Mat& output, int row, int begin, int end)
{
  int best = begin;
  float best_value = output.at<float>(row, begin);
  for (int index = begin + 1; index < end; ++index)
  {
    const float value = output.at<float>(row, index);
    if (value > best_value)
    {
      best_value = value;
      best = index;
    }
  }
  return best - begin;
}

inline bool FinitePoint(const cv::Point2f& point)
{
  return std::isfinite(point.x) && std::isfinite(point.y);
}

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

inline ArmorColor color_from_yolo_id(int color_id)
{
  if (color_id == 0)
  {
    return ArmorColor::BLUE;
  }
  if (color_id == 1)
  {
    return ArmorColor::RED;
  }
  if (color_id == 2)
  {
    return ArmorColor::EXTINGUISH;
  }
  return ArmorColor::UNKNOWN;
}

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

inline ArmorNumber number_from_yolo_id(int number_id)
{
  switch (number_id)
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

inline ArmorNumber number_from_shtech_target_id(int target_id)
{
  switch (target_id)
  {
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
    case 7:
      return ArmorNumber::GUARD;
    case 8:
      return ArmorNumber::OUTPOST;
    case 9:
      return ArmorNumber::BASE;
    default:
      return ArmorNumber::UNKNOWN;
  }
}

inline int shtech_target_id_from_class_id(int class_id)
{
  if (class_id == 7 || class_id == 8)
  {
    return 9;
  }
  if (class_id == 0)
  {
    return 7;
  }
  if (class_id == 6)
  {
    return 8;
  }
  return class_id;
}

inline std::array<cv::Point2f, 4> sort_keypoints(
    const std::array<cv::Point2f, 4>& keypoints)
{
  std::array<cv::Point2f, 4> sorted = keypoints;

  std::sort(sorted.begin(), sorted.end(),
            [](const cv::Point2f& lhs, const cv::Point2f& rhs)
            {
              return lhs.y < rhs.y;
            });

  std::array<cv::Point2f, 2> top_points = {sorted[0], sorted[1]};
  std::array<cv::Point2f, 2> bottom_points = {sorted[2], sorted[3]};
  std::sort(top_points.begin(), top_points.end(),
            [](const cv::Point2f& lhs, const cv::Point2f& rhs)
            {
              return lhs.x < rhs.x;
            });
  std::sort(bottom_points.begin(), bottom_points.end(),
            [](const cv::Point2f& lhs, const cv::Point2f& rhs)
            {
              return lhs.x < rhs.x;
            });

  return {top_points[0], top_points[1], bottom_points[1], bottom_points[0]};
}

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

inline cv::Point2f quad_center(const std::array<cv::Point2f, 4>& points)
{
  return (points[0] + points[1] + points[2] + points[3]) * 0.25F;
}

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

inline cv::Rect ExpandRect(const cv::Rect& rect, double padding_ratio)
{
  if (padding_ratio <= 0.0)
  {
    return rect;
  }

  const int pad_x =
      std::max(1, static_cast<int>(std::lround(rect.width * padding_ratio)));
  const int pad_y =
      std::max(1, static_cast<int>(std::lround(rect.height * padding_ratio)));
  return {rect.x - pad_x, rect.y - pad_y, rect.width + pad_x * 2,
          rect.height + pad_y * 2};
}

}  // namespace armor_detector_detail

namespace detail = armor_detector_detail;
