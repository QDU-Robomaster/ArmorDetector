#pragma once

// 仅供 ArmorDetector.hpp 在类声明之后包含。
namespace armor_detector_detail
{
// 只保留与模块主体无关的低层工具，避免 detector 主逻辑里充满魔法数字和绘图细节。
constexpr double deg2rad = CV_PI / 180.0;
constexpr int yolo_input_size = 640;
constexpr int info_panel_width = 360;
constexpr int max_debug_armors = 6;
constexpr double header_bar_alpha = 0.78;
constexpr float point_radius = 4.0F;
constexpr int preview_header_height = 54;
constexpr uint32_t sync_frame_wait_timeout_ms = 100;
constexpr uint32_t metrics_log_period = 30;
constexpr size_t sync_frame_thread_stack_size = 1024U * 128U;

struct OutputLayout
{
  static constexpr int objectness_index = 8;
  static constexpr int color_begin = 9;
  static constexpr int color_end = 13;
  static constexpr int number_begin = 13;
  static constexpr int number_end = 22;
};

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

inline std::string armor_number_to_string(ArmorNumber number)
{
  const std::size_t index = static_cast<std::size_t>(number);
  if (index >= ARMOR_NUMBER_NAMES.size())
  {
    return "invalid";
  }
  return std::string(ARMOR_NUMBER_NAMES[index]);
}

inline std::string armor_type_to_string(ArmorType type)
{
  const std::size_t index = static_cast<std::size_t>(type);
  if (index >= ARMOR_TYPE_NAMES.size())
  {
    return "invalid";
  }
  return std::string(ARMOR_TYPE_NAMES[index]);
}

inline std::string format_float(double value, int precision = 3)
{
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

inline cv::Scalar color_to_scalar(ArmorColor color)
{
  switch (color)
  {
    case ArmorColor::BLUE:
      return cv::Scalar(255, 180, 40);
    case ArmorColor::RED:
      return cv::Scalar(60, 90, 255);
    case ArmorColor::EXTINGUISH:
      return cv::Scalar(180, 180, 180);
    case ArmorColor::PURPLE:
      return cv::Scalar(220, 70, 220);
    case ArmorColor::UNKNOWN:
    default:
      return cv::Scalar(150, 220, 150);
  }
}

inline void draw_label_chip(cv::Mat& canvas, const std::string& text,
                            const cv::Point& origin, const cv::Scalar& color)
{
  constexpr int FONT = cv::FONT_HERSHEY_DUPLEX;
  constexpr double FONT_SCALE = 0.55;
  constexpr int THICKNESS = 1;
  constexpr int PADDING_X = 8;
  constexpr int PADDING_Y = 6;

  int baseline = 0;
  const cv::Size text_size =
      cv::getTextSize(text, FONT, FONT_SCALE, THICKNESS, &baseline);
  const cv::Rect bg_rect(origin.x, origin.y - text_size.height - PADDING_Y,
                         text_size.width + 2 * PADDING_X,
                         text_size.height + 2 * PADDING_Y);
  cv::rectangle(canvas, bg_rect, color, cv::FILLED, cv::LINE_AA);
  cv::putText(canvas, text,
              cv::Point(origin.x + PADDING_X,
                        origin.y - PADDING_Y + baseline / 2),
              FONT, FONT_SCALE, cv::Scalar(12, 16, 24), THICKNESS, cv::LINE_AA);
}

inline void draw_info_row(cv::Mat& canvas, int x, int y, const std::string& key,
                          const std::string& value,
                          const cv::Scalar& value_color)
{
  constexpr int FONT = cv::FONT_HERSHEY_DUPLEX;
  constexpr double FONT_SCALE = 0.55;
  constexpr int THICKNESS = 1;

  cv::putText(canvas, key, cv::Point(x, y), FONT, FONT_SCALE,
              cv::Scalar(170, 182, 196), THICKNESS, cv::LINE_AA);
  cv::putText(canvas, value, cv::Point(x + 145, y), FONT, FONT_SCALE, value_color,
              THICKNESS, cv::LINE_AA);
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

inline const char* target_color_name(ArmorColor color)
{
  if (color == ArmorColor::BLUE)
  {
    return "blue";
  }
  if (color == ArmorColor::RED)
  {
    return "red";
  }
  return "any";
}

inline std::string armor_display_name(ArmorNumber number, ArmorType type)
{
  return armor_number_to_string(number) + " / " + armor_type_to_string(type);
}
}  // namespace armor_detector_detail

namespace detail = armor_detector_detail;
