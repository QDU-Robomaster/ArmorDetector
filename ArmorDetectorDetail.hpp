#pragma once

/**
 * @file ArmorDetectorDetail.hpp
 * @brief ArmorDetector 内部使用的常量、profile 描述和轻量几何/语义工具。
 */

/**
 * @brief ArmorDetector 内部实现命名空间。
 *
 * 这些工具不构成跨模块 ABI；对外请使用 armor.hpp 中的结果结构。
 */
namespace armor_detector_detail
{

/**
 * @brief 角度到弧度的换算系数。
 */
constexpr double deg2rad = CV_PI / 180.0;

/**
 * @brief 正方形 keypoint detector 默认输入边长。
 */
constexpr int yolo_input_size = 640;

/**
 * @brief dense-grid keypoint detector 输入宽度。
 */
constexpr int direct_keypoint_input_width = 640;

/**
 * @brief dense-grid keypoint detector 输入高度。
 */
constexpr int direct_keypoint_input_height = 512;

/**
 * @brief dense-grid keypoint profile 的 NMS bbox 膨胀比例。
 */
constexpr double direct_keypoint_nms_box_padding_ratio = 0.0;

/**
 * @brief dense-grid keypoint profile 的输出候选数量。
 */
constexpr int direct_keypoint_candidate_count = 6720;

/**
 * @brief dense-grid keypoint profile 的输出列数。
 */
constexpr int direct_keypoint_output_width = 21;

/**
 * @brief dense-grid keypoint profile 置信度排序后参与交叠抑制的最大候选数。
 */
constexpr int direct_keypoint_keep_topk = 128;

/**
 * @brief 同步帧 worker 单次等待超时，单位 ms。
 */
constexpr uint32_t sync_frame_wait_timeout_ms = 100;

/**
 * @brief 周期性指标日志输出帧间隔。
 */
constexpr uint32_t metrics_log_period = 30;

/**
 * @brief 同步帧 worker 线程栈大小。
 */
constexpr size_t sync_frame_thread_stack_size = 1024U * 128U;

/**
 * @brief detector 网络输入尺寸。
 */
struct NetworkInputShape
{
  int width{yolo_input_size};  ///< 输入宽度，单位 px。
  int height{yolo_input_size}; ///< 输入高度，单位 px。
};

/**
 * @brief detector 模型和 decoder profile。
 */
enum class DetectorProfile : uint8_t
{
  YOLO_KEYPOINT_640X640 = 0,   ///< 640x640 等比例输入的 YOLO keypoint 输出。
  DIRECT_KEYPOINT_640X512 = 1, ///< 640x512 拉伸输入的 dense-grid keypoint 输出。
};

/**
 * @brief dense-grid keypoint 输出角点转 detector 统一角点的策略。
 */
enum class DirectKeypointPointOrder : uint8_t
{
  CANONICAL_SORT = 0, ///< 按像素几何排序为左上、右上、右下、左下。
  DECLARED_ORDER = 1, ///< 信任模型声明顺序，仅做固定下标重排。
};

/**
 * @brief 传统角点细化策略。
 */
enum class CornerRefineMode : uint8_t
{
  PAIR_ROI = 0,        ///< 当前实现：一个装甲板 ROI 内找灯条对。
  SPLIT_ROI_WEIGHTED = 1, ///< 源参考实现：左右灯条分 ROI 并加权选择。
};

/**
 * @brief 输入图像 resize 策略。
 */
enum class ResizeMode : uint8_t
{
  PROPORTIONAL, ///< 保持比例并填充 letterbox。
  STRETCH,      ///< 直接拉伸到模型输入尺寸。
};

/**
 * @brief detector profile 的静态规格。
 */
struct DetectorProfileSpec
{
  const char* name{"yolo_keypoint_640x640"};          ///< 配置/日志使用的 profile 名称。
  NetworkInputShape input_shape{};                    ///< 网络输入宽高。
  ResizeMode resize_mode{ResizeMode::PROPORTIONAL};   ///< 输入 resize 策略。
  double nms_box_padding_ratio{0.0};                  ///< NMS 前 bbox 膨胀比例。
};

/**
 * @brief 查询 detector profile 对应的静态规格。
 * @param profile detector profile。
 * @return profile 规格；未知值返回安全兜底规格。
 */
inline DetectorProfileSpec ProfileSpecFor(DetectorProfile profile)
{
  switch (profile)
  {
    case DetectorProfile::YOLO_KEYPOINT_640X640:
      return {};
    case DetectorProfile::DIRECT_KEYPOINT_640X512:
      return {"direct_keypoint_640x512",
              {direct_keypoint_input_width, direct_keypoint_input_height},
              ResizeMode::STRETCH,
              direct_keypoint_nms_box_padding_ratio};
    default:
      return {"unknown", {}, ResizeMode::PROPORTIONAL, 0.0};
  }
}

/**
 * @brief 查询 detector profile 的可读名称。
 * @param profile detector profile。
 * @return profile 名称字符串。
 */
inline const char* DetectorProfileName(DetectorProfile profile)
{
  return ProfileSpecFor(profile).name;
}

/**
 * @brief 查询 dense-grid 角点顺序策略名称。
 * @param order 角点顺序策略。
 * @return 稳定日志名称。
 */
inline const char* DirectKeypointPointOrderName(DirectKeypointPointOrder order)
{
  switch (order)
  {
    case DirectKeypointPointOrder::CANONICAL_SORT:
      return "canonical_sort";
    case DirectKeypointPointOrder::DECLARED_ORDER:
      return "declared_order";
    default:
      return "unknown";
  }
}

/**
 * @brief 查询传统角点细化策略名称。
 * @param mode 细化策略。
 * @return 稳定日志名称。
 */
inline const char* CornerRefineModeName(CornerRefineMode mode)
{
  switch (mode)
  {
    case CornerRefineMode::PAIR_ROI:
      return "pair_roi";
    case CornerRefineMode::SPLIT_ROI_WEIGHTED:
      return "split_roi_weighted";
    default:
      return "unknown";
  }
}

/**
 * @brief 网络输入坐标到原始 detector 图像坐标的映射。
 */
struct NetworkInputMapping
{
  double x_scale{1.0};         ///< 网络 x 坐标还原到源图像的比例。
  double y_scale{1.0};         ///< 网络 y 坐标还原到源图像的比例。
  cv::Point2f input_offset{};  ///< letterbox 填充导致的输入坐标偏移。

  /**
   * @brief 将模型输入平面上的点还原到 detector 图像平面。
   * @param x 模型输入坐标 x。
   * @param y 模型输入坐标 y。
   * @return 源图像像素坐标。
   */
  [[nodiscard]] cv::Point2f MapToSource(float x, float y) const
  {
    return {
        static_cast<float>((static_cast<double>(x) - input_offset.x) * x_scale),
        static_cast<float>((static_cast<double>(y) - input_offset.y) * y_scale)};
  }
};

/**
 * @brief 将 64-bit 计数安全压缩成日志可打印的 32-bit 值。
 * @param value 原始值。
 * @return 截断饱和后的 uint32_t。
 */
inline uint32_t to_log_u32(uint64_t value)
{
  return value > std::numeric_limits<uint32_t>::max()
             ? std::numeric_limits<uint32_t>::max()
             : static_cast<uint32_t>(value);
}

/**
 * @brief 缩放浮点指标并转换成日志用 uint32_t。
 * @param value 原始浮点值。
 * @param scale 缩放倍数。
 * @return 非法/非正值返回 0，溢出时饱和。
 */
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

/**
 * @brief 网络输出行的字段布局。
 */
struct OutputLayout
{
  static constexpr int objectness_index = 8; ///< 目标置信度 logit 列。
  static constexpr int color_begin = 9;      ///< 颜色分类起始列，闭区间。
  static constexpr int color_end = 13;       ///< 颜色分类结束列，开区间。
  static constexpr int number_begin = 13;    ///< 编号分类起始列，闭区间。
  static constexpr int number_end = 22;      ///< 编号分类结束列，开区间。
};

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
 * @brief 查询 profile 对应的输出列数。
 * @param profile detector profile。
 * @return 输出列数；未知 profile 返回 YOLO keypoint 列数作为兜底。
 */
inline int OutputColumnCountFor(DetectorProfile profile)
{
  if (profile == DetectorProfile::DIRECT_KEYPOINT_640X512)
  {
    return direct_keypoint_output_width;
  }
  return OutputLayout::number_end;
}

/**
 * @brief dense-grid 输出行对应的网格单元。
 */
struct DirectKeypointGridCell
{
  int center_x{0}; ///< 网格中心 x，单位为模型输入像素。
  int center_y{0}; ///< 网格中心 y，单位为模型输入像素。
  int stride{8};   ///< 当前检测层 stride。
};

/**
 * @brief 将 dense-grid 输出行号转换为网格中心和 stride。
 * @param row 输出行号，范围为 [0, direct_keypoint_candidate_count)。
 * @return 对应网格单元；越界时返回最后一层的兜底单元。
 */
inline DirectKeypointGridCell DirectKeypointGridCellForRow(int row)
{
  if (row < 0)
  {
    return {};
  }

  constexpr int stride8_cols = direct_keypoint_input_width / 8;
  constexpr int stride8_count =
      stride8_cols * (direct_keypoint_input_height / 8);
  constexpr int stride16_cols = direct_keypoint_input_width / 16;
  constexpr int stride16_count =
      stride16_cols * (direct_keypoint_input_height / 16);

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
  constexpr int stride32_cols = direct_keypoint_input_width / 32;
  return {(row % stride32_cols) * 32, (row / stride32_cols) * 32, 32};
}

/**
 * @brief 在输出矩阵某一行的指定列范围内取 argmax。
 * @param output 输出矩阵。
 * @param row 行号。
 * @param begin 起始列，闭区间。
 * @param end 结束列，开区间。
 * @return 最大值列号减去 begin 后的类别 id。
 */
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
 * @brief 将 YOLO keypoint profile 的颜色类别 id 转为 ArmorColor。
 * @param color_id 模型颜色类别 id。
 * @return detector 统一颜色枚举。
 */
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

/**
 * @brief 将 dense-grid keypoint profile 的颜色类别 id 转为 ArmorColor。
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
 * @brief 将 YOLO keypoint profile 的编号类别 id 转为 ArmorNumber。
 * @param number_id 模型编号类别 id。
 * @return detector 统一编号枚举。
 */
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

/**
 * @brief 将 dense-grid keypoint profile 的 8 类编号 id 转为 ArmorNumber。
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
 * @brief 将任意四个角点排序为 detector/PnP 统一顺序。
 * @param keypoints 输入四点。
 * @return 左上、右上、右下、左下顺序的四点。
 */
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

/**
 * @brief 将 dense-grid 声明顺序固定映射到 detector/PnP 统一顺序。
 *
 * dense-grid 2025 模型在交换第 2/3 个点后声明顺序为左上、左下、右下、
 * 右上；detector 统一使用左上、右上、右下、左下。
 *
 * @param keypoints 已按源实现交换第 2/3 点后的四点。
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
 * @brief 按宽高比例向外扩展包围盒。
 * @param rect 原始包围盒。
 * @param padding_ratio 宽高方向各自的扩展比例。
 * @return 扩展后的包围盒；比例非正时返回原包围盒。
 */
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
