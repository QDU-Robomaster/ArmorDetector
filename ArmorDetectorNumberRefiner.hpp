#pragma once

/**
 * @file ArmorDetectorNumberRefiner.hpp
 * @brief 使用 CPU 执行的装甲板数字 refine 分类器。
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "ArmorDetectorTypes.hpp"
#include "logger.hpp"

namespace armor_detector_detail
{

/**
 * @brief 只使用 CPU 的 MLP 数字分类器。
 *
 * 分类器输入为从装甲板四边形中裁出的 `20x28` 二值数字图，裁剪几何与离线检查脚本一致。
 */
class MlpNumberRefiner
{
 public:
  struct Prediction
  {
    bool valid{false};                         ///< 分类器是否成功输出结果。
    ArmorNumber number{ArmorNumber::UNKNOWN}; ///< 分类器预测编号。
    int label_index{8};                        ///< 分类器原始类别下标。
    float confidence{0.0F};                    ///< softmax 置信度。
  };

  /**
   * @brief 加载 ONNX 模型并固定使用 OpenCV DNN CPU 推理。
   * @param model_path MLP 数字 refine 模型路径。
   * @return 模型可用时返回 true。
   */
  bool Configure(const char* model_path)
  {
    ready_ = false;
    net_ = cv::dnn::Net();
    if (model_path == nullptr || model_path[0] == '\0')
    {
      XR_LOG_WARN("ArmorDetector number refine model path is empty");
      return false;
    }

    try
    {
      net_ = cv::dnn::readNetFromONNX(model_path);
      if (net_.empty())
      {
        XR_LOG_WARN("ArmorDetector number refine model is empty: %s", model_path);
        return false;
      }
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
      ready_ = true;
      XR_LOG_PASS("ArmorDetector loaded CPU number refine model: %s", model_path);
      return true;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector failed to load number refine model %s: %s",
                   model_path, exception.what());
      return false;
    }
  }

  /**
   * @brief 关闭分类器并释放模型。
   */
  void Reset()
  {
    ready_ = false;
    net_ = cv::dnn::Net();
  }

  /**
   * @brief 判断分类器是否可用。
   */
  [[nodiscard]] bool Ready() const { return ready_; }

  /**
   * @brief 裁剪数字 ROI 并运行 MLP 分类。
   * @param bgr_img 源 BGR 图像。
   * @param points 装甲板四角点，顺序为左上、右上、右下、左下。
   * @param type 当前装甲板尺寸类型，用于选择透视展开宽度。
   * @return 裁剪或推理失败时返回 valid=false。
   */
  Prediction Predict(const cv::Mat& bgr_img,
                     const std::array<cv::Point2f, 4>& points,
                     ArmorType type)
  {
    Prediction prediction;
    if (!ready_ || bgr_img.empty())
    {
      return prediction;
    }

    cv::Mat binary_roi;
    if (!ExtractDigitRoi(bgr_img, points, type, binary_roi))
    {
      return prediction;
    }

    try
    {
      cv::Mat input;
      binary_roi.convertTo(input, CV_32F, 1.0 / 255.0);
      cv::Mat blob = cv::dnn::blobFromImage(input);
      net_.setInput(blob);
      cv::Mat logits = net_.forward();
      return DecodeSoftmax(logits);
    }
    catch (const std::exception& exception)
    {
      XR_LOG_WARN("ArmorDetector number refine inference failed: %s",
                  exception.what());
      return prediction;
    }
  }

 private:
  static ArmorNumber NumberFromModelIndex(int index)
  {
    switch (index)
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
      case 8:
      default:
        return ArmorNumber::NEGATIVE;
    }
  }

  static Prediction DecodeSoftmax(const cv::Mat& logits)
  {
    Prediction prediction;
    cv::Mat flat = logits.reshape(1, 1);
    if (flat.empty())
    {
      return prediction;
    }
    if (flat.type() != CV_32F)
    {
      flat.convertTo(flat, CV_32F);
    }

    const int count = std::min<int>(9, static_cast<int>(flat.total()));
    if (count <= 0)
    {
      return prediction;
    }

    const float* values = flat.ptr<float>(0);
    float max_value = values[0];
    for (int i = 1; i < count; ++i)
    {
      max_value = std::max(max_value, values[i]);
    }

    std::array<float, 9> exp_values{};
    float sum = 0.0F;
    for (int i = 0; i < count; ++i)
    {
      const float value = std::exp(values[i] - max_value);
      exp_values[static_cast<std::size_t>(i)] = value;
      sum += value;
    }
    if (!std::isfinite(sum) || sum <= 0.0F)
    {
      return prediction;
    }

    int best = 0;
    float best_prob = exp_values[0] / sum;
    for (int i = 1; i < count; ++i)
    {
      const float prob = exp_values[static_cast<std::size_t>(i)] / sum;
      if (prob > best_prob)
      {
        best_prob = prob;
        best = i;
      }
    }

    prediction.valid = true;
    prediction.label_index = best;
    prediction.number = NumberFromModelIndex(best);
    prediction.confidence = best_prob;
    return prediction;
  }

  static bool ExtractDigitRoi(const cv::Mat& bgr_img,
                              const std::array<cv::Point2f, 4>& points,
                              ArmorType type, cv::Mat& binary_roi)
  {
    binary_roi.release();
    for (const auto& point : points)
    {
      if (!std::isfinite(point.x) || !std::isfinite(point.y))
      {
        return false;
      }
    }

    constexpr int roi_width = 20;
    constexpr int canonical_height = 28;
    constexpr int light_length = 12;
    constexpr int top_light_y = (canonical_height - light_length) / 2 - 1;
    constexpr int bottom_light_y = top_light_y + light_length;
    const int canonical_width = type == ArmorType::LARGE ? 54 : 32;

    const std::vector<cv::Point2f> src = {
        points[3], points[0], points[1], points[2]};
    const std::vector<cv::Point2f> dst = {
        cv::Point2f(0.0F, static_cast<float>(bottom_light_y)),
        cv::Point2f(0.0F, static_cast<float>(top_light_y)),
        cv::Point2f(static_cast<float>(canonical_width - 1),
                    static_cast<float>(top_light_y)),
        cv::Point2f(static_cast<float>(canonical_width - 1),
                    static_cast<float>(bottom_light_y))};

    const cv::Mat transform = cv::getPerspectiveTransform(src, dst);
    cv::Mat warped;
    cv::warpPerspective(bgr_img, warped, transform,
                        cv::Size(canonical_width, canonical_height),
                        cv::INTER_LINEAR);
    if (warped.empty())
    {
      return false;
    }

    const int x0 = std::max(0, (canonical_width - roi_width) / 2);
    if (x0 + roi_width > warped.cols || warped.rows != canonical_height)
    {
      return false;
    }

    const cv::Mat digit = warped(cv::Rect(x0, 0, roi_width, canonical_height));
    cv::Mat gray;
    if (digit.channels() == 1)
    {
      gray = digit;
    }
    else
    {
      cv::cvtColor(digit, gray, cv::COLOR_BGR2GRAY);
    }
    cv::threshold(gray, binary_roi, 0.0, 255.0,
                  cv::THRESH_BINARY | cv::THRESH_OTSU);
    return binary_roi.rows == canonical_height && binary_roi.cols == roi_width;
  }

  bool ready_{false};
  cv::dnn::Net net_{};
};

}  // namespace armor_detector_detail
