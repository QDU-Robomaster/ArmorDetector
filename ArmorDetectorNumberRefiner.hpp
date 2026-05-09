#pragma once

/**
 * @file ArmorDetectorNumberRefiner.hpp
 * @brief Fater-style armor number classifier used as a CPU post-refine stage.
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
 * @brief CPU-only Fater-style MLP number classifier wrapper.
 *
 * The classifier consumes a 20x28 binary digit ROI extracted from the armor
 * quadrilateral using the same light-bar geometry used in the offline
 * cross-check scripts.
 */
class FaterMlpNumberRefiner
{
 public:
  struct Prediction
  {
    bool valid{false};                         ///< Classifier ran successfully.
    ArmorNumber number{ArmorNumber::UNKNOWN}; ///< Predicted ArmorNumber enum.
    int label_index{8};                        ///< Raw Fater label index.
    float confidence{0.0F};                    ///< Softmax confidence.
  };

  /**
   * @brief Load the ONNX model and force OpenCV DNN CPU execution.
   * @param model_path Fater-style mlp.onnx path.
   * @return true when the model is ready.
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
   * @brief Disable the classifier and release model memory.
   */
  void Reset()
  {
    ready_ = false;
    net_ = cv::dnn::Net();
  }

  /**
   * @brief Returns whether the classifier is available.
   */
  [[nodiscard]] bool Ready() const { return ready_; }

  /**
   * @brief Extract the digit ROI and run Fater MLP classification.
   * @param bgr_img Source BGR image.
   * @param points Armor corners in top-left, top-right, bottom-right, bottom-left order.
   * @param type Current armor size type used to select canonical warp width.
   * @return Prediction with valid=false on extraction or inference failure.
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
  static ArmorNumber NumberFromFaterIndex(int index)
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
    prediction.number = NumberFromFaterIndex(best);
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
