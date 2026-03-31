#include "NumberClassifier.hpp"

#include <algorithm>

#include "logger.hpp"

namespace
{
constexpr int INPUT_IMAGE_SIZE = 32;

ArmorNumber label_to_number(int label_id)
{
  switch (label_id)
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
      return ArmorNumber::GUARD;
    case 6:
      return ArmorNumber::OUTPOST;
    case 7:
      return ArmorNumber::BASE;
    default:
      return ArmorNumber::NEGATIVE;
  }
}
}  // namespace

NumberClassifier::NumberClassifier(const char* model_path, Config config)
{
  Configure(model_path, std::move(config));
}

void NumberClassifier::Configure(const char* model_path, Config config)
{
  config_ = std::move(config);
  ignored_numbers_.clear();
  for (const auto number : config_.ignore_classes)
  {
    ignored_numbers_.insert(static_cast<int>(number));
  }

  net_ = cv::dnn::readNetFromONNX(model_path);
  if (net_.empty())
  {
    XR_LOG_ERROR("Failed to load number classifier model: %s", model_path);
  }
}

bool NumberClassifier::Classify(const cv::Mat& pattern, ArmorNumber& number,
                                float& confidence) const
{
  number = ArmorNumber::NEGATIVE;
  confidence = 0.0F;
  if (net_.empty() || pattern.empty())
  {
    return false;
  }

  cv::Mat gray;
  if (pattern.channels() == 3)
  {
    cv::cvtColor(pattern, gray, cv::COLOR_BGR2GRAY);
  }
  else if (pattern.channels() == 4)
  {
    cv::cvtColor(pattern, gray, cv::COLOR_BGRA2GRAY);
  }
  else
  {
    gray = pattern;
  }

  cv::Mat input(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, CV_8UC1, cv::Scalar(0));
  const double x_scale = static_cast<double>(INPUT_IMAGE_SIZE) / std::max(1, gray.cols);
  const double y_scale = static_cast<double>(INPUT_IMAGE_SIZE) / std::max(1, gray.rows);
  const double scale = std::min(x_scale, y_scale);
  const int resized_height = std::max(1, static_cast<int>(gray.rows * scale));
  const int resized_width = std::max(1, static_cast<int>(gray.cols * scale));
  cv::resize(gray, input(cv::Rect(0, 0, resized_width, resized_height)),
             cv::Size(resized_width, resized_height));

  cv::Mat blob =
      cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar(), false, false);
  net_.setInput(blob);
  const cv::Mat logits = net_.forward();
  const cv::Mat probability = Softmax(logits);

  double max_confidence = 0.0;
  cv::Point label_point;
  cv::minMaxLoc(probability.reshape(1, 1), nullptr, &max_confidence, nullptr, &label_point);

  confidence = static_cast<float>(max_confidence);
  number = label_to_number(label_point.x);
  if (confidence < static_cast<float>(config_.threshold) ||
      ignored_numbers_.count(static_cast<int>(number)) > 0)
  {
    number = ArmorNumber::NEGATIVE;
    return false;
  }

  return true;
}

cv::Mat NumberClassifier::Softmax(const cv::Mat& logits)
{
  cv::Mat flat = logits.reshape(1, 1);
  const float max_logit = *std::max_element(flat.begin<float>(), flat.end<float>());
  cv::Mat exps;
  cv::exp(flat - max_logit, exps);
  const float denominator = std::max(static_cast<float>(cv::sum(exps)[0]), 1e-12F);
  return exps / denominator;
}
