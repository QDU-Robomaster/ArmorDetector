#pragma once

#include <unordered_set>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "armor.hpp"

class NumberClassifier
{
 public:
  struct Config
  {
    double threshold{0.8};
    std::vector<ArmorNumber> ignore_classes{ArmorNumber::NEGATIVE};
  };

  NumberClassifier() = default;
  NumberClassifier(const char* model_path, Config config);

  void Configure(const char* model_path, Config config);
  bool Classify(const cv::Mat& pattern, ArmorNumber& number, float& confidence) const;

 private:
  static cv::Mat Softmax(const cv::Mat& logits);

  mutable cv::dnn::Net net_{};
  Config config_{};
  std::unordered_set<int> ignored_numbers_{};
};
