#pragma once

#include <array>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "AutoAimTypes.hpp"

/**
 * @brief 装甲板编号分类器（armor-models num-v1）：按四个角点从灰度图矫正出 36×40 小图，
 *        标准化后分 9 类。
 *        Armor number classifier (armor-models num-v1): rectifies a 36×40 patch from
 *        the grey image by the four corners, standardises it and classifies 9 classes.
 *
 * 类别顺序 1、2、3、4、5、前哨站、哨兵、基地、非装甲板，与 ArmorNumber 的取值一一对应。
 * Classes 1, 2, 3, 4, 5, outpost, sentry, base, negative map one-to-one onto
 * ArmorNumber.
 */
class NumberClassifier
{
 public:
  static constexpr int WIDTH = 36;
  static constexpr int HEIGHT = 40;
  static constexpr int CLASSES = 9;

  explicit NumberClassifier(const std::string& onnx_path)
      : net_(cv::dnn::readNetFromONNX(onnx_path))
  {
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }

  /// 原始 BayerRG8 帧转灰度（OpenCV 称 (0,0) 为 R 的排列为 BayerBG）/ Raw BayerRG8 to
  /// grey (OpenCV calls R-at-(0,0) BayerBG).
  static cv::Mat Gray(const uint8_t* bayer, int width, int height)
  {
    cv::Mat gray;
    cv::cvtColor(cv::Mat(height, width, CV_8UC1, const_cast<uint8_t*>(bayer)), gray,
                 cv::COLOR_BayerBG2GRAY);
    return gray;
  }

  /// 标准化后的输入小图，供测试核对 / The standardised input patch, for tests.
  static cv::Mat Patch(const cv::Mat& gray, const std::array<cv::Point2f, 4>& corners)
  {
    // 两根灯条落在第 2、33 列，上下各留约 0.75 个灯条长度 / Bars at columns 2 and 33.
    static const std::array<cv::Point2f, 4> CANON = {
        cv::Point2f{2, 12}, cv::Point2f{2, 28}, cv::Point2f{33, 28}, cv::Point2f{33, 12}};
    const cv::Mat m = cv::getPerspectiveTransform(corners.data(), CANON.data());
    cv::Mat patch;
    cv::warpPerspective(gray, patch, m, {WIDTH, HEIGHT}, cv::INTER_LINEAR,
                        cv::BORDER_REPLICATE);
    patch.convertTo(patch, CV_32F);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(patch, mean, stddev);
    return (patch - mean[0]) / (stddev[0] + 4.0);
  }

  ArmorNumber Classify(const cv::Mat& gray, const std::array<cv::Point2f, 4>& corners)
  {
    const cv::Mat patch = Patch(gray, corners);
    const int shape[] = {1, 1, HEIGHT, WIDTH};
    net_.setInput(cv::Mat(4, shape, CV_32F, const_cast<float*>(patch.ptr<float>())));
    const cv::Mat logits = net_.forward();
    int best = 0;
    for (int c = 1; c < CLASSES; ++c)
    {
      if (logits.at<float>(0, c) > logits.at<float>(0, best))
      {
        best = c;
      }
    }
    return static_cast<ArmorNumber>(best);
  }

 private:
  cv::dnn::Net net_;
};
