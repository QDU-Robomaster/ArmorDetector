/**
 * @file number_classifier.cpp
 * @brief 数字装甲板分类器（实现）。
 */

// OpenCV
#include <opencv2/opencv.hpp>

// STL
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "NumberClassifier.hpp"
#include "logger.hpp"

namespace
{
// ================== 透视展开与 ROI 固定参数 ==================

/**
 * @brief 展开图中“灯条高度”的像素数
 */
constexpr int K_LIGHT_LENGTH = 12;

/**
 * @brief 透视展开后的图像高度
 */
constexpr int K_WARP_HEIGHT = 28;

/**
 * @brief 小装甲的展开宽度
 */
constexpr int K_SMALL_WIDTH = 32;

/**
 * @brief 大装甲的展开宽度
 */
constexpr int K_LARGE_WIDTH = 54;

/**
 * @brief 数字 ROI（在展开图中的）宽与高
 */
constexpr int K_ROI_W = 20;
constexpr int K_ROI_H = 28;
}  // namespace

// ================== 工具函数实现 ==================

int NumberClassifier::WarpWidthFor(ArmorType type)
{
  return (type == ArmorType::SMALL) ? K_SMALL_WIDTH : K_LARGE_WIDTH;
}

void NumberClassifier::FillSrcDstQuads(const Armor& armor, cv::Point2f dst[4],
                                       cv::Point2f src[4])
{
  // 目标四边形（展开图）：
  // 将灯条竖直高度放在展开图的中间，便于从中部截数字。
  const int top_y = (K_WARP_HEIGHT - K_LIGHT_LENGTH) / 2 - 1;
  const int bottom_y = top_y + K_LIGHT_LENGTH;
  const int warp_w = WarpWidthFor(armor.type);

  dst[0] = cv::Point2f(0.f, static_cast<float>(bottom_y));  // 左下
  dst[1] = cv::Point2f(0.f, static_cast<float>(top_y));     // 左上
  dst[2] = cv::Point2f(static_cast<float>(warp_w - 1),
                       static_cast<float>(top_y));  // 右上
  dst[3] = cv::Point2f(static_cast<float>(warp_w - 1),
                       static_cast<float>(bottom_y));  // 右下

  // 源四边形（像素坐标）：
  // 按 左下/左上/右上/右下，与目标顶点一一对应。
  src[0] = armor.left_light.bottom;
  src[1] = armor.left_light.top;
  src[2] = armor.right_light.top;
  src[3] = armor.right_light.bottom;
}

cv::Rect NumberClassifier::SafeRoi(const cv::Size& img_size, const cv::Rect& roi)
{
  cv::Rect safe = roi;
  safe.x = std::max(0, std::min(safe.x, img_size.width - 1));
  safe.y = std::max(0, std::min(safe.y, img_size.height - 1));
  safe.width = std::max(0, std::min(roi.width, img_size.width - safe.x));
  safe.height = std::max(0, std::min(roi.height, img_size.height - safe.y));
  return safe;
}

cv::Mat NumberClassifier::Softmax(const cv::Mat& logits)
{
  // 支持 (1,C) 或 (1,C,1,1)，展平到 1xC 进行 softmax。
  CV_Assert(!logits.empty());
  cv::Mat flat = logits.reshape(1, 1);  // 1 x (C or C*1*1)
  const float max_logit = *std::max_element(flat.begin<float>(), flat.end<float>());
  cv::Mat exps;
  cv::exp(flat - max_logit, exps);
  const float sum = static_cast<float>(cv::sum(exps)[0]);
  const float safe = std::max(sum, 1e-12f);
  return exps / safe;  // 1 x C
}

// ================== 类实现 ==================

NumberClassifier::NumberClassifier(
    const std::string& model_path, double threshold,
    const std::initializer_list<ArmorNumber>& ignore_classes)
    : threshold_(threshold)
{
  // 读取 ONNX 模型
  net_ = cv::dnn::readNetFromONNX(model_path);
  if (net_.empty())
  {
    XR_LOG_WARN("NumberClassifier: failed to load ONNX: %s", model_path.c_str());
  }

  // 构建忽略集合
  for (const auto& c : ignore_classes)
  {
    ignore_classes_.insert(c);
  }
}

void NumberClassifier::ExtractNumbers(const cv::Mat& src, std::vector<Armor>& armors)
{
  if (src.empty())
  {
    return;
  }

  for (auto& armor : armors)
  {
    // 1) 透视展开
    cv::Point2f dst_vertices[4];
    cv::Point2f src_vertices[4];
    FillSrcDstQuads(armor, dst_vertices, src_vertices);

    const int warp_w = WarpWidthFor(armor.type);
    cv::Mat number_image;
    const cv::Mat M = cv::getPerspectiveTransform(src_vertices, dst_vertices);
    cv::warpPerspective(src, number_image, M,
                        cv::Size(warp_w, K_WARP_HEIGHT),  // <-- NOTE: typo fix below
                        cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    // （修正：上行拼写应为 K_WARP_HEIGHT）
    if (number_image.rows != K_WARP_HEIGHT)
    {
      // 某些平台上，意外的 size 也进行一次强制调整
      cv::resize(number_image, number_image, cv::Size(warp_w, K_WARP_HEIGHT), 0, 0,
                 cv::INTER_LINEAR);
    }

    // 2) 截取中间数字 ROI（安全裁剪）
    const int x = (warp_w - K_ROI_W) / 2;
    const int y = 0;
    const cv::Rect want_roi{x, y, K_ROI_W, K_ROI_H};
    const cv::Rect roi = SafeRoi(number_image.size(), want_roi);
    if (roi.width != K_ROI_W || roi.height != K_ROI_H)
    {
      XR_LOG_DEBUG("Number ROI clipped: want=(%d,%d,%d,%d), got=(%d,%d,%d,%d)",
                   want_roi.x, want_roi.y, want_roi.width, want_roi.height, roi.x, roi.y,
                   roi.width, roi.height);
    }

    cv::Mat digit = number_image(roi).clone();

    // 3) 灰度 + Otsu 二值化
    if (digit.channels() == 3)
    {
      cv::cvtColor(digit, digit, cv::COLOR_BGR2GRAY);
    }
    else if (digit.channels() == 4)
    {
      cv::cvtColor(digit, digit, cv::COLOR_BGRA2GRAY);
    }
    cv::threshold(digit, digit, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    armor.number_img = std::move(digit);  // 单通道 8U
  }
}

void NumberClassifier::Classify(std::vector<Armor>& armors)
{
  if (net_.empty())
  {
    // 模型未加载：全部过滤
    armors.clear();
    return;
  }

  for (auto& armor : armors)
  {
    cv::Mat image = armor.number_img;  // 期望单通道 8U
    if (image.empty())
    {
      armor.confidence = 0.f;
      armor.number = ArmorNumber::INVALID;
      continue;
    }

#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
    cv::imshow("number", image);
    cv::waitKey(1);
#endif

    // 归一化到 [0,1]，保持单通道
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    // 构造输入 blob: (1,1,H,W)
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob,
                           /*scalefactor=*/1.0,
                           /*size=*/cv::Size(),
                           /*mean=*/cv::Scalar(),
                           /*swapRB=*/false,
                           /*crop=*/false);

    // 前向推理
    net_.setInput(blob);
    cv::Mat logits = net_.forward();  // 常见为 (1,C) 或 (1,C,1,1)

    // Softmax（数值稳定）
    cv::Mat prob = Softmax(logits);  // 1 x C

    // 取最大概率与类别
    double confidence = 0.0;
    cv::Point class_id;
    cv::minMaxLoc(prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id);
    const int label_id = class_id.x;

    armor.confidence = static_cast<float>(confidence);
    armor.number = static_cast<ArmorNumber>(label_id);

    XR_LOG_INFO("Number: %d, confidence: %.3f", label_id, armor.confidence);
  }

  // 末端过滤：阈值、忽略类别、大小-数字先验
  armors.erase(
      std::remove_if(
          armors.begin(), armors.end(),
          [this](const Armor& a)
          {
            // 1) 置信度过滤
            if (a.confidence < static_cast<float>(threshold_))
            {
              return true;
            }

            // 2) 忽略指定类别
            if (ignore_classes_.find(a.number) != ignore_classes_.end())
            {
              return true;
            }

            // 3) 大/小装甲与数字先验（按项目规则可调整）
            bool mismatch = false;
            if (a.type == ArmorType::LARGE)
            {
              mismatch = (a.number == ArmorNumber::OUTPOST ||
                          a.number == ArmorNumber::TWO || a.number == ArmorNumber::GUARD);
            }
            else if (a.type == ArmorType::SMALL)
            {
              mismatch = (a.number == ArmorNumber::ONE || a.number == ArmorNumber::BASE);
            }
            return mismatch;
          }),
      armors.end());
}
