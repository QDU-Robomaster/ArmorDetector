#pragma once
/**
 * @file number_classifier.hpp
 * @brief 数字装甲板分类器（头文件）。提供从装甲 ROI 提取数字图像并使用 ONNX
 *        模型进行分类的接口。
 *
 * 典型流程：
 *  1. 调用 ExtractNumbers()：根据灯条对的四点进行透视展开，并裁剪出中部数字 ROI。
 *  2. 调用 Classify()：用 DNN 模型进行前向推理，写回 Armor::number 与 confidence，
 *     并按阈值/忽略类别/大小-数字先验进行过滤。
 *
 * @endcode
 */

// OpenCV
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

// STL
#include <initializer_list>
#include <string>
#include <unordered_set>
#include <vector>

#include "armor.hpp"  // 定义 Armor、ArmorType、ArmorNumber

/**
 * @class NumberClassifier
 * @brief 数字分类器：对装甲板中的数字进行识别与筛选。
 */
class NumberClassifier
{
 public:
  /**
   * @brief 构造函数
   * @param model_path     ONNX 模型文件路径
   * @param threshold      置信度阈值（低于此阈值的结果会被丢弃）
   * @param ignore_classes 可选；需要忽略的类别列表（如：哨兵、基地等）
   *
   * @note ignore_classes 会被拷贝到内部的 unordered_set 存储。
   */
  NumberClassifier(const std::string& model_path, double threshold,
                   const std::initializer_list<ArmorNumber>& ignore_classes = {});

  /**
   * @brief 为每个装甲从原图中提取数字 ROI（透视展开 + 二值化）
   * @param src    输入图（BGR 或 BGRA）
   * @param armors 装甲集合，函数会为每个 Armor 写入 number_img 成员（单通道 8U）
   *
   * @details
   *  - 以灯条四个顶点（左下、左上、右上、右下）构建源四边形；
   *  - 将灯条竖直高度放在展开图中部，便于从中部截取数字区域；
   *  - 数字 ROI 大小固定，超出边界时会进行安全裁剪。
   */
  void ExtractNumbers(const cv::Mat& src, std::vector<Armor>& armors);

  /**
   * @brief 对已提取的数字图进行分类，并基于阈值/忽略类别/大小匹配进行过滤
   * @param armors 输入/输出的装甲集合；函数会写回 Armor::number 与
   *               Armor::confidence，并移除不满足条件的装甲
   *
   * @note 模型输出会做数值稳定的 Softmax；支持 (1,C) 或 (1,C,1,1)。
   */
  void Classify(std::vector<Armor>& armors);

  /**
   * @brief 设置置信度阈值
   * @param threshold 新阈值
   */
  void SetThreshold(double threshold) { threshold_ = threshold; }

 private:
  // ==== 工具函数 ====

  /**
   * @brief 数值稳定 Softmax
   * @param logits 形状 (1,C) 或 (1,C,1,1)
   * @return 同形状 Softmax 概率
   */
  static cv::Mat Softmax(const cv::Mat& logits);

  /**
   * @brief 安全 ROI（确保不越界）
   * @param img_size 图像尺寸
   * @param roi      期望 ROI
   * @return 裁剪到图像边界后的 ROI
   */
  static cv::Rect SafeRoi(const cv::Size& img_size, const cv::Rect& roi);

  /**
   * @brief 根据装甲大小返回展开宽度
   */
  static int WarpWidthFor(ArmorType type);

  /**
   * @brief 构建透视变换的源/目标顶点
   * @param armor 输入装甲（需包含左右灯条的 top/bottom）
   * @param dst   输出：目标四边形（展开图），顺序为 左下/左上/右上/右下
   * @param src   输出：源四边形（像素坐标），顺序为 左下/左上/右上/右下
   */
  static void FillSrcDstQuads(const Armor& armor, cv::Point2f dst[4], cv::Point2f src[4]);

 private:
  cv::dnn::Net net_;                                ///< DNN 模型
  std::unordered_set<ArmorNumber> ignore_classes_;  ///< 忽略类别集合
  double threshold_ = 0.0;                          ///< 置信度阈值
};
