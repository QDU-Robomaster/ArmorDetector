#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

/**
 * @brief v4 输出格式（armor-models det-v4、det-v7）的解码：两个尺度合并、sigmoid 门限、
 *        角点不加权、四边形外接框 NMS。与 rm_model.decode（fuse=None）逐项一致。颜色类数
 *        取自张量：v4 两类，v7 四类。
 *        Decoding of the v4 output format (armor-models det-v4, det-v7): both scales
 *        merged, sigmoid threshold, unweighted corners, NMS on the quad bounding boxes.
 *        Matches rm_model.decode (fuse=None) step by step. The number of colour classes
 *        comes from the tensor: two for v4, four for v7.
 */
namespace ArmorV4
{
inline constexpr std::array<int, 2> STRIDES = {8, 16};
/// 角点偏移 1.0 等于 4 个格子 / A corner offset of 1.0 spans 4 cells.
inline constexpr float CORNER_UNIT = 4.0F;
inline constexpr int MAX_DETECTIONS = 50;

/**
 * @brief 一个输出张量的只读视图：浮点或量化 uint8，CHW 或 HWC。
 *        Read-only view of one output tensor: float or quantised uint8, CHW or HWC.
 */
struct TensorView
{
  enum class Type : uint8_t
  {
    F32,
    U8,
  };

  const void* data = nullptr;
  Type type = Type::F32;
  bool hwc = false;
  int height = 0;
  int width = 0;
  int channels = 0;
  float scale = 1.0F;       ///< 仅 U8：(q − zero_point) × scale / U8 only
  float zero_point = 0.0F;  ///< 仅 U8 / U8 only

  float At(int c, int y, int x) const
  {
    const std::size_t i = hwc ? (static_cast<std::size_t>(y) * width + x) * channels + c
                              : (static_cast<std::size_t>(c) * height + y) * width + x;
    return type == Type::F32
               ? static_cast<const float*>(data)[i]
               : (static_cast<float>(static_cast<const uint8_t*>(data)[i]) - zero_point) *
                     scale;
  }
};

/// 一个尺度的四个输出 / The four outputs of one scale.
struct ScaleOutputs
{
  TensorView obj;     ///< [1] logit
  TensorView color;   ///< [2 或 4] logit，类别见 Detection::color / see Detection::color
  TensorView size;    ///< [2] logit：0 小、1 大 / 0 small, 1 large
  TensorView corner;  ///< [8] LT、LB、RB、RT 的 (x, y) 偏移 / (x, y) offsets
};

/// stride 8、16 两个尺度 / Scales of stride 8 and 16.
using Outputs = std::array<ScaleOutputs, 2>;

/// 帧坐标（640×512）下的一个检测 / One detection in frame coordinates (640×512).
struct Detection
{
  std::array<cv::Point2f, 4> corners;  ///< LT、LB、RB、RT
  float score;
  int color;  ///< 0 蓝、1 红、2 紫、3 灭灯 / 0 blue, 1 red, 2 purple, 3 off
  int size;   ///< 0 小、1 大 / 0 small, 1 large
};

/// 各通道中最大者的下标，相等取前者 / Index of the largest channel; ties go to the first.
inline int ArgMax(const TensorView& t, int y, int x)
{
  int best = 0;
  for (int c = 1; c < t.channels; ++c)
  {
    if (t.At(c, y, x) > t.At(best, y, x))
    {
      best = c;
    }
  }
  return best;
}

/**
 * @brief 中心包含去重：按分数从高到低，与已保留检测互相包含对方中心的检测视为同一块板的
 *        第二个框，丢弃。补框 IoU 漏掉的大小、角度不同的重复框（金标准上重复框每百帧
 *        2.5 → 0.2，召回约 −1 个百分点）。
 *        Centre-containment suppression: in score order, a detection whose centre lies
 *        in a kept quad, or whose quad contains a kept centre, is a second box on the
 *        same plate and is dropped. It catches duplicates of another size or angle that
 *        box IoU misses.
 */
inline bool QuadContains(const std::array<cv::Point2f, 4>& quad, cv::Point2f p)
{
  return cv::pointPolygonTest(std::vector<cv::Point2f>(quad.begin(), quad.end()), p,
                              false) >= 0;
}

inline cv::Point2f Centre(const Detection& d)
{
  return (d.corners[0] + d.corners[1] + d.corners[2] + d.corners[3]) * 0.25F;
}

inline std::vector<Detection> SuppressContained(const std::vector<Detection>& sorted)
{
  std::vector<Detection> kept;
  for (const Detection& d : sorted)
  {
    bool duplicate = false;
    for (const Detection& k : kept)
    {
      duplicate = duplicate || QuadContains(k.corners, Centre(d)) ||
                  QuadContains(d.corners, Centre(k));
    }
    if (!duplicate)
    {
      kept.push_back(d);
    }
  }
  return kept;
}

inline std::vector<Detection> Decode(const Outputs& outputs, float conf, float iou)
{
  std::vector<Detection> candidates;
  for (std::size_t s = 0; s < outputs.size(); ++s)
  {
    const ScaleOutputs& o = outputs[s];
    const float stride = static_cast<float>(STRIDES[s]);
    for (int i = 0; i < o.obj.height; ++i)
    {
      for (int j = 0; j < o.obj.width; ++j)
      {
        const float score = 1.0F / (1.0F + std::exp(-o.obj.At(0, i, j)));
        if (!(score > conf))
        {
          continue;
        }
        // 格心按像素中心约定 / Cell centre in the pixel-centre convention.
        const float cx = (static_cast<float>(j) + 0.5F) * stride - 0.5F;
        const float cy = (static_cast<float>(i) + 0.5F) * stride - 0.5F;
        Detection d{{}, score, ArgMax(o.color, i, j), ArgMax(o.size, i, j)};
        for (int k = 0; k < 4; ++k)
        {
          d.corners[k] = {o.corner.At(2 * k, i, j) * CORNER_UNIT * stride + cx,
                          o.corner.At(2 * k + 1, i, j) * CORNER_UNIT * stride + cy};
        }
        candidates.push_back(d);
      }
    }
  }
  if (candidates.empty())
  {
    return {};
  }

  std::vector<cv::Rect2d> boxes;
  std::vector<float> scores;
  for (const Detection& d : candidates)
  {
    float x0 = d.corners[0].x, x1 = x0, y0 = d.corners[0].y, y1 = y0;
    for (const cv::Point2f& p : d.corners)
    {
      x0 = std::min(x0, p.x);
      x1 = std::max(x1, p.x);
      y0 = std::min(y0, p.y);
      y1 = std::max(y1, p.y);
    }
    boxes.emplace_back(x0, y0, x1 - x0, y1 - y0);
    scores.push_back(d.score);
  }
  std::vector<int> keep;
  cv::dnn::NMSBoxes(boxes, scores, conf, iou, keep);
  std::vector<Detection> nms;
  for (std::size_t n = 0; n < keep.size() && n < MAX_DETECTIONS; ++n)
  {
    nms.push_back(candidates[keep[n]]);
  }
  return SuppressContained(nms);
}
}  // namespace ArmorV4
