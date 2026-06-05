#pragma once

#include <array>
#include <cmath>

#include <opencv2/core.hpp>

#include "infer/ArmorDetectorInt16Model.hpp"
#include "infer/ArmorDetectorInt8Model.hpp"

namespace armor_detector_infer
{

inline const ModelInferAdapter& resolve_model_infer_adapter(ArmorDetectorModel model)
{
  const auto& resolved = resolve_detector_model_or_default(model);
  return resolved.line == ModelLine::INT8 ? int8_model_infer_adapter()
                                          : int16_model_infer_adapter();
}

inline const ModelInferAdapter& resolve_model_infer_adapter(ModelLine line)
{
  return line == ModelLine::INT8 ? int8_model_infer_adapter()
                                 : int16_model_infer_adapter();
}

inline bool reject_raw_color(const ModelInferAdapter& adapter, int raw_color_id)
{
  return adapter.reject_aux_colors && (raw_color_id == 2 || raw_color_id == 3);
}

inline float decode_confidence(const ModelInferAdapter& adapter,
                               float objectness_logit)
{
  if (adapter.confidence_is_logit)
  {
    return objectness_logit;
  }
  return 1.0F / (1.0F + std::exp(-objectness_logit));
}

template <typename PointT>
inline std::array<PointT, 4> canonicalize_points(
    const ModelInferAdapter& adapter,
    const std::array<PointT, 4>& declared_points)
{
  return {
      declared_points[static_cast<std::size_t>(adapter.canonical_point_order[0])],
      declared_points[static_cast<std::size_t>(adapter.canonical_point_order[1])],
      declared_points[static_cast<std::size_t>(adapter.canonical_point_order[2])],
      declared_points[static_cast<std::size_t>(adapter.canonical_point_order[3])],
  };
}

}  // namespace armor_detector_infer
