#pragma once

#include "infer/ArmorDetectorModelRegistry.hpp"

namespace armor_detector_infer
{

inline ArmorColor int8_color_from_raw_id(int color_id)
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

inline ArmorNumber int8_number_from_raw_class(int class_id)
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

inline const ModelInferAdapter& int8_model_infer_adapter()
{
  static const ModelInferAdapter kAdapter{
      ModelLine::INT8,
      6720,
      21,
      17,
      19,
      9,
      17,
      false,
      true,
      {{0, 1, 3, 2}},
      int8_color_from_raw_id,
      int8_number_from_raw_class,
  };
  return kAdapter;
}

}  // namespace armor_detector_infer
