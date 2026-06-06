#pragma once

#include "infer/ArmorDetectorModelRegistry.hpp"

namespace armor_detector_infer
{

inline ArmorColor int16_color_from_raw_id(int color_id)
{
  if (color_id == 0)
  {
    return ArmorColor::BLUE;
  }
  if (color_id == 1)
  {
    return ArmorColor::RED;
  }
  return ArmorColor::UNKNOWN;
}

inline ArmorNumber int16_number_from_raw_class(int class_id)
{
  int tag = class_id;
  if (tag == 7 || tag == 8)
  {
    tag = 9;
  }
  else if (tag == 0)
  {
    tag = 7;
  }
  else if (tag == 6)
  {
    tag = 8;
  }

  switch (tag)
  {
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
    case 7:
      return ArmorNumber::GUARD;
    case 8:
      return ArmorNumber::OUTPOST;
    case 9:
      return ArmorNumber::BASE;
    default:
      return ArmorNumber::UNKNOWN;
  }
}

inline const ModelInferAdapter& int16_model_infer_adapter()
{
  static const ModelInferAdapter kAdapter{
      ModelLine::INT16,
      20160,
      22,
      9,
      13,
      13,
      22,
      true,
      false,
      {{0, 3, 2, 1}},
      int16_color_from_raw_id,
      int16_number_from_raw_class,
  };
  return kAdapter;
}

}  // namespace armor_detector_infer
