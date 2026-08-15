#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "ArmorDetectorPublishGeometry.hpp"

namespace
{

using armor_detector_detail::MapPublishGeometry;
using armor_detector_detail::PublishGeometry;

const std::array<cv::Point2f, 4> frame_points = {
    cv::Point2f{100.0F, 50.0F}, cv::Point2f{150.0F, 60.0F}, cv::Point2f{145.0F, 110.0F},
    cv::Point2f{95.0F, 105.0F}};
const cv::Point2f frame_center{122.5F, 81.25F};
const cv::Rect frame_box{90, 40, 70, 80};

CameraTypes::FrameGeometry MakeWideGeometry(uint16_t flags)
{
  return {
      720, 540, 2160, 0, 0, 2, 2, flags, 0, 0.0F, 0.0F,
  };
}

void ExpectPoint(const cv::Point2f& actual, const cv::Point2f& expected,
                 const char* case_name, std::size_t index)
{
  if (actual.x != expected.x || actual.y != expected.y)
  {
    std::cerr << case_name << " point[" << index << "]: actual=(" << actual.x << ", "
              << actual.y << ") expected=(" << expected.x << ", " << expected.y << ")\n";
    std::exit(EXIT_FAILURE);
  }
}

void ExpectGeometry(const char* case_name, uint16_t flags,
                    const std::array<cv::Point2f, 4>& expected_points,
                    const cv::Point2f& expected_center, const cv::Rect& expected_box)
{
  const PublishGeometry actual =
      MapPublishGeometry(frame_points, frame_center, frame_box, MakeWideGeometry(flags));
  for (std::size_t index = 0; index < actual.points.size(); ++index)
  {
    ExpectPoint(actual.points[index], expected_points[index], case_name, index);
  }
  ExpectPoint(actual.center, expected_center, case_name, actual.points.size());
  if (actual.box != expected_box)
  {
    std::cerr << case_name << " box: actual=(" << actual.box.x << ", " << actual.box.y
              << ", " << actual.box.width << ", " << actual.box.height << ") expected=("
              << expected_box.x << ", " << expected_box.y << ", " << expected_box.width
              << ", " << expected_box.height << ")\n";
    std::exit(EXIT_FAILURE);
  }
}

void TestWideNoFlip()
{
  ExpectGeometry("wide/no-flip", CameraTypes::FRAME_GEOMETRY_NONE,
                 {cv::Point2f{200.0F, 100.0F}, cv::Point2f{300.0F, 120.0F},
                  cv::Point2f{290.0F, 220.0F}, cv::Point2f{190.0F, 210.0F}},
                 {245.0F, 162.5F}, {180, 80, 140, 160});
}

void TestWideReverseX()
{
  ExpectGeometry("wide/reverse-x", CameraTypes::FRAME_GEOMETRY_REVERSE_X,
                 {cv::Point2f{1138.0F, 120.0F}, cv::Point2f{1238.0F, 100.0F},
                  cv::Point2f{1248.0F, 210.0F}, cv::Point2f{1148.0F, 220.0F}},
                 {1193.0F, 162.5F}, {1120, 80, 140, 160});
}

void TestWideReverseY()
{
  ExpectGeometry("wide/reverse-y", CameraTypes::FRAME_GEOMETRY_REVERSE_Y,
                 {cv::Point2f{190.0F, 868.0F}, cv::Point2f{290.0F, 858.0F},
                  cv::Point2f{300.0F, 958.0F}, cv::Point2f{200.0F, 978.0F}},
                 {245.0F, 915.5F}, {180, 840, 140, 160});
}

void TestWideReverseXY()
{
  ExpectGeometry("wide/reverse-xy",
                 static_cast<uint16_t>(CameraTypes::FRAME_GEOMETRY_REVERSE_X |
                                       CameraTypes::FRAME_GEOMETRY_REVERSE_Y),
                 {cv::Point2f{1148.0F, 858.0F}, cv::Point2f{1248.0F, 868.0F},
                  cv::Point2f{1238.0F, 978.0F}, cv::Point2f{1138.0F, 958.0F}},
                 {1193.0F, 915.5F}, {1120, 840, 140, 160});
}

}  // namespace

int main()
{
  TestWideNoFlip();
  TestWideReverseX();
  TestWideReverseY();
  TestWideReverseXY();
  return EXIT_SUCCESS;
}
