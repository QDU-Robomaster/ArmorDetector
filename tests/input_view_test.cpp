#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "ArmorDetectorInputView.hpp"

namespace
{

using armor_detector_detail::BindRawRgbInputView;
using armor_detector_detail::MatchesRawRgbInputView;
using armor_detector_detail::RawRgbInputViewSpec;

void Expect(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

RawRgbInputViewSpec MakeSpec(std::vector<uint8_t>& storage, int width, int height)
{
  return {storage.data(), storage.size(), width, height};
}

bool MatchesConstSlotView(const std::vector<uint8_t>& storage, int width, int height,
                          const cv::Mat& input)
{
  return input.data == storage.data() &&
         MatchesRawRgbInputView({input.data, storage.size(), width, height}, input);
}

cv::Mat MakePattern(int width, int height)
{
  cv::Mat image(height, width, CV_8UC3);
  for (int row = 0; row < image.rows; ++row)
  {
    for (int col = 0; col < image.cols; ++col)
    {
      image.at<cv::Vec3b>(row, col) = {
          static_cast<uint8_t>(row * 17 + col),
          static_cast<uint8_t>(row * 11 + col * 3 + 1),
          static_cast<uint8_t>(row * 5 + col * 7 + 2),
      };
    }
  }
  return image;
}

bool PrepareLikePipeline(const cv::Mat& bgr, const RawRgbInputViewSpec& spec,
                         cv::Mat& slot_input)
{
  if (!MatchesRawRgbInputView(spec, slot_input))
  {
    return false;
  }

  cv::Mat working_input = slot_input;
  if (bgr.empty())
  {
    return false;
  }

  cv::Mat resized_bgr;
  cv::resize(bgr, resized_bgr, cv::Size(spec.width, spec.height));
  cv::cvtColor(resized_bgr, working_input, cv::COLOR_BGR2RGB);
  return MatchesRawRgbInputView(spec, slot_input) &&
         MatchesRawRgbInputView(spec, working_input);
}

void TestInPlaceBytesMatchReferenceAndGuard()
{
  constexpr int width = 4;
  constexpr int height = 3;
  std::vector<uint8_t> storage(width * height * 3, 0xA5U);
  const auto spec = MakeSpec(storage, width, height);
  cv::Mat slot_input;
  Expect(BindRawRgbInputView(spec, slot_input), "binding the slot input must pass");

  const cv::Mat bgr = MakePattern(width, height);
  cv::Mat expected;
  cv::cvtColor(bgr, expected, cv::COLOR_BGR2RGB);
  Expect(PrepareLikePipeline(bgr, spec, slot_input),
         "same-size preprocessing must retain the slot view");
  Expect(MatchesRawRgbInputView(spec, slot_input),
         "identity guard must accept the retained slot view");
  Expect(std::equal(storage.begin(), storage.end(), expected.datastart, expected.dataend),
         "in-place RGB bytes must match the ordinary OpenCV result");

  std::vector<uint8_t> other_storage(storage.size(), 0U);
  cv::Mat wrong_pointer(height, width, CV_8UC3, other_storage.data());
  Expect(!MatchesRawRgbInputView(spec, wrong_pointer),
         "identity guard must reject an equal-shaped foreign buffer");
}

void TestWrongSizeRebindsOnlyWorkingHeader()
{
  constexpr int width = 4;
  constexpr int height = 3;
  std::vector<uint8_t> storage(width * height * 3, 0x5AU);
  const std::vector<uint8_t> before = storage;
  const auto spec = MakeSpec(storage, width, height);
  cv::Mat slot_input;
  Expect(BindRawRgbInputView(spec, slot_input), "binding the slot input must pass");

  cv::Mat working_input = slot_input;
  const cv::Mat wrong_size = MakePattern(width - 1, height - 1);
  cv::cvtColor(wrong_size, working_input, cv::COLOR_BGR2RGB);

  Expect(!MatchesRawRgbInputView(spec, working_input),
         "guard must reject a working header rebound to the wrong size");
  Expect(MatchesRawRgbInputView(spec, slot_input),
         "a local-header rebind must not alter the persistent slot header");
  Expect(storage == before,
         "a wrong-size destination rebind must not write the slot buffer");
}

void TestConstSlotIdentityPath()
{
  constexpr int width = 4;
  constexpr int height = 3;
  std::vector<uint8_t> storage(width * height * 3, 0x6BU);
  const auto spec = MakeSpec(storage, width, height);
  cv::Mat slot_input;
  Expect(BindRawRgbInputView(spec, slot_input), "binding the slot input must pass");

  const std::vector<uint8_t>& const_storage = storage;
  Expect(MatchesConstSlotView(const_storage, width, height, slot_input),
         "the const slot path must accept its retained input view");

  std::vector<uint8_t> other_storage(storage.size(), 0U);
  const cv::Mat foreign_input(height, width, CV_8UC3, other_storage.data());
  Expect(!MatchesConstSlotView(const_storage, width, height, foreign_input),
         "the const slot path must reject a foreign input pointer");
}

void TestEmptySourceRollsBack()
{
  constexpr int width = 4;
  constexpr int height = 3;
  std::vector<uint8_t> storage(width * height * 3, 0x3CU);
  const std::vector<uint8_t> before = storage;
  const auto spec = MakeSpec(storage, width, height);
  cv::Mat slot_input;
  Expect(BindRawRgbInputView(spec, slot_input), "binding the slot input must pass");

  Expect(!PrepareLikePipeline({}, spec, slot_input), "an empty source must fail closed");
  Expect(MatchesRawRgbInputView(spec, slot_input),
         "an empty-source failure must retain the persistent slot header");
  Expect(storage == before, "an empty-source failure must retain the slot bytes");
}

void TestInvalidSpecClearsDestination()
{
  std::vector<uint8_t> storage(12U, 0U);
  cv::Mat input(1, 1, CV_8UC3, storage.data());
  const RawRgbInputViewSpec invalid{storage.data(), storage.size() - 1U, 2, 2};
  Expect(!BindRawRgbInputView(invalid, input), "a byte-count mismatch must fail binding");
  Expect(input.empty(), "a failed bind must clear the destination header");
}

}  // namespace

int main()
{
  try
  {
    TestInPlaceBytesMatchReferenceAndGuard();
    TestWrongSizeRebindsOnlyWorkingHeader();
    TestConstSlotIdentityPath();
    TestEmptySourceRollsBack();
    TestInvalidSpecClearsDestination();
    std::cout << "armor_detector_input_view_test: PASS\n";
    return EXIT_SUCCESS;
  }
  catch (const std::exception& exception)
  {
    std::cerr << "armor_detector_input_view_test: FAIL: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
}
