// 不依赖模型文件的测试：v4 解码、SHA-256、编号分类器的输入小图。
// Tests without model files: v4 decoding, SHA-256 and the number classifier's patch.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "ArmorV4Decoder.hpp"
#include "NumberClassifier.hpp"
#include "Sha256.hpp"

namespace
{
void Expect(bool condition, const char* message)
{
  if (!condition)
  {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

bool Near(float a, float b) { return std::fabs(a - b) < 1e-4F; }

/// 一个尺度的 CHW 浮点张量 / Float CHW tensors of one scale.
struct Scale
{
  int h, w, colors;
  std::vector<float> obj, color, size, corner;
  Scale(int h_, int w_, int colors_)
      : h(h_),
        w(w_),
        colors(colors_),
        obj(h * w, -10.0F),
        color(colors * h * w, 0.0F),
        size(2 * h * w, 0.0F),
        corner(8 * h * w, 0.0F)
  {
  }
  void Cell(int i, int j, float logit, int colour, int sz, const float (&offsets)[8])
  {
    obj[i * w + j] = logit;
    color[(colour * h + i) * w + j] = 5.0F;
    size[(sz * h + i) * w + j] = 5.0F;
    for (int c = 0; c < 8; ++c)
    {
      corner[(c * h + i) * w + j] = offsets[c];
    }
  }
  ArmorV4::ScaleOutputs View() const
  {
    using T = ArmorV4::TensorView;
    return {T{obj.data(), T::Type::F32, false, h, w, 1},
            T{color.data(), T::Type::F32, false, h, w, colors},
            T{size.data(), T::Type::F32, false, h, w, 2},
            T{corner.data(), T::Type::F32, false, h, w, 8}};
  }
};

/// colors 为 2（v4）或 4（v7）/ colors is 2 (v4) or 4 (v7).
void TestDecode(int colors)
{
  Scale p3(64, 80, colors);
  Scale p4(32, 40, colors);
  // LT、LB、RB、RT 的偏移，单位 4 个格子 / Offsets in units of 4 cells.
  const float plate[8] = {-0.5F, -0.25F, -0.5F, 0.25F, 0.5F, 0.25F, 0.5F, -0.25F};
  p3.Cell(10, 20, 2.0F, 1, 1, plate);  // A：红、大，得分 0.881 / red, large
  p3.Cell(10, 21, 1.0F, 1, 1, plate);  // B：与 A 重叠，被 NMS 去掉 / overlaps A
  const float small[8] = {-0.25F, -0.125F, -0.25F, 0.125F, 0.25F, 0.125F, 0.25F, -0.125F};
  p4.Cell(20, 30, 0.0F, 0, 0, small);  // C：蓝、小，得分 0.5 / blue, small
  p4.Cell(5, 5, -0.5F, 0, 0, small);   // D：0.378 < 0.4，不出现 / below the threshold
  // E：中心在 (167.5, 87.5) 的大框，与 A 的框 IoU 0.06 逃过 NMS，但包含 A 的中心，按中心
  // 包含去掉 / A large box around A: box IoU 0.06 passes NMS, but it contains A's
  // centre and is dropped by centre containment.
  const float large[8] = {-1.0F, -0.5F, -1.0F, 0.5F, 1.0F, 0.5F, 1.0F, -0.5F};
  p4.Cell(5, 10, 1.5F, 1, 1, large);
  if (colors == 4)
  {
    p4.Cell(25, 5, 0.25F, 3, 0, small);  // F：灭灯、小，得分 0.562 / off, small
  }
  const auto dets = ArmorV4::Decode({p3.View(), p4.View()}, 0.4F, 0.3F);
  Expect(dets.size() == (colors == 4 ? 3U : 2U),
         "A, C (and F) survive; B by NMS, E by centre containment");
  const ArmorV4::Detection& a = dets[0];
  Expect(Near(a.score, 1.0F / (1.0F + std::exp(-2.0F))), "sigmoid score");
  Expect(a.color == 1 && a.size == 1, "argmax colour and size");
  // 格心 (20.5·8 − 0.5, 10.5·8 − 0.5) = (163.5, 83.5)；偏移 × 4 × 8 / Cell centre.
  Expect(Near(a.corners[0].x, 163.5F - 16.0F) && Near(a.corners[0].y, 83.5F - 8.0F),
         "A LT");
  Expect(Near(a.corners[2].x, 163.5F + 16.0F) && Near(a.corners[2].y, 83.5F + 8.0F),
         "A RB");
  const ArmorV4::Detection& c = dets.back();
  Expect(c.color == 0 && c.size == 0, "C blue small");
  if (colors == 4)
  {
    Expect(dets[1].color == 3 && dets[1].size == 0, "F off small");
  }
  Expect(Near(c.corners[3].x, 487.5F + 16.0F) && Near(c.corners[3].y, 327.5F - 8.0F),
         "stride 16 RT");
}

void TestQuantizedHwcView()
{
  // 量化 HWC 视图与浮点 CHW 视图读出同样的值 / A quantised HWC view reads the same
  // values as the float CHW view.
  const float scale = 0.1F;
  const float zero = 128.0F;
  const int h = 2, w = 3, c = 2;
  std::vector<float> chw(c * h * w);
  std::vector<uint8_t> hwc(h * w * c);
  for (int k = 0; k < c; ++k)
  {
    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        const int q = 100 + 10 * k + 3 * y + x;
        hwc[(y * w + x) * c + k] = static_cast<uint8_t>(q);
        chw[(k * h + y) * w + x] = (static_cast<float>(q) - zero) * scale;
      }
    }
  }
  using T = ArmorV4::TensorView;
  const T f{chw.data(), T::Type::F32, false, h, w, c};
  const T u{hwc.data(), T::Type::U8, true, h, w, c, scale, zero};
  for (int k = 0; k < c; ++k)
  {
    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        Expect(Near(f.At(k, y, x), u.At(k, y, x)), "U8 HWC equals F32 CHW");
      }
    }
  }
}

void TestSha256()
{
  const auto dir = std::filesystem::temp_directory_path();
  const auto write = [&dir](const char* name, const std::string& text)
  {
    std::ofstream(dir / name, std::ios::binary) << text;
    return (dir / name).string();
  };
  Expect(Sha256::OfFile(write("sha_empty", "")) ==
             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
         "empty");
  Expect(Sha256::OfFile(write("sha_abc", "abc")) ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "abc");
  Expect(Sha256::OfFile(write("sha_million", std::string(1000000, 'a'))) ==
             "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
         "one million a");
  Expect(Sha256::OfFile((dir / "sha_missing_file").string()).empty(), "missing file");
}

void TestNumberPatch()
{
  // 角点正好是标准位置时，小图就是原图左上 36×40；标准化后均值 0。
  // With corners at the canonical points the patch is the top-left 36×40 crop.
  cv::Mat gray(512, 640, CV_8UC1, cv::Scalar(50));
  cv::rectangle(gray, {10, 10, 10, 10}, cv::Scalar(200), cv::FILLED);
  const std::array<cv::Point2f, 4> canon = {cv::Point2f{2, 12}, cv::Point2f{2, 28},
                                            cv::Point2f{33, 28}, cv::Point2f{33, 12}};
  const cv::Mat patch = NumberClassifier::Patch(gray, canon);
  Expect(patch.rows == 40 && patch.cols == 36 && patch.type() == CV_32F, "36x40 float");
  cv::Scalar mean;
  cv::Scalar stddev;
  cv::meanStdDev(patch, mean, stddev);
  Expect(std::fabs(mean[0]) < 1e-4, "zero mean");
  // 原图标准差 s，标准化后为 s / (s + 4) / Standardised std is s / (s + 4).
  cv::Scalar raw_mean;
  cv::Scalar raw_std;
  cv::meanStdDev(gray(cv::Rect(0, 0, 36, 40)), raw_mean, raw_std);
  Expect(std::fabs(stddev[0] - raw_std[0] / (raw_std[0] + 4.0)) < 1e-4, "std + 4");
}
}  // namespace

int main()
{
  TestDecode(2);
  TestDecode(4);
  TestQuantizedHwcView();
  TestSha256();
  TestNumberPatch();
  std::puts("armor_detector_decoder_test passed");
  return 0;
}
