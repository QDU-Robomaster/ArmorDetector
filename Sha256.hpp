#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

/// 文件 SHA-256，用于启动时核对模型文件 / File SHA-256 for checking model files.
namespace Sha256
{
namespace Detail
{
inline constexpr std::array<uint32_t, 64> K = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};

inline uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

inline void Block(std::array<uint32_t, 8>& h, const uint8_t* p)
{
  std::array<uint32_t, 64> w{};
  for (int i = 0; i < 16; ++i)
  {
    w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
           (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
  }
  for (int i = 16; i < 64; ++i)
  {
    const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  std::array<uint32_t, 8> v = h;
  for (int i = 0; i < 64; ++i)
  {
    const uint32_t s1 = Rotr(v[4], 6) ^ Rotr(v[4], 11) ^ Rotr(v[4], 25);
    const uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
    const uint32_t t1 = v[7] + s1 + ch + K[i] + w[i];
    const uint32_t s0 = Rotr(v[0], 2) ^ Rotr(v[0], 13) ^ Rotr(v[0], 22);
    const uint32_t maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
    v = {t1 + s0 + maj, v[0], v[1], v[2], v[3] + t1, v[4], v[5], v[6]};
  }
  for (int i = 0; i < 8; ++i)
  {
    h[i] += v[i];
  }
}
}  // namespace Detail

/// 返回小写十六进制摘要；读文件失败返回空串 / Lowercase hex digest; empty on a read
/// error.
inline std::string OfFile(const std::string& path)
{
  FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr)
  {
    return {};
  }
  std::array<uint32_t, 8> h = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                               0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<uint8_t, 64 * 1024> buffer{};
  uint64_t total = 0;
  std::size_t n = 0;
  std::size_t pending = 0;  // buffer 开头尚未处理的字节 / Unprocessed bytes at the start
  while ((n = std::fread(buffer.data() + pending, 1, buffer.size() - pending, file)) > 0)
  {
    total += n;
    pending += n;
    std::size_t done = 0;
    for (; pending - done >= 64; done += 64)
    {
      Detail::Block(h, buffer.data() + done);
    }
    for (std::size_t i = done; i < pending; ++i)
    {
      buffer[i - done] = buffer[i];
    }
    pending -= done;
  }
  const bool read_error = std::ferror(file) != 0;
  std::fclose(file);
  if (read_error)
  {
    return {};
  }
  // 补位：0x80、零、64 位位长 / Padding: 0x80, zeros, 64-bit bit length.
  std::array<uint8_t, 128> tail{};
  for (std::size_t i = 0; i < pending; ++i)
  {
    tail[i] = buffer[i];
  }
  tail[pending] = 0x80;
  const std::size_t tail_len = pending + 9 <= 64 ? 64 : 128;
  const uint64_t bits = total * 8;
  for (int i = 0; i < 8; ++i)
  {
    tail[tail_len - 1 - i] = static_cast<uint8_t>(bits >> (8 * i));
  }
  for (std::size_t i = 0; i < tail_len; i += 64)
  {
    Detail::Block(h, tail.data() + i);
  }
  std::string hex;
  for (uint32_t word : h)
  {
    char part[9];
    std::snprintf(part, sizeof(part), "%08x", word);
    hex += part;
  }
  return hex;
}
}  // namespace Sha256
