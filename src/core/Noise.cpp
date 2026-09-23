#include "Noise.h"

#include <cmath>

namespace cosmic {
namespace {

constexpr float kF4 = 0.309016994374947f;  // (sqrt(5) - 1) / 4
constexpr float kG4 = 0.138196601125011f;  // (5 - sqrt(5)) / 20

// The 32 edge midpoints of a 4D hypercube.
const signed char kGrad4[32][4] = {
    {0, 1, 1, 1},   {0, 1, 1, -1},   {0, 1, -1, 1},   {0, 1, -1, -1},  {0, -1, 1, 1},  {0, -1, 1, -1},
    {0, -1, -1, 1}, {0, -1, -1, -1}, {1, 0, 1, 1},    {1, 0, 1, -1},   {1, 0, -1, 1},  {1, 0, -1, -1},
    {-1, 0, 1, 1},  {-1, 0, 1, -1},  {-1, 0, -1, 1},  {-1, 0, -1, -1}, {1, 1, 0, 1},   {1, 1, 0, -1},
    {1, -1, 0, 1},  {1, -1, 0, -1},  {-1, 1, 0, 1},   {-1, 1, 0, -1},  {-1, -1, 0, 1}, {-1, -1, 0, -1},
    {1, 1, 1, 0},   {1, 1, -1, 0},   {1, -1, 1, 0},   {1, -1, -1, 0},  {-1, 1, 1, 0},  {-1, 1, -1, 0},
    {-1, -1, 1, 0}, {-1, -1, -1, 0}};

inline int FastFloor(float v) {
    const int i = static_cast<int>(v);
    return v < static_cast<float>(i) ? i - 1 : i;
}

inline int GradientIndex(int i, int j, int k, int l, std::uint32_t seed) {
    std::uint32_t h = static_cast<std::uint32_t>(i) * 0x8da6b343u;
    h ^= static_cast<std::uint32_t>(j) * 0xd8163841u;
    h ^= static_cast<std::uint32_t>(k) * 0xcb1ab31fu;
    h ^= static_cast<std::uint32_t>(l) * 0x165667b1u;
    return static_cast<int>(Hash32(h ^ seed) & 31u);
}

inline float Corner(float x, float y, float z, float w, int gi) {
    float t = 0.6f - x * x - y * y - z * z - w * w;
    if (t <= 0.0f) return 0.0f;
    t *= t;
    const signed char* g = kGrad4[gi];
    return t * t * (g[0] * x + g[1] * y + g[2] * z + g[3] * w);
}

}  // namespace

float Simplex4(float x, float y, float z, float w, std::uint32_t seed) {
    const float s = (x + y + z + w) * kF4;
    const int i = FastFloor(x + s);
    const int j = FastFloor(y + s);
    const int k = FastFloor(z + s);
    const int l = FastFloor(w + s);
    const float t = static_cast<float>(i + j + k + l) * kG4;
    const float x0 = x - (static_cast<float>(i) - t);
    const float y0 = y - (static_cast<float>(j) - t);
    const float z0 = z - (static_cast<float>(k) - t);
    const float w0 = w - (static_cast<float>(l) - t);

    // Which of the 24 simplices we are in follows from the order of the
    // offsets' magnitudes.
    int rank_x = 0, rank_y = 0, rank_z = 0, rank_w = 0;
    if (x0 > y0) ++rank_x; else ++rank_y;
    if (x0 > z0) ++rank_x; else ++rank_z;
    if (x0 > w0) ++rank_x; else ++rank_w;
    if (y0 > z0) ++rank_y; else ++rank_z;
    if (y0 > w0) ++rank_y; else ++rank_w;
    if (z0 > w0) ++rank_z; else ++rank_w;

    const int i1 = rank_x >= 3, j1 = rank_y >= 3, k1 = rank_z >= 3, l1 = rank_w >= 3;
    const int i2 = rank_x >= 2, j2 = rank_y >= 2, k2 = rank_z >= 2, l2 = rank_w >= 2;
    const int i3 = rank_x >= 1, j3 = rank_y >= 1, k3 = rank_z >= 1, l3 = rank_w >= 1;

    const float x1 = x0 - i1 + kG4, y1 = y0 - j1 + kG4, z1 = z0 - k1 + kG4, w1 = w0 - l1 + kG4;
    const float x2 = x0 - i2 + 2.0f * kG4, y2 = y0 - j2 + 2.0f * kG4, z2 = z0 - k2 + 2.0f * kG4,
                w2 = w0 - l2 + 2.0f * kG4;
    const float x3 = x0 - i3 + 3.0f * kG4, y3 = y0 - j3 + 3.0f * kG4, z3 = z0 - k3 + 3.0f * kG4,
                w3 = w0 - l3 + 3.0f * kG4;
    const float x4 = x0 - 1.0f + 4.0f * kG4, y4 = y0 - 1.0f + 4.0f * kG4, z4 = z0 - 1.0f + 4.0f * kG4,
                w4 = w0 - 1.0f + 4.0f * kG4;

    float n = Corner(x0, y0, z0, w0, GradientIndex(i, j, k, l, seed));
    n += Corner(x1, y1, z1, w1, GradientIndex(i + i1, j + j1, k + k1, l + l1, seed));
    n += Corner(x2, y2, z2, w2, GradientIndex(i + i2, j + j2, k + k2, l + l2, seed));
    n += Corner(x3, y3, z3, w3, GradientIndex(i + i3, j + j3, k + k3, l + l3, seed));
    n += Corner(x4, y4, z4, w4, GradientIndex(i + 1, j + 1, k + 1, l + 1, seed));
    return 27.0f * n;
}

float LoopingFbm(float x, float y, float phase, const FbmSettings& settings) {
    const float cz = std::cos(phase) * settings.loop_radius;
    const float cw = std::sin(phase) * settings.loop_radius;
    float octaves = settings.octaves < 1.0f ? 1.0f : (settings.octaves > 10.0f ? 10.0f : settings.octaves);

    float sum = 0.0f;
    float norm = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    for (int o = 0; octaves > 0.0f; ++o) {
        const float weight = octaves >= 1.0f ? 1.0f : octaves;
        // A per-octave shift keeps the octaves from sharing a lattice origin,
        // which would line their features up into a visible grid.
        const float shift = static_cast<float>(o) * 19.19f;
        const float n = Simplex4(x * frequency + shift, y * frequency - shift, cz * frequency, cw * frequency,
                                 settings.seed + static_cast<std::uint32_t>(o) * 0x632be5abu);
        sum += n * amplitude * weight;
        norm += amplitude * weight;
        amplitude *= 0.5f;
        frequency *= 2.0f;
        octaves -= 1.0f;
    }
    return norm > 0.0f ? sum / norm : 0.0f;
}

}  // namespace cosmic
