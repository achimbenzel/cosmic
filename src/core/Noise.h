#pragma once

#include <cstdint>

namespace cosmic {

// Integer hash (lowbias32 by Chris Wellons): cheap, and good enough that
// neighbouring pixels show no visible pattern.
inline std::uint32_t Hash32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline std::uint32_t Hash2(std::int32_t x, std::int32_t y, std::uint32_t seed) {
    return Hash32(static_cast<std::uint32_t>(x) * 0x8da6b343u ^ Hash32(static_cast<std::uint32_t>(y) ^ seed));
}

// Uniform in [0, 1).
inline float HashToUnit(std::uint32_t h) {
    return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}

// Triangular distribution in (-1, 1): the sum of two uniforms. Used for dither
// (it decorrelates the quantisation error from the signal) and for grain.
inline float TriangularNoise(std::int32_t x, std::int32_t y, std::uint32_t seed) {
    const std::uint32_t h = Hash2(x, y, seed);
    return HashToUnit(h) + HashToUnit(Hash32(h ^ 0x9e3779b9u)) - 1.0f;
}

// 4D simplex noise (after Stefan Gustavson's public domain reference), in
// roughly [-1, 1]. Four dimensions let two of them trace a circle, so the
// noise evolves and returns exactly to where it started: a seamless loop.
float Simplex4(float x, float y, float z, float w, std::uint32_t seed);

struct FbmSettings {
    float octaves = 3.0f;     // fractional: the last octave fades in
    float loop_radius = 0.6f; // how far one full evolution turn travels
    std::uint32_t seed = 0;
};

// Fractal sum of Simplex4. `phase` is the evolution angle in radians; phase and
// phase + 2*pi give identical results.
float LoopingFbm(float x, float y, float phase, const FbmSettings& settings);

}  // namespace cosmic
