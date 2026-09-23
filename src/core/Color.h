#pragma once

#include <cmath>

#include "Pixel.h"

namespace cosmic {

// sRGB transfer functions, extended symmetrically so HDR and negative values
// survive a round trip.
inline float SrgbToLinear(float c) {
    const float s = c < 0.0f ? -1.0f : 1.0f;
    const float a = std::fabs(c);
    if (a <= 0.04045f) return s * a / 12.92f;
    return s * std::pow((a + 0.055f) / 1.055f, 2.4f);
}

inline float LinearToSrgb(float c) {
    const float s = c < 0.0f ? -1.0f : 1.0f;
    const float a = std::fabs(c);
    if (a <= 0.0031308f) return s * a * 12.92f;
    return s * (1.055f * std::pow(a, 1.0f / 2.4f) - 0.055f);
}

struct Rgb {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

struct Lab {
    float l = 0.0f;
    float a = 0.0f;
    float b = 0.0f;
};

// Oklab (Björn Ottosson, 2020). Blending in it keeps the lightness of a ramp
// even and avoids the grey dip that linear RGB puts between complementary
// colours, which is most of what makes a gradient read as "designed".
inline Lab LinearSrgbToOklab(const Rgb& c) {
    const float l = 0.4122214708f * c.r + 0.5363325363f * c.g + 0.0514459929f * c.b;
    const float m = 0.2119034982f * c.r + 0.6806995451f * c.g + 0.1073969566f * c.b;
    const float s = 0.0883024619f * c.r + 0.2817188376f * c.g + 0.6299787005f * c.b;
    const float l_ = std::cbrt(l);
    const float m_ = std::cbrt(m);
    const float s_ = std::cbrt(s);
    return Lab{0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_,
               1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_,
               0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_};
}

inline Rgb OklabToLinearSrgb(const Lab& c) {
    const float l_ = c.l + 0.3963377774f * c.a + 0.2158037573f * c.b;
    const float m_ = c.l - 0.1055613458f * c.a - 0.0638541728f * c.b;
    const float s_ = c.l - 0.0894841775f * c.a - 1.2914855480f * c.b;
    const float l = l_ * l_ * l_;
    const float m = m_ * m_ * m_;
    const float s = s_ * s_ * s_;
    return Rgb{4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
               -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
               -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s};
}

// Smooth shoulder: identity below `knee`, asymptotic to 1 above it, C1 at the
// join.
inline float SoftSaturate(float x, float knee) {
    if (!(x > knee)) return x < 0.0f ? 0.0f : x;
    const float head = 1.0f - knee;
    return knee + head * (1.0f - std::exp(-(x - knee) / head));
}

}  // namespace cosmic
