#include "Palette.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace cosmic {
namespace {

// Each palette runs from its deepest colour to its brightest, so the glow,
// which picks out the bright end, always lands on the palette's highlight.
const PalettePreset kPresets[] = {
    {"Deep Space", {{3, 4, 12}, {18, 16, 58}, {58, 28, 122}, {46, 111, 216}, {168, 232, 255}}},
    {"Nebula", {{10, 3, 24}, {59, 10, 87}, {155, 27, 158}, {240, 82, 156}, {255, 199, 225}}},
    {"Aurora", {{2, 11, 20}, {6, 50, 74}, {10, 143, 134}, {82, 224, 166}, {228, 255, 217}}},
    {"Sunset", {{27, 10, 42}, {92, 26, 91}, {199, 56, 102}, {255, 123, 74}, {255, 210, 138}}},
    {"Solar Flare", {{20, 3, 0}, {94, 11, 0}, {209, 56, 10}, {255, 154, 31}, {255, 240, 184}}},
    {"Ocean", {{1, 11, 25}, {3, 43, 82}, {4, 100, 154}, {19, 169, 201}, {166, 244, 255}}},
    {"Vaporwave", {{18, 0, 46}, {75, 18, 184}, {224, 36, 154}, {255, 139, 209}, {126, 245, 255}}},
    {"Ember", {{13, 4, 4}, {61, 12, 12}, {142, 27, 18}, {226, 84, 27}, {255, 176, 103}}},
    {"Emerald", {{2, 13, 8}, {6, 51, 31}, {14, 112, 69}, {61, 191, 122}, {207, 247, 211}}},
    {"Rose Gold", {{26, 14, 18}, {90, 42, 53}, {181, 100, 106}, {232, 164, 142}, {252, 227, 210}}},
    {"Glacier", {{6, 16, 28}, {28, 54, 84}, {79, 127, 168}, {169, 203, 230}, {244, 251, 255}}},
    {"Eclipse", {{0, 0, 0}, {20, 12, 32}, {86, 36, 60}, {232, 120, 60}, {255, 244, 214}}},
    {"Mono", {{0, 0, 0}, {38, 38, 38}, {110, 110, 110}, {189, 189, 189}, {255, 255, 255}}},
};

constexpr int kPresetCount = static_cast<int>(sizeof(kPresets) / sizeof(kPresets[0]));

// Fritsch-Carlson tangents: a cubic through the stops that never overshoots,
// so a smooth palette cannot invent a colour darker or brighter than its
// neighbours, yet has no kink (and so no Mach band) at a stop.
void MonotoneTangents(const float (&y)[kStopCount], float (&m)[kStopCount]) {
    constexpr float h = 1.0f / static_cast<float>(kStopCount - 1);
    float d[kStopCount - 1];
    for (int k = 0; k < kStopCount - 1; ++k) d[k] = (y[k + 1] - y[k]) / h;
    m[0] = d[0];
    m[kStopCount - 1] = d[kStopCount - 2];
    for (int k = 1; k < kStopCount - 1; ++k) {
        m[k] = (d[k - 1] * d[k] <= 0.0f) ? 0.0f : 0.5f * (d[k - 1] + d[k]);
    }
    for (int k = 0; k < kStopCount - 1; ++k) {
        if (d[k] == 0.0f) {
            m[k] = 0.0f;
            m[k + 1] = 0.0f;
            continue;
        }
        const float alpha = m[k] / d[k];
        const float beta = m[k + 1] / d[k];
        const float sum = alpha * alpha + beta * beta;
        if (sum > 9.0f) {
            const float tau = 3.0f / std::sqrt(sum);
            m[k] = tau * alpha * d[k];
            m[k + 1] = tau * beta * d[k];
        }
    }
}

float Hermite(const float (&y)[kStopCount], const float (&m)[kStopCount], float t) {
    constexpr float h = 1.0f / static_cast<float>(kStopCount - 1);
    const float scaled = t * static_cast<float>(kStopCount - 1);
    int k = static_cast<int>(scaled);
    if (k > kStopCount - 2) k = kStopCount - 2;
    if (k < 0) k = 0;
    const float s = scaled - static_cast<float>(k);
    const float s2 = s * s;
    const float s3 = s2 * s;
    return (2.0f * s3 - 3.0f * s2 + 1.0f) * y[k] + (s3 - 2.0f * s2 + s) * h * m[k] +
           (-2.0f * s3 + 3.0f * s2) * y[k + 1] + (s3 - s2) * h * m[k + 1];
}

float Piecewise(const float (&y)[kStopCount], float t) {
    const float scaled = t * static_cast<float>(kStopCount - 1);
    int k = static_cast<int>(scaled);
    if (k > kStopCount - 2) k = kStopCount - 2;
    if (k < 0) k = 0;
    const float s = scaled - static_cast<float>(k);
    return y[k] + (y[k + 1] - y[k]) * s;
}

}  // namespace

int PresetCount() { return kPresetCount; }

const PalettePreset& Preset(int index) {
    return kPresets[std::clamp(index, 0, kPresetCount - 1)];
}

const char* PresetPopupString() {
    static const std::string names = [] {
        std::string joined;
        for (const PalettePreset& preset : kPresets) {
            joined += preset.name;
            joined += '|';
        }
        joined += "Custom";
        return joined;
    }();
    return names.c_str();
}

void GradientLut::Build(const Rgb (&stops)[kStopCount], ColorBlend blend, bool reverse) {
    // Channels of the stops in the space the blend works in.
    float c0[kStopCount];
    float c1[kStopCount];
    float c2[kStopCount];
    for (int k = 0; k < kStopCount; ++k) {
        const Rgb& s = stops[reverse ? kStopCount - 1 - k : k];
        switch (blend) {
            case ColorBlend::kOklabSmooth:
            case ColorBlend::kOklab: {
                const Lab lab = LinearSrgbToOklab(Rgb{std::max(0.0f, s.r), std::max(0.0f, s.g), std::max(0.0f, s.b)});
                c0[k] = lab.l;
                c1[k] = lab.a;
                c2[k] = lab.b;
                break;
            }
            case ColorBlend::kSrgb:
                c0[k] = LinearToSrgb(s.r);
                c1[k] = LinearToSrgb(s.g);
                c2[k] = LinearToSrgb(s.b);
                break;
            case ColorBlend::kLinearLight:
            default:
                c0[k] = s.r;
                c1[k] = s.g;
                c2[k] = s.b;
                break;
        }
    }

    float m0[kStopCount] = {};
    float m1[kStopCount] = {};
    float m2[kStopCount] = {};
    const bool smooth = blend == ColorBlend::kOklabSmooth;
    if (smooth) {
        MonotoneTangents(c0, m0);
        MonotoneTangents(c1, m1);
        MonotoneTangents(c2, m2);
    }

    for (int i = 0; i < kSize; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kSize - 1);
        float v0, v1, v2;
        if (smooth) {
            v0 = Hermite(c0, m0, t);
            v1 = Hermite(c1, m1, t);
            v2 = Hermite(c2, m2, t);
        } else {
            v0 = Piecewise(c0, t);
            v1 = Piecewise(c1, t);
            v2 = Piecewise(c2, t);
        }
        Rgb out;
        switch (blend) {
            case ColorBlend::kOklabSmooth:
            case ColorBlend::kOklab:
                out = OklabToLinearSrgb(Lab{v0, v1, v2});
                break;
            case ColorBlend::kSrgb:
                out = Rgb{SrgbToLinear(v0), SrgbToLinear(v1), SrgbToLinear(v2)};
                break;
            case ColorBlend::kLinearLight:
            default:
                out = Rgb{v0, v1, v2};
                break;
        }
        // Oklab mixes of saturated stops can land a hair outside sRGB.
        table_[i] = Rgb{std::max(0.0f, out.r), std::max(0.0f, out.g), std::max(0.0f, out.b)};
    }
}

}  // namespace cosmic
