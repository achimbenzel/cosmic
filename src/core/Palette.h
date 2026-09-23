#pragma once

#include <cstdint>

#include "Color.h"

namespace cosmic {

constexpr int kStopCount = 5;

// How the colours between two stops are mixed.
enum class ColorBlend {
    kOklabSmooth = 0,  // perceptual, with a monotone spline through the stops
    kOklab = 1,        // perceptual, straight segments
    kLinearLight = 2,  // physical mix of light
    kSrgb = 3          // mixed on the encoded values, like a classic 8-bit ramp
};

struct PalettePreset {
    const char* name;
    std::uint8_t srgb[kStopCount][3];  // display-referred sRGB, 8 bit
};

// Curated palettes. The plug-in's popup lists these in order, followed by
// "Custom", so the order is part of the saved project format: append only.
int PresetCount();
const PalettePreset& Preset(int index);

// Pipe-separated preset names plus a trailing "Custom", ready for a popup.
const char* PresetPopupString();

// A palette baked into a lookup table of linear-light colours, so the per-pixel
// cost of any blend mode is one interpolated fetch.
class GradientLut {
public:
    static constexpr int kSize = kLutSize;

    // `stops` are linear-light colours, evenly spaced from t = 0 to t = 1.
    void Build(const Rgb (&stops)[kStopCount], ColorBlend blend, bool reverse);

    // `t` is clamped to [0, 1].
    Rgb Sample(float t) const { return SampleLut(table_, t); }

    // kSize entries, for uploading to the GPU.
    const Rgb* Table() const { return table_; }

private:
    Rgb table_[kSize];
};

}  // namespace cosmic
