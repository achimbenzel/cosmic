#pragma once

#include <cstdint>

#include "CosmicPipeline.h"
#include "Palette.h"

namespace cosmic {

// The effect's controls in the units the user sees: percentages, degrees,
// pixels, 1-based popup indices and 8-bit colours. The plug-in reads its
// parameters into this and the tests build scenes with it, so both go through
// the same conversion and share the same defaults.
struct UiValues {
    // Palette. `palette` is 1-based; PresetCount() + 1 is "Custom".
    int palette = 1;
    std::uint8_t colors[kStopCount][3] = {{3, 4, 12}, {18, 16, 58}, {58, 28, 122}, {46, 111, 216}, {168, 232, 255}};
    int color_blend = 1;
    bool reverse = false;

    // Gradient. Points are in full-resolution square layer pixels.
    int gradient_type = 1;
    int fit = 1;  // 1 content bounds, 2 layer
    float center_x = 0.0f;
    float center_y = 0.0f;
    float angle_deg = 180.0f;
    float size_pct = 100.0f;  // of the reference box's extent along the gradient
    float cycles = 1.0f;
    float offset_pct = 0.0f;
    int repeat = 1;

    // Depth.
    int depth_shape = 1;
    float depth_pct = 35.0f;
    float depth_x = 0.0f;
    float depth_y = 0.0f;
    float depth_radius_pct = 40.0f;  // of the reference box's longer side
    // Bulge (v1.2). Softness is relative to the shape's typical stroke: 100%
    // raises a stroke all the way to its middle.
    float bulge_pct = 0.0f;
    float rounding_pct = 100.0f;
    float softness_pct = 100.0f;
    float light_angle_deg = -45.0f;
    float contrast_pct = 50.0f;

    // Turbulence, both relative to the reference box's shorter side.
    float turbulence_pct = 5.0f;
    float turbulence_size_pct = 40.0f;
    float complexity = 3.0f;
    float evolution_deg = 0.0f;
    // Off by default, so turning the Angle does not also set the noise moving;
    // switch it on to make one keyframed angle a seamless loop. Projects saved
    // with v1.0 keep the value they were saved with.
    bool evolve_with_angle = false;
    int seed = 0;

    // Focus. Radius and falloff are in percent of the layer height: focus is
    // a camera property, so it does not follow the content.
    float focus_x = 0.0f;
    float focus_y = 0.0f;
    float focus_radius_pct = 25.0f;
    float focus_falloff_pct = 50.0f;
    float defocus_px = 0.0f;

    // Glow.
    float glow_intensity_pct = 60.0f;
    float glow_radius_px = 80.0f;
    float glow_falloff = 1.6f;
    float glow_threshold = 0.4f;
    float glow_softness_pct = 50.0f;
    float protection_pct = 50.0f;

    // Diffusion.
    float diffusion_pct = 20.0f;
    float diffusion_radius_px = 30.0f;

    // Grain.
    float grain_pct = 5.0f;
    float grain_size_px = 1.2f;
    bool animate_grain = false;

    // Composite.
    int matte = 1;
    int blend = 1;
    float opacity_pct = 100.0f;
    bool expand_bounds = true;
    int working_space = 1;

    // Performance.
    bool gpu = true;
};

// Default positions of the point controls, in percent of the layer.
constexpr float kDefaultCenterPct[2] = {50.0f, 50.0f};
constexpr float kDefaultDepthCenterPct[2] = {50.0f, 72.0f};
constexpr float kDefaultFocusPct[2] = {50.0f, 50.0f};

// Places the point controls at their default positions for a layer of the
// given full-resolution size.
void PlaceDefaultPoints(UiValues* ui, float layer_width, float layer_height);

// Copies a preset's colours into the colour controls. Does nothing for
// "Custom" or an out-of-range index.
void ApplyPreset(UiValues* ui, int palette);

// `layer_width` and `layer_height` are the layer's full-resolution size, the
// frame the point controls are given in. `frame` feeds the grain when it
// animates.
CosmicSettings SettingsFromUi(const UiValues& ui, float layer_width, float layer_height, int frame);

}  // namespace cosmic
