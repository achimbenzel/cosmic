#include "UiModel.h"

#include <algorithm>
#include <cmath>

namespace cosmic {
namespace {

constexpr float kDegToRad = 3.14159265358979f / 180.0f;

template <typename Enum>
Enum PopupToEnum(int popup, int count) {
    return static_cast<Enum>(std::clamp(popup - 1, 0, count - 1));
}

}  // namespace

void PlaceDefaultPoints(UiValues* ui, float layer_width, float layer_height) {
    ui->center_x = layer_width * kDefaultCenterPct[0] * 0.01f;
    ui->center_y = layer_height * kDefaultCenterPct[1] * 0.01f;
    ui->depth_x = layer_width * kDefaultDepthCenterPct[0] * 0.01f;
    ui->depth_y = layer_height * kDefaultDepthCenterPct[1] * 0.01f;
    ui->focus_x = layer_width * kDefaultFocusPct[0] * 0.01f;
    ui->focus_y = layer_height * kDefaultFocusPct[1] * 0.01f;
}

void ApplyPreset(UiValues* ui, int palette) {
    if (palette < 1 || palette > PresetCount()) return;
    const PalettePreset& preset = Preset(palette - 1);
    for (int k = 0; k < kStopCount; ++k) {
        for (int c = 0; c < 3; ++c) ui->colors[k][c] = preset.srgb[k][c];
    }
    ui->palette = palette;
}

CosmicSettings SettingsFromUi(const UiValues& ui, float layer_width, float layer_height, int frame) {
    CosmicSettings s;
    const float h = layer_height > 0.0f ? layer_height : 1.0f;
    const float pct = 0.01f * h;
    s.fit = PopupToEnum<FitMode>(ui.fit, 2);
    s.layer_width = layer_width > 0.0f ? layer_width : 1.0f;
    s.layer_height = h;

    // Colour controls are display-referred sRGB; the render works in linear
    // light and re-encodes for the working space at the end.
    for (int k = 0; k < kStopCount; ++k) {
        s.stops[k] = Rgb{SrgbToLinear(ui.colors[k][0] / 255.0f), SrgbToLinear(ui.colors[k][1] / 255.0f),
                         SrgbToLinear(ui.colors[k][2] / 255.0f)};
    }
    s.color_blend = PopupToEnum<ColorBlend>(ui.color_blend, 4);
    s.reverse = ui.reverse;

    s.type = PopupToEnum<GradientType>(ui.gradient_type, 5);
    s.center_x = ui.center_x;
    s.center_y = ui.center_y;
    s.angle = ui.angle_deg * kDegToRad;
    s.size = std::max(0.01f, ui.size_pct) * 0.01f;
    s.cycles = std::max(0.0f, ui.cycles);
    s.offset = ui.offset_pct * 0.01f;
    s.repeat = PopupToEnum<RepeatMode>(ui.repeat, 3);

    s.depth_shape = PopupToEnum<DepthShape>(ui.depth_shape, 5);
    s.depth = ui.depth_pct * 0.01f;
    s.depth_x = ui.depth_x;
    s.depth_y = ui.depth_y;
    s.depth_radius = std::max(0.01f, ui.depth_radius_pct) * 0.01f;
    s.bulge = std::max(0.0f, ui.bulge_pct) * 0.01f;
    s.rounding = std::clamp(ui.rounding_pct * 0.01f, 0.0f, 1.0f);
    s.bulge_softness = std::max(1.0f, ui.softness_pct) * 0.01f;
    s.light_angle = ui.light_angle_deg * kDegToRad;
    s.contrast = std::max(0.0f, ui.contrast_pct) * 0.01f;

    s.turbulence = std::max(0.0f, ui.turbulence_pct) * 0.01f;
    s.turbulence_size = std::max(0.1f, ui.turbulence_size_pct) * 0.01f;
    s.complexity = std::clamp(ui.complexity, 1.0f, 8.0f);
    // One keyframed angle drives the loop: when evolution follows the angle, a
    // full turn of the angle brings both the gradient and the noise back to
    // where they started.
    s.evolution = (ui.evolution_deg + (ui.evolve_with_angle ? ui.angle_deg : 0.0f)) * kDegToRad;
    s.seed = static_cast<std::uint32_t>(ui.seed);

    s.focus_x = ui.focus_x;
    s.focus_y = ui.focus_y;
    s.focus_radius = std::max(0.0f, ui.focus_radius_pct) * pct;
    s.focus_falloff = std::max(0.0f, ui.focus_falloff_pct) * pct;
    s.defocus = std::max(0.0f, ui.defocus_px);

    s.glow_intensity = std::max(0.0f, ui.glow_intensity_pct) * 0.01f;
    s.glow_radius = std::max(0.0f, ui.glow_radius_px);
    s.glow_falloff = std::clamp(ui.glow_falloff, 1.0f, 3.0f);
    s.glow_threshold = std::max(0.0f, ui.glow_threshold);
    s.glow_softness = std::clamp(ui.glow_softness_pct * 0.01f, 0.0f, 1.0f);
    s.highlight_protection = std::clamp(ui.protection_pct * 0.01f, 0.0f, 1.0f);

    s.diffusion = std::clamp(ui.diffusion_pct * 0.01f, 0.0f, 1.0f);
    s.diffusion_radius = std::max(0.0f, ui.diffusion_radius_px);

    s.grain = std::clamp(ui.grain_pct * 0.01f, 0.0f, 1.0f);
    s.grain_size = std::max(0.1f, ui.grain_size_px);
    s.grain_seed = static_cast<std::uint32_t>(ui.seed) * 7919u +
                   (ui.animate_grain ? static_cast<std::uint32_t>(frame) * 104729u : 0u);

    s.matte = PopupToEnum<MatteMode>(ui.matte, 3);
    s.blend = PopupToEnum<BlendMode>(ui.blend, 5);
    s.opacity = std::clamp(ui.opacity_pct * 0.01f, 0.0f, 1.0f);
    s.working_space = PopupToEnum<WorkingSpace>(ui.working_space, 3);
    return s;
}

}  // namespace cosmic
