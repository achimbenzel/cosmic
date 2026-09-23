#include "Plan.h"

#include <algorithm>
#include <cmath>

namespace cosmic {

// Spreads a blur of the given sigma over pyramid levels whose weights follow a
// 1/r^falloff point spread: a Gaussian of sigma s has its peak at 1/s^2, so a
// level weighted s^(2 - falloff) contributes s^-falloff at its own scale. The
// top level fades in with the fractional part of the size, so animating the
// radius is continuous. Weights sum to one: the blur moves light, it does not
// make any.
BlurPlan MakeBlurPlan(float sigma, float falloff, int max_levels) {
    BlurPlan plan;
    if (!(sigma > 0.0f) || max_levels <= 0) return plan;
    const float first = PyramidLevelSigma(1);
    float position = sigma > first ? 1.0f + std::log2(sigma / first) : 1.0f;
    int top = static_cast<int>(std::ceil(position - 1.0e-4f));
    top = std::clamp(top, 1, std::min(max_levels, kMaxPyramidLevels));
    const float top_weight = std::clamp(position - static_cast<float>(top - 1), 0.05f, 1.0f);

    float sum = 0.0f;
    for (int k = 1; k <= top; ++k) {
        float w = std::pow(PyramidLevelSigma(k), 2.0f - falloff);
        if (k == top) w *= top_weight;
        plan.weights[k] = w;
        sum += w;
    }
    for (int k = 1; k <= top; ++k) plan.weights[k] /= sum;
    plan.levels = top;
    plan.reach = kReachSigmas * PyramidLevelSigma(top) + kReachMargin;
    return plan;
}

int LevelsForSigma(float sigma) {
    int k = 0;
    while (k < kMaxPyramidLevels && PyramidLevelSigma(k) < sigma) ++k;
    return k;
}

// The defocus blends the two pyramid levels around its sigma, so its tail is
// that of a two-Gaussian mixture: find where each component falls to the level
// a single Gaussian reaches at kReachSigmas, and take the farther.
float DefocusReach(float sigma) {
    if (!(sigma > 0.0f)) return 0.0f;
    const int upper = LevelsForSigma(sigma);
    const int lower = upper > 0 ? upper - 1 : 0;
    const float s_lo = PyramidLevelSigma(lower);
    const float s_hi = PyramidLevelSigma(upper);
    const float t = s_hi > s_lo ? std::clamp((sigma - s_lo) / (s_hi - s_lo), 0.0f, 1.0f) : 1.0f;
    const float floor_log = -0.5f * kReachSigmas * kReachSigmas;
    float reach = 0.0f;
    const float weights[2] = {1.0f - t, t};
    const float sigmas[2] = {s_lo, s_hi};
    for (int i = 0; i < 2; ++i) {
        if (weights[i] <= 0.0f || sigmas[i] <= 0.0f) continue;
        const float budget = std::log(weights[i]) - floor_log;
        if (budget > 0.0f) reach = std::max(reach, sigmas[i] * std::sqrt(2.0f * budget));
    }
    return reach + kReachMargin;
}

Field MakeField(const CosmicSettings& s, const ReferenceBox& box) {
    // The layer's frame is mapped proportionally onto the box, so a control at
    // the layer's centre sits at the centre of the content.
    float sx = 1.0f;
    float sy = 1.0f;
    if (s.layer_width > 0.0f && s.layer_height > 0.0f) {
        sx = box.width / s.layer_width;
        sy = box.height / s.layer_height;
    }
    auto map_x = [&](float x) { return box.x0 + x * sx; };
    auto map_y = [&](float y) { return box.y0 + y * sy; };

    Field f;
    const float w = std::max(1.0f, box.width);
    const float h = std::max(1.0f, box.height);
    f.type = s.type;
    f.cx = map_x(s.center_x);
    f.cy = map_y(s.center_y);
    // After Effects angles: 0 is up, clockwise positive, y down.
    f.dx = std::sin(s.angle);
    f.dy = -std::cos(s.angle);
    f.ex = -f.dy;
    f.ey = f.dx;

    // Size 1 makes one palette length span the box: like CSS, a linear ramp
    // covers the box's extent along its own direction, a radial one reaches
    // the corners.
    const float along = std::fabs(w * f.dx) + std::fabs(h * f.dy);
    const float across = std::fabs(w * f.ex) + std::fabs(h * f.ey);
    float extent;
    switch (s.type) {
        case GradientType::kRadial: extent = std::sqrt(w * w + h * h); break;
        case GradientType::kDiamond: extent = along + across; break;
        case GradientType::kLinear:
        case GradientType::kReflected:
        case GradientType::kConic:
        default: extent = along; break;
    }
    const float span = std::max(1.0e-3f, s.size * extent);
    f.inv_span = 1.0f / span;
    f.inv_half_span = 2.0f / span;
    f.cycles = s.cycles;
    f.offset = s.offset;
    f.repeat = s.repeat;
    f.depth_shape = s.depth_shape;
    f.depth = s.depth;
    f.depth_x = map_x(s.depth_x);
    f.depth_y = map_y(s.depth_y);
    f.inv_depth_radius = 1.0f / std::max(1.0e-3f, s.depth_radius * std::max(w, h));
    return f;
}

bool WantsContentBounds(const CosmicSettings& settings) {
    return settings.fit == FitMode::kContentBounds && settings.matte == MatteMode::kLayerAlpha;
}

ReferenceBox BoxFromSourceBounds(const CosmicSettings& settings, int x0, int x1, int y0, int y1, int source_left,
                                 int source_top, float to_full_x, float to_full_y) {
    ReferenceBox layer;
    layer.width = settings.layer_width;
    layer.height = settings.layer_height;
    if (x1 < x0 || y1 < y0) return layer;
    ReferenceBox box;
    box.x0 = static_cast<float>(x0 + source_left) * to_full_x;
    box.y0 = static_cast<float>(y0 + source_top) * to_full_y;
    box.width = static_cast<float>(x1 - x0 + 1) * to_full_x;
    box.height = static_cast<float>(y1 - y0 + 1) * to_full_y;
    return box;
}

WarpPlan MakeWarpPlan(const CosmicSettings& s, const ReferenceBox& box, float blur_scale, int width, int height) {
    WarpPlan plan;
    const float shorter = std::max(1.0f, std::min(box.width, box.height));
    const float turbulence = s.turbulence * shorter;
    const float turbulence_size = s.turbulence_size * shorter;
    plan.active = turbulence > 0.0f && turbulence_size > 0.0f;
    if (!plan.active) return plan;
    // The noise is smooth at the scale asked for, so it is evaluated on a
    // coarse grid a quarter of its finest octave apart.
    const float size_render = turbulence_size * blur_scale;
    const float finest = size_render / std::ldexp(1.0f, static_cast<int>(std::ceil(s.complexity)) - 1);
    plan.step = std::clamp(static_cast<int>(finest * 0.25f), 1, 16);
    plan.nx = width / plan.step + 5;
    plan.ny = height / plan.step + 5;
    plan.inv_size = 1.0f / turbulence_size;
    // Fractal noise sits well inside [-1, 1]; this brings its typical swing up
    // to about the displacement asked for.
    plan.amount = turbulence * 1.6f;
    plan.evolution = s.evolution;
    plan.fbm.octaves = s.complexity;
    plan.fbm.seed = Hash32(s.seed * 0x9e3779b9u + 0x85ebca6bu);
    plan.fbm2 = plan.fbm;
    plan.fbm2.seed = Hash32(plan.fbm.seed ^ 0x5bd1e995u);
    return plan;
}

FinishParams MakeFinishParams(const CosmicSettings& settings, float to_full_x, float to_full_y, bool encode_srgb) {
    FinishParams p;
    p.protect = settings.highlight_protection > 0.0f ? 1 : 0;
    p.protection_knee = 1.0f - 0.6f * std::clamp(settings.highlight_protection, 0.0f, 1.0f);
    p.grain = std::clamp(settings.grain, 0.0f, 1.0f);
    p.grain_inv_size = 1.0f / std::max(0.05f, settings.grain_size);
    p.grain_seed = Hash32(settings.grain_seed ^ 0x27d4eb2fu);
    p.encode_srgb = encode_srgb ? 1 : 0;
    p.to_full_x = to_full_x;
    p.to_full_y = to_full_y;
    return p;
}

GlowThreshold MakeGlowThreshold(const CosmicSettings& settings) {
    GlowThreshold threshold;
    threshold.level = std::max(0.0f, settings.glow_threshold);
    threshold.knee = std::clamp(settings.glow_softness, 0.0f, 1.0f) * std::max(threshold.level, 0.05f);
    return threshold;
}

bool IsLinearWorkingSpace(WorkingSpace space, bool float_depth) {
    switch (space) {
        case WorkingSpace::kLinear: return true;
        case WorkingSpace::kSrgb: return false;
        case WorkingSpace::kAuto:
        default:
            // The usual setups: 32 bpc projects are linearised, 8 and 16 are not.
            return float_depth;
    }
}

}  // namespace cosmic
