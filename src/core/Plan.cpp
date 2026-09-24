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

bool WantsShapeMeasure(const CosmicSettings& settings) {
    return WantsContentBounds(settings) || settings.bulge > 0.0f;
}

ShapeMeasure MeasureShape(const RowStats* rows, int count) {
    ShapeMeasure m;
    int x0 = 0;
    int x1 = -1;
    int y0 = -1;
    int y1 = -1;
    double area = 0.0;
    double outline = 0.0;
    for (int y = 0; y < count; ++y) {
        const RowStats& r = rows[y];
        area += r.area;
        outline += r.outline;
        if (r.first < 0) continue;
        if (y0 < 0) {
            y0 = y;
            x0 = r.first;
            x1 = r.last;
        }
        y1 = y;
        x0 = std::min(x0, r.first);
        x1 = std::max(x1, r.last);
    }
    m.area = static_cast<float>(area);
    m.outline = static_cast<float>(outline);
    if (y0 < 0) return m;

    // A vertical edge covering a fraction c of its column starts 1 - c into
    // it; take the column's best-covered pixel as the edge's.
    float a_left = 0.0f;
    float a_right = 0.0f;
    for (int y = y0; y <= y1; ++y) {
        const RowStats& r = rows[y];
        if (r.first == x0) a_left = std::max(a_left, r.alpha_first);
        if (r.last == x1) a_right = std::max(a_right, r.alpha_last);
    }
    const float a_top = std::clamp(rows[y0].alpha_max, 0.0f, 1.0f);
    const float a_bottom = std::clamp(rows[y1].alpha_max, 0.0f, 1.0f);
    m.left = static_cast<float>(x0) + 1.0f - std::clamp(a_left, 0.0f, 1.0f);
    m.right = static_cast<float>(x1) + std::clamp(a_right, 0.0f, 1.0f);
    m.top = static_cast<float>(y0) + 1.0f - a_top;
    m.bottom = static_cast<float>(y1) + a_bottom;
    // Something thinner than a pixel: a pixel-wide box around it.
    if (m.right - m.left < 1.0f) {
        const float c = 0.5f * (m.left + m.right);
        m.left = c - 0.5f;
        m.right = c + 0.5f;
    }
    if (m.bottom - m.top < 1.0f) {
        const float c = 0.5f * (m.top + m.bottom);
        m.top = c - 0.5f;
        m.bottom = c + 0.5f;
    }
    m.visible = true;
    return m;
}

ReferenceBox LayerBox(const CosmicSettings& settings) {
    ReferenceBox layer;
    layer.width = settings.layer_width;
    layer.height = settings.layer_height;
    return layer;
}

ReferenceBox BoxFromShape(const CosmicSettings& settings, const ShapeMeasure& shape, int source_left,
                          int source_top, float to_full_x, float to_full_y) {
    if (!WantsContentBounds(settings) || !shape.visible) return LayerBox(settings);
    ReferenceBox box;
    box.x0 = (shape.left + static_cast<float>(source_left)) * to_full_x;
    box.y0 = (shape.top + static_cast<float>(source_top)) * to_full_y;
    box.width = (shape.right - shape.left) * to_full_x;
    box.height = (shape.bottom - shape.top) * to_full_y;
    return box;
}

WarpPlan MakeWarpPlan(const CosmicSettings& s, const ReferenceBox& box, float to_full_x, float to_full_y,
                      int canvas_left, int canvas_top, int width, int height) {
    WarpPlan plan;
    const float shorter = std::max(1.0f, std::min(box.width, box.height));
    const float turbulence = s.turbulence * shorter;
    const float size = s.turbulence_size * shorter;
    plan.active = turbulence > 0.0f && size > 0.0f && width > 0 && height > 0;
    if (!plan.active) return plan;

    // The noise is smooth at the scale asked for, so it is evaluated on nodes
    // a quarter of its finest octave apart and reconstructed in between. The
    // spacing is set in noise units, so the reconstruction is the same at any
    // resolution; it only coarsens where nodes would be closer than a render
    // pixel.
    float spacing = 0.25f / std::ldexp(1.0f, static_cast<int>(std::ceil(s.complexity)) - 1);
    const float render_pixel = std::max(to_full_x, to_full_y);
    for (int k = 0; k < 64 && spacing * size < render_pixel; ++k) spacing *= 2.0f;

    WarpLattice& l = plan.lattice;
    l.spacing = spacing;
    l.anchor_x = box.x0 + 0.5f * box.width;
    l.anchor_y = box.y0 + 0.5f * box.height;
    l.scale = 1.0f / (size * spacing);
    // Cover the canvas with the B-spline's reach and a node to spare.
    const float gx0 = WarpGridCoordinate(0, canvas_left, to_full_x, l.anchor_x, l.scale, 0);
    const float gx1 = WarpGridCoordinate(width - 1, canvas_left, to_full_x, l.anchor_x, l.scale, 0);
    const float gy0 = WarpGridCoordinate(0, canvas_top, to_full_y, l.anchor_y, l.scale, 0);
    const float gy1 = WarpGridCoordinate(height - 1, canvas_top, to_full_y, l.anchor_y, l.scale, 0);
    l.origin_i = static_cast<int>(std::floor(gx0)) - 2;
    l.origin_j = static_cast<int>(std::floor(gy0)) - 2;
    l.nx = static_cast<int>(std::floor(gx1)) + 4 - l.origin_i;
    l.ny = static_cast<int>(std::floor(gy1)) + 4 - l.origin_j;

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

float SmoothLevelSigma(int level) {
    if (level <= 0) return 0.0f;
    // Each level adds the B3-spline's variance, 1, times its spread squared.
    return std::sqrt((std::ldexp(1.0f, 2 * level) - 1.0f) / 3.0f);
}

ReliefPlan MakeReliefPlan(const CosmicSettings& s, const ReferenceBox& box, const ShapeMeasure& shape,
                          const ReliefGeometry& g) {
    ReliefPlan plan;
    if (!(s.bulge > 0.0f) || !shape.visible || !(shape.area > 0.0f) || !(shape.outline > 0.0f)) return plan;
    if (g.width <= 0 || g.height <= 0 || g.source_width <= 0 || g.source_height <= 0) return plan;
    const float to_full_x = g.to_full_x;
    const float to_full_y = g.to_full_y;
    const float blur_scale = g.blur_scale;

    // Twice the area over the outline is a stroke's width, for text and
    // anything else made of strokes: the relief's scale, so it fits bold and
    // thin type alike and grows with the shape.
    const float longer = std::max(1.0f, std::max(box.width, box.height));
    const float stroke = std::clamp(2.0f * shape.area / shape.outline, 1.0f, 4.0f * longer);
    const float rise = std::max(0.5f, s.bulge_softness * stroke * 0.5f);  // full-resolution pixels

    // Blurred by half the rise, the outline's 0.5 contour ramps up to 0.95
    // over the rise.
    const float sigma = std::max(0.5f * rise * blur_scale, SmoothLevelSigma(1));
    int lo = 1;
    while (lo < kMaxReliefLevels && SmoothLevelSigma(lo + 1) <= sigma) ++lo;
    float mix = 0.0f;
    if (lo < kMaxReliefLevels) {
        const float s_lo = SmoothLevelSigma(lo);
        const float s_hi = SmoothLevelSigma(lo + 1);
        mix = std::clamp((sigma - s_lo) / (s_hi - s_lo), 0.0f, 1.0f);
    }
    plan.lo = lo;
    plan.mix = mix;
    plan.hi = mix > 0.0f ? lo + 1 : lo;

    // The pyramid's reach: each level's filter spans two taps of its step
    // either side.
    const int reach = 2 * ((1 << plan.hi) - 1);
    const int source_x = g.source_left - g.canvas_left;
    const int source_y = g.source_top - g.canvas_top;
    const int left = std::min(0, std::max(source_x - reach, -kMaxReliefMargin));
    const int top = std::min(0, std::max(source_y - reach, -kMaxReliefMargin));
    const int right = std::max(g.width, std::min(source_x + g.source_width + reach, g.width + kMaxReliefMargin));
    const int bottom = std::max(g.height, std::min(source_y + g.source_height + reach, g.height + kMaxReliefMargin));
    plan.domain_left = left;
    plan.domain_top = top;
    plan.domain_width = right - left;
    plan.domain_height = bottom - top;
    plan.invert = s.matte == MatteMode::kInvertedAlpha ? 1 : 0;

    ReliefShape& r = plan.shape;
    r.inv_full_x = 1.0f / to_full_x;
    r.inv_full_y = 1.0f / to_full_y;
    r.height = rise;
    r.rounding = std::clamp(s.rounding, 0.0f, 1.0f);
    const float shorter = std::max(1.0f, std::min(box.width, box.height));
    r.refraction = s.bulge * kReliefRefraction * shorter;
    r.shade = s.bulge * std::max(0.0f, s.contrast) * kReliefShade;
    r.rim = s.bulge * kReliefRim;
    r.light_x = std::sin(s.light_angle);
    r.light_y = -std::cos(s.light_angle);
    // The light stands kLightElevation above the surface, the eye straight on.
    const float ce = std::cos(kLightElevation);
    const float se = std::sin(kLightElevation);
    float hx = r.light_x * ce;
    float hy = r.light_y * ce;
    float hz = se + 1.0f;
    const float hn = std::sqrt(hx * hx + hy * hy + hz * hz);
    r.half_x = hx / hn;
    r.half_y = hy / hn;
    r.half_z = hz / hn;
    r.shininess = kGlintShininess;
    r.glint_floor = std::pow(r.half_z, r.shininess);
    r.glint = s.bulge * std::max(0.0f, s.contrast) * kReliefGlint;
    if (!(r.glint_floor < 0.999f)) r.glint = 0.0f;
    plan.active = true;
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
