#pragma once

// Decisions both renderers must make identically - which pyramid levels a
// blur uses and with what weights, how far light can reach, where the field
// sits, how the turbulence grid is laid out - made once, on the host, so the
// CPU and GPU pipelines cannot disagree about them.

#include "CosmicPipeline.h"
#include "Pyramid.h"
#include "Shared.h"

namespace cosmic {

// Glow, diffusion and defocus bounds: measured, a glow falls below a thousandth
// of its value next to the source within 3.0 to 3.6 sigma of its widest
// level, so this keeps everything visible inside the bounds without rendering
// far more empty canvas than needed.
constexpr float kReachSigmas = 3.4f;
constexpr float kReachMargin = 8.0f;

// Diffusion is a veil rather than a bloom: its levels are weighted a little
// towards the fine end, like the scatter of a mist filter.
constexpr float kDiffusionFalloff = 2.3f;

struct BlurPlan {
    int levels = 0;                             // pyramid levels 1..levels
    float weights[kMaxPyramidLevels + 1] = {};  // indexed by level
    float reach = 0.0f;                         // render px
};

// Spreads a blur of the given sigma over pyramid levels whose weights follow a
// 1/r^falloff point spread. Weights sum to one.
BlurPlan MakeBlurPlan(float sigma, float falloff, int max_levels);

// Levels needed so the coarsest one is at least `sigma` wide.
int LevelsForSigma(float sigma);

// How far a defocus of this sigma (render px) spreads light.
float DefocusReach(float sigma);

// The field resolved onto the reference box.
Field MakeField(const CosmicSettings& settings, const ReferenceBox& box);

// True when the settings want the reference box from the layer's pixels.
bool WantsContentBounds(const CosmicSettings& settings);

// True when the render needs the layer's shape measured: for content bounds,
// or for the Bulge relief's scale.
bool WantsShapeMeasure(const CosmicSettings& settings);

// Alpha above which a pixel counts towards the content bounds.
constexpr float kVisibleAlpha = 0.5f / 255.0f;

// The layer's shape, from its RowStats.
struct ShapeMeasure {
    bool visible = false;
    // Bounds of the visible pixels in the source buffer's pixels, to a
    // fraction of a pixel: an edge column's coverage says how far into it the
    // shape reaches, so the bounds glide with moving content instead of
    // jumping a pixel at a time.
    float left = 0.0f;
    float top = 0.0f;
    float right = 0.0f;
    float bottom = 0.0f;
    float area = 0.0f;     // full-resolution square pixels
    float outline = 0.0f;  // full-resolution pixels
};

// Rows in order, one per row of the source buffer.
ShapeMeasure MeasureShape(const RowStats* rows, int count);

// The layer's own box.
ReferenceBox LayerBox(const CosmicSettings& settings);

// The reference box: the measured shape's bounds when the settings fit to
// content and anything is visible, else the layer.
ReferenceBox BoxFromShape(const CosmicSettings& settings, const ShapeMeasure& shape, int source_left,
                          int source_top, float to_full_x, float to_full_y);

// Turbulence: the lattice covering a canvas of render pixels, and the noise.
struct WarpPlan {
    bool active = false;
    WarpLattice lattice;
    float amount = 0.0f;
    float evolution = 0.0f;
    FbmSettings fbm;
    FbmSettings fbm2;
};

WarpPlan MakeWarpPlan(const CosmicSettings& settings, const ReferenceBox& box, float to_full_x, float to_full_y,
                      int canvas_left, int canvas_top, int width, int height);

// Bulge relief. The shape's alpha is blurred through an undecimated pyramid
// (see SmoothAt) - levels lo and hi mixed by `mix` - and the relief is read
// from the blur's slope.
//
// The pyramid is built on a domain reaching past the canvas wherever the
// layer comes within the blur's reach of the canvas's edge, so the blur
// sees the layer's true surroundings rather than an edge: the relief then
// depends only on the shape, not on where the canvas happens to end.
struct ReliefPlan {
    bool active = false;
    int lo = 1;
    int hi = 1;  // the last level to build: lo, or lo + 1 when mixing
    float mix = 0.0f;
    int invert = 0;  // raised from the space around the layer (inverted matte)
    int domain_left = 0;  // the domain, in canvas pixels
    int domain_top = 0;
    int domain_width = 0;
    int domain_height = 0;
    ReliefShape shape;
};

constexpr int kMaxReliefLevels = 14;
// How far past the canvas the domain may reach, in render pixels: beyond it a
// very wide relief sees the canvas's edge again.
constexpr int kMaxReliefMargin = 512;

// Blur, in render pixels, of level k of the relief's pyramid.
float SmoothLevelSigma(int level);

// The distance between taps of level k's filter.
inline int SmoothLevelStep(int level) { return 1 << (level - 1); }

// Strengths at Bulge 100%: the lookup moves by this share of the box's
// shorter side under a vertical wall; the palette shifts by the others (the
// light's two scaled by Contrast).
constexpr float kReliefRefraction = 0.3f;
constexpr float kReliefShade = 0.35f;
constexpr float kReliefRim = 0.1f;
constexpr float kReliefGlint = 0.6f;
constexpr float kLightElevation = 0.7f;  // radians above the surface
constexpr float kGlintShininess = 40.0f;

// The canvas and the source are given in layer render pixels, like the
// buffers' origins.
struct ReliefGeometry {
    int canvas_left = 0;
    int canvas_top = 0;
    int width = 0;
    int height = 0;
    int source_left = 0;
    int source_top = 0;
    int source_width = 0;
    int source_height = 0;
    float to_full_x = 1.0f;
    float to_full_y = 1.0f;
    float blur_scale = 1.0f;
};

ReliefPlan MakeReliefPlan(const CosmicSettings& settings, const ReferenceBox& box, const ShapeMeasure& shape,
                          const ReliefGeometry& geometry);

FinishParams MakeFinishParams(const CosmicSettings& settings, float to_full_x, float to_full_y, bool encode_srgb);

GlowThreshold MakeGlowThreshold(const CosmicSettings& settings);

// Whether the working space is linear, for a render at the given depth.
bool IsLinearWorkingSpace(WorkingSpace space, bool float_depth);

}  // namespace cosmic
