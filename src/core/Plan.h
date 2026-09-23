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

// The reference box from the visible bounds of the source, given as inclusive
// pixel indices of the source buffer; x1 < x0 means nothing is visible.
ReferenceBox BoxFromSourceBounds(const CosmicSettings& settings, int x0, int x1, int y0, int y1, int source_left,
                                 int source_top, float to_full_x, float to_full_y);

// True when the settings want the reference box from the layer's pixels.
bool WantsContentBounds(const CosmicSettings& settings);

// Alpha above which a pixel counts towards the content bounds.
constexpr float kVisibleAlpha = 0.5f / 255.0f;

// Turbulence grid: nodes every `step` render pixels from two before the canvas
// to past its far edge, each holding a displacement in full-resolution pixels.
struct WarpPlan {
    bool active = false;
    int step = 1;
    int nx = 0;
    int ny = 0;
    float inv_size = 1.0f;
    float amount = 0.0f;
    float evolution = 0.0f;
    FbmSettings fbm;
    FbmSettings fbm2;
};

WarpPlan MakeWarpPlan(const CosmicSettings& settings, const ReferenceBox& box, float blur_scale, int width,
                      int height);

FinishParams MakeFinishParams(const CosmicSettings& settings, float to_full_x, float to_full_y, bool encode_srgb);

GlowThreshold MakeGlowThreshold(const CosmicSettings& settings);

// Whether the working space is linear, for a render at the given depth.
bool IsLinearWorkingSpace(WorkingSpace space, bool float_depth);

}  // namespace cosmic
