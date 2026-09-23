#pragma once

#include <cstdint>

#include "Allocator.h"
#include "Color.h"
#include "Palette.h"
#include "SourceImage.h"
#include "TaskRunner.h"

namespace cosmic {

// The numeric values of these enums are the popup indices minus one, so they
// are part of the saved project format: append only.
enum class GradientType { kLinear = 0, kRadial = 1, kConic = 2, kDiamond = 3, kReflected = 4 };
enum class RepeatMode { kNone = 0, kRepeat = 1, kMirror = 2 };
enum class DepthShape { kDome = 0, kSphere = 1, kRidge = 2, kWave = 3 };
enum class MatteMode { kLayerAlpha = 0, kInvertedAlpha = 1, kFullFrame = 2 };
enum class BlendMode { kNormal = 0, kMultiply = 1, kScreen = 2, kOverlay = 3, kColor = 4 };
enum class WorkingSpace { kAuto = 0, kLinear = 1, kSrgb = 2 };
enum class FitMode { kContentBounds = 0, kLayer = 1 };

// Everything the render needs. Positions and absolute distances are in
// full-resolution square pixels of the layer (see CosmicRender for the mapping
// to the buffers), angles in radians with After Effects' convention: 0 points
// up, positive is clockwise.
//
// The gradient's own geometry is laid out on a reference box: the layer, or
// with FitMode::kContentBounds the bounding box of the layer's visible pixels,
// so a gradient applied to a line of text spans the text rather than the
// comp-sized layer around it. The point controls are given in the layer's
// frame and mapped proportionally onto that box; the lengths marked
// "relative" are fractions of it.
struct CosmicSettings {
    FitMode fit = FitMode::kContentBounds;
    float layer_width = 1000.0f;   // full-resolution size of the layer, the
    float layer_height = 1000.0f;  // frame the point controls live in

    // Palette, as linear-light colours from the first stop to the last.
    Rgb stops[kStopCount] = {};
    ColorBlend color_blend = ColorBlend::kOklabSmooth;
    bool reverse = false;

    // Gradient.
    GradientType type = GradientType::kLinear;
    float center_x = 0.0f;
    float center_y = 0.0f;
    float angle = 3.14159265f;  // the direction the palette runs, first stop to last
    float size = 1.0f;          // relative: 1 spans the box along the gradient
    float cycles = 1.0f;
    float offset = 0.0f;        // in palette lengths
    RepeatMode repeat = RepeatMode::kNone;

    // Depth field: a height map added to the gradient coordinate, which bends
    // the colour bands around it.
    DepthShape depth_shape = DepthShape::kDome;
    float depth = 0.0f;  // in palette lengths
    float depth_x = 0.0f;
    float depth_y = 0.0f;
    float depth_radius = 0.4f;  // relative, to the box's longer side

    // Turbulence: a looping fractal noise that displaces the field.
    float turbulence = 0.0f;        // relative, to the box's shorter side
    float turbulence_size = 0.4f;   // relative, to the box's shorter side
    float complexity = 3.0f;
    float evolution = 0.0f;  // radians; a full turn is a seamless loop
    std::uint32_t seed = 0;

    // Focus: sharp inside the radius, defocused beyond it.
    float focus_x = 0.0f;
    float focus_y = 0.0f;
    float focus_radius = 300.0f;
    float focus_falloff = 500.0f;
    float defocus = 0.0f;  // blur (sigma, px) reached at the end of the falloff

    // Glow, in linear light.
    float glow_intensity = 0.0f;
    float glow_radius = 100.0f;
    float glow_falloff = 1.6f;
    float glow_threshold = 0.35f;
    float glow_softness = 0.5f;
    float highlight_protection = 0.6f;

    // Optical diffusion: a soft veil of the image's own light.
    float diffusion = 0.0f;
    float diffusion_radius = 40.0f;

    // Grain.
    float grain = 0.0f;
    float grain_size = 1.0f;
    std::uint32_t grain_seed = 0;

    // Composite.
    MatteMode matte = MatteMode::kLayerAlpha;
    BlendMode blend = BlendMode::kNormal;
    float opacity = 1.0f;
    WorkingSpace working_space = WorkingSpace::kAuto;
};

// Where the buffers sit. Positions are in the layer's render-pixel grid (the
// coordinate system of After Effects' rects: layer origin at 0, scaled by the
// downsample factor).
struct CosmicRender {
    HostImage source;  // may be empty
    int source_left = 0;
    int source_top = 0;

    HostImage dest;
    int dest_left = 0;
    int dest_top = 0;

    // Render pixel -> full-resolution square pixel.
    float to_full_x = 1.0f;
    float to_full_y = 1.0f;
    // Full-resolution pixel -> render pixel, for blur sizes.
    float blur_scale = 1.0f;
};

enum class CosmicResult { kOk, kOutOfMemory, kInvalidArguments };

// How far, in render pixels, light can travel from the layer: what the output
// bounds must grow by so nothing is clipped.
float EffectReach(const CosmicSettings& settings, float blur_scale);

// True when the effect can put pixels outside the layer's own rectangle. With
// a full-frame or inverted matte the layer is already covered edge to edge.
inline bool CanExpand(const CosmicSettings& settings) {
    return settings.matte == MatteMode::kLayerAlpha;
}

CosmicResult RenderCosmic(const CosmicSettings& settings, const CosmicRender& render, Allocator& allocator,
                          TaskRunner& runner);

// The box the gradient is laid out on, in full-resolution square pixels.
struct ReferenceBox {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

// The reference box for a render: the layer, or the bounds of its visible
// pixels when the settings fit to content and the layer has any.
ReferenceBox FindReferenceBox(const CosmicSettings& settings, const CosmicRender& render);

// Exposed for tests: the gradient coordinate at a full-resolution position,
// before turbulence, after repeat.
float GradientCoordinate(const CosmicSettings& settings, const ReferenceBox& box, float x, float y);

}  // namespace cosmic
