#pragma once

// Parameter blocks for the CUDA kernels. Each kernel takes exactly one of
// these by value, so a launch is one pointer and the emulator in the tests can
// run the same kernels on the CPU. They hold only 4- and 8-byte scalars, so
// the host compiler and NVRTC lay them out identically.

#include "core/Shared.h"

namespace cosmic {

typedef unsigned long long DevicePtr;

constexpr int kGpuMaxLevels = 14;

// After Effects' GPU frames (PF_PixelFormat_GPU_BGRA128): premultiplied float
// B, G, R, A. The kernels' own images are PixelF (A, R, G, B), tightly packed.
struct FrameBGRA {
    float b, g, r, a;
};

struct BoundsParams {
    DevicePtr source;  // FrameBGRA
    DevicePtr out;     // int[4]: x0, x1, y0, y1
    int width;
    int height;
    int pitch;
    float visible;
};

struct WarpGridParams {
    DevicePtr out;  // float pairs, nx * ny
    int nx;
    int ny;
    int step;
    int canvas_left;
    int canvas_top;
    float to_full_x;
    float to_full_y;
    float inv_size;
    float amount;
    float evolution;
    FbmSettings fbm;
    FbmSettings fbm2;
};

struct BaseParams {
    DevicePtr source;  // FrameBGRA, 0 when there is no layer
    DevicePtr out;     // PixelF, width x height
    DevicePtr lut;     // Rgb[kLutSize]
    DevicePtr warp;    // WarpGridParams::out, 0 when inactive
    int source_width;
    int source_height;
    int source_pitch;
    int source_left;
    int source_top;
    int width;
    int height;
    int canvas_left;
    int canvas_top;
    int warp_nx;
    int warp_ny;
    int warp_step;
    int decode_srgb;
    int matte;  // MatteMode
    int blend;  // BlendMode
    float opacity;
    float to_full_x;
    float to_full_y;
    Field field;
};

// One half of a pyramid reduce: horizontal (src_width x src_height into
// dst_width x src_height) or vertical (dst_width x src_height into
// dst_width x dst_height).
struct ReduceParams {
    DevicePtr src;  // PixelF
    DevicePtr dst;  // PixelF
    int src_width;
    int src_height;
    int dst_width;
    int dst_height;
    int border;   // BorderMode
    int extract;  // isolate highlights while reading (the glow's level 0)
    GlowThreshold threshold;
};

struct FocusParams {
    DevicePtr base;  // PixelF, width x height
    DevicePtr out;   // PixelF, width x height
    DevicePtr levels[kGpuMaxLevels];  // pyramid levels 1..top
    int level_width[kGpuMaxLevels];
    int level_height[kGpuMaxLevels];
    float sigmas[kGpuMaxLevels + 2];  // PyramidLevelSigma(0..top), top repeated
    int width;
    int height;
    int canvas_left;
    int canvas_top;
    int top;
    int border;
    float to_full_x;
    float to_full_y;
    float focus_x;
    float focus_y;
    float focus_radius;
    float falloff;
    float defocus;
    float unused;
};

// One step of collapsing a pyramid: out = weight * level + upsampled coarse.
struct CollapseParams {
    DevicePtr coarse;  // PixelF accumulator of the next coarser level, 0 at the top
    DevicePtr level;   // PixelF, this level
    DevicePtr out;     // PixelF, this level's size
    int coarse_width;
    int coarse_height;
    int width;
    int height;
    int border;
    float weight;
};

struct CompositeParams {
    DevicePtr image;      // PixelF, width x height
    DevicePtr glow;       // PixelF at level 1, or 0
    DevicePtr diffusion;  // PixelF at level 1, or 0
    DevicePtr dest;       // FrameBGRA
    int width;
    int height;
    int dest_pitch;
    int dest_left;
    int dest_top;
    int glow_width;
    int glow_height;
    int diffusion_width;
    int diffusion_height;
    int border;
    float intensity;
    float diffusion_amount;
    FinishParams finish;
};

}  // namespace cosmic
