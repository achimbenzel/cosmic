#include "gpu/GpuPipeline.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <vector>

#include "core/Palette.h"
#include "core/Plan.h"
#include "core/Pyramid.h"

namespace cosmic {
namespace {

static_assert(kGpuMaxLevels == kMaxPyramidLevels, "the GPU and CPU pyramids must be equally deep");

constexpr std::size_t kPixelBytes = sizeof(PixelF);

// Owns the render's device allocations. Kernels run asynchronously, so memory
// is only handed back once the device has finished with it.
class DeviceBuffers {
public:
    explicit DeviceBuffers(GpuDevice& device) : device_(device) {}

    ~DeviceBuffers() {
        device_.Finish();
        for (DevicePtr ptr : live_) device_.Free(ptr);
    }

    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;

    // 0 when the device is out of memory.
    DevicePtr Allocate(std::size_t bytes) {
        DevicePtr ptr = 0;
        if (bytes == 0 || !device_.Allocate(bytes, &ptr) || ptr == 0) return 0;
        live_.push_back(ptr);
        return ptr;
    }

    DevicePtr Images(int width, int height) {
        return Allocate(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * kPixelBytes);
    }

    // Frees early, to keep the peak down on large frames.
    void Release(DevicePtr ptr) {
        if (ptr == 0) return;
        auto it = std::find(live_.begin(), live_.end(), ptr);
        if (it == live_.end()) return;
        device_.Finish();
        device_.Free(ptr);
        live_.erase(it);
    }

private:
    GpuDevice& device_;
    std::vector<DevicePtr> live_;
};

template <typename Params>
bool Run(GpuDevice& device, Kernel kernel, int width, int height, const Params& params) {
    if (width <= 0 || height <= 0) return true;
    return device.Launch(kernel, width, height, &params, sizeof(Params));
}

struct GpuPyramid {
    DevicePtr level[kGpuMaxLevels] = {};
    int width[kGpuMaxLevels] = {};
    int height[kGpuMaxLevels] = {};
    int count = 0;
};

enum class Status { kOk, kOutOfMemory, kDeviceError };

// Mirrors Pyramid::Build: each level is a horizontal then a vertical six-tap
// reduce of the one before, the first optionally isolating highlights.
Status BuildPyramid(GpuDevice& device, DeviceBuffers& buffers, DevicePtr level0, int width, int height, int levels,
                    BorderMode border, bool extract, const GlowThreshold& threshold, GpuPyramid* out) {
    out->count = 0;
    levels = std::min(levels, kGpuMaxLevels);
    if (levels <= 0) return Status::kOk;
    const DevicePtr tmp = buffers.Images((width + 1) / 2, height);
    if (tmp == 0) return Status::kOutOfMemory;

    DevicePtr src = level0;
    int w = width;
    int h = height;
    for (int k = 1; k <= levels; ++k) {
        const int hw = (w + 1) / 2;
        const int hh = (h + 1) / 2;
        const DevicePtr dst = buffers.Images(hw, hh);
        if (dst == 0) return Status::kOutOfMemory;

        ReduceParams p;
        p.src = src;
        p.dst = tmp;
        p.src_width = w;
        p.src_height = h;
        p.dst_width = hw;
        p.dst_height = hh;
        p.border = static_cast<int>(border);
        p.extract = (extract && k == 1) ? 1 : 0;
        p.threshold = threshold;
        if (!Run(device, Kernel::kReduceH, hw, h, p)) return Status::kDeviceError;
        p.src = tmp;
        p.dst = dst;
        p.extract = 0;
        if (!Run(device, Kernel::kReduceV, hw, hh, p)) return Status::kDeviceError;

        out->level[k - 1] = dst;
        out->width[k - 1] = hw;
        out->height[k - 1] = hh;
        out->count = k;
        src = dst;
        w = hw;
        h = hh;
    }
    buffers.Release(tmp);
    return Status::kOk;
}

void ReleasePyramid(DeviceBuffers& buffers, GpuPyramid* pyramid) {
    for (int k = 0; k < pyramid->count; ++k) buffers.Release(pyramid->level[k]);
    pyramid->count = 0;
}

// Mirrors CollapsePyramid for levels 1..last: the result is on level 1's grid.
Status Collapse(GpuDevice& device, DeviceBuffers& buffers, const GpuPyramid& pyramid, const float* weights, int last,
                BorderMode border, DevicePtr* out) {
    *out = 0;
    DevicePtr acc = buffers.Images(pyramid.width[last - 1], pyramid.height[last - 1]);
    if (acc == 0) return Status::kOutOfMemory;
    CollapseParams p;
    p.coarse = 0;
    p.level = pyramid.level[last - 1];
    p.out = acc;
    p.coarse_width = 0;
    p.coarse_height = 0;
    p.width = pyramid.width[last - 1];
    p.height = pyramid.height[last - 1];
    p.border = static_cast<int>(border);
    p.weight = weights[last];
    if (!Run(device, Kernel::kCollapse, p.width, p.height, p)) return Status::kDeviceError;

    for (int k = last - 1; k >= 1; --k) {
        const DevicePtr next = buffers.Images(pyramid.width[k - 1], pyramid.height[k - 1]);
        if (next == 0) return Status::kOutOfMemory;
        p.coarse = acc;
        p.coarse_width = pyramid.width[k];
        p.coarse_height = pyramid.height[k];
        p.level = pyramid.level[k - 1];
        p.out = next;
        p.width = pyramid.width[k - 1];
        p.height = pyramid.height[k - 1];
        p.weight = weights[k];
        if (!Run(device, Kernel::kCollapse, p.width, p.height, p)) return Status::kDeviceError;
        buffers.Release(acc);
        acc = next;
    }
    *out = acc;
    return Status::kOk;
}

CosmicResult ToResult(Status status) {
    switch (status) {
        case Status::kOk: return CosmicResult::kOk;
        case Status::kOutOfMemory: return CosmicResult::kOutOfMemory;
        case Status::kDeviceError:
        default: return CosmicResult::kDeviceError;
    }
}

}  // namespace

const char* KernelName(Kernel kernel) {
    switch (kernel) {
        case Kernel::kBounds: return "CosmicBounds";
        case Kernel::kWarpGrid: return "CosmicWarpGrid";
        case Kernel::kBase: return "CosmicBase";
        case Kernel::kReduceH: return "CosmicReduceH";
        case Kernel::kReduceV: return "CosmicReduceV";
        case Kernel::kFocus: return "CosmicFocus";
        case Kernel::kCollapse: return "CosmicCollapse";
        case Kernel::kComposite: return "CosmicComposite";
        case Kernel::kCount:
        default: return "";
    }
}

CosmicResult RenderCosmicGpu(const CosmicSettings& settings, const GpuRender& render, GpuDevice& device) {
    const GpuFrame& dest = render.dest;
    if (dest.data == 0 || dest.width <= 0 || dest.height <= 0 || dest.pitch < dest.width) {
        return CosmicResult::kInvalidArguments;
    }
    if (!(render.to_full_x > 0.0f) || !(render.to_full_y > 0.0f) || !(render.blur_scale > 0.0f)) {
        return CosmicResult::kInvalidArguments;
    }
    const bool has_source = render.source.data != 0 && render.source.width > 0 && render.source.height > 0;

    DeviceBuffers buffers(device);
    const int width = dest.width;
    const int height = dest.height;
    const int canvas_left = render.dest_left;
    const int canvas_top = render.dest_top;
    const BorderMode border = CanExpand(settings) ? BorderMode::kZero : BorderMode::kClamp;
    const bool linear = IsLinearWorkingSpace(settings.working_space, render.float_project);

    // Palette.
    GradientLut lut;
    lut.Build(settings.stops, settings.color_blend, settings.reverse);
    const DevicePtr lut_buffer = buffers.Allocate(sizeof(Rgb) * kLutSize);
    if (lut_buffer == 0) return CosmicResult::kOutOfMemory;
    if (!device.Upload(lut_buffer, lut.Table(), sizeof(Rgb) * kLutSize)) return CosmicResult::kDeviceError;

    // Reference box.
    ReferenceBox box = BoxFromSourceBounds(settings, 0, -1, 0, -1, 0, 0, 1.0f, 1.0f);
    if (WantsContentBounds(settings) && has_source) {
        const DevicePtr bounds = buffers.Allocate(sizeof(int) * 4);
        if (bounds == 0) return CosmicResult::kOutOfMemory;
        int init[4] = {INT_MAX, -1, INT_MAX, -1};
        if (!device.Upload(bounds, init, sizeof(init))) return CosmicResult::kDeviceError;
        BoundsParams p;
        p.source = render.source.data;
        p.out = bounds;
        p.width = render.source.width;
        p.height = render.source.height;
        p.pitch = render.source.pitch;
        p.visible = kVisibleAlpha;
        if (!Run(device, Kernel::kBounds, render.source.height, 1, p)) return CosmicResult::kDeviceError;
        int found[4] = {0, -1, 0, -1};
        if (!device.Finish() || !device.Download(found, bounds, sizeof(found))) return CosmicResult::kDeviceError;
        box = BoxFromSourceBounds(settings, found[0], found[1], found[2], found[3], render.source_left,
                                  render.source_top, render.to_full_x, render.to_full_y);
    }

    // Turbulence grid.
    const WarpPlan warp = MakeWarpPlan(settings, box, render.blur_scale, width, height);
    DevicePtr warp_buffer = 0;
    if (warp.active) {
        warp_buffer = buffers.Allocate(sizeof(float) * 2 * static_cast<std::size_t>(warp.nx) * warp.ny);
        if (warp_buffer == 0) return CosmicResult::kOutOfMemory;
        WarpGridParams p;
        p.out = warp_buffer;
        p.nx = warp.nx;
        p.ny = warp.ny;
        p.step = warp.step;
        p.canvas_left = canvas_left;
        p.canvas_top = canvas_top;
        p.to_full_x = render.to_full_x;
        p.to_full_y = render.to_full_y;
        p.inv_size = warp.inv_size;
        p.amount = warp.amount;
        p.evolution = warp.evolution;
        p.fbm = warp.fbm;
        p.fbm2 = warp.fbm2;
        if (!Run(device, Kernel::kWarpGrid, warp.nx, warp.ny, p)) return CosmicResult::kDeviceError;
    }

    // 1. The gradient, matted and blended with the layer.
    const DevicePtr base = buffers.Images(width, height);
    if (base == 0) return CosmicResult::kOutOfMemory;
    {
        BaseParams p;
        p.source = has_source ? render.source.data : 0;
        p.out = base;
        p.lut = lut_buffer;
        p.warp = warp_buffer;
        p.source_width = render.source.width;
        p.source_height = render.source.height;
        p.source_pitch = render.source.pitch;
        p.source_left = render.source_left;
        p.source_top = render.source_top;
        p.width = width;
        p.height = height;
        p.canvas_left = canvas_left;
        p.canvas_top = canvas_top;
        p.warp_nx = warp.nx;
        p.warp_ny = warp.ny;
        p.warp_step = warp.step;
        p.decode_srgb = linear ? 0 : 1;
        p.matte = static_cast<int>(settings.matte);
        p.blend = static_cast<int>(settings.blend);
        p.opacity = std::clamp(settings.opacity, 0.0f, 1.0f);
        p.to_full_x = render.to_full_x;
        p.to_full_y = render.to_full_y;
        p.field = MakeField(settings, box);
        if (!Run(device, Kernel::kBase, width, height, p)) return CosmicResult::kDeviceError;
    }
    buffers.Release(warp_buffer);

    // 2. Focus.
    DevicePtr image = base;
    const float defocus = settings.defocus * render.blur_scale;
    if (defocus > 0.05f) {
        GpuPyramid pyramid;
        const int levels = std::min(LevelsForSigma(defocus), MaxUsefulLevels(width, height));
        Status status = BuildPyramid(device, buffers, base, width, height, levels, border, false, GlowThreshold(),
                                     &pyramid);
        if (status != Status::kOk) return ToResult(status);
        const DevicePtr focused = buffers.Images(width, height);
        if (focused == 0) return CosmicResult::kOutOfMemory;
        FocusParams p;
        p.base = base;
        p.out = focused;
        for (int k = 0; k < kGpuMaxLevels; ++k) {
            p.levels[k] = k < pyramid.count ? pyramid.level[k] : 0;
            p.level_width[k] = k < pyramid.count ? pyramid.width[k] : 0;
            p.level_height[k] = k < pyramid.count ? pyramid.height[k] : 0;
        }
        const int top = pyramid.count;
        for (int k = 0; k < kGpuMaxLevels + 2; ++k) p.sigmas[k] = PyramidLevelSigma(std::min(k, top));
        p.width = width;
        p.height = height;
        p.canvas_left = canvas_left;
        p.canvas_top = canvas_top;
        p.top = top;
        p.border = static_cast<int>(border);
        p.to_full_x = render.to_full_x;
        p.to_full_y = render.to_full_y;
        p.focus_x = settings.focus_x;
        p.focus_y = settings.focus_y;
        p.focus_radius = settings.focus_radius;
        p.falloff = std::max(0.0f, settings.focus_falloff);
        p.defocus = defocus;
        p.unused = 0.0f;
        if (!Run(device, Kernel::kFocus, width, height, p)) return CosmicResult::kDeviceError;
        ReleasePyramid(buffers, &pyramid);
        buffers.Release(base);
        image = focused;
    }

    // 3. Glow and diffusion, each a collapsed pyramid on level 1's grid.
    const int max_levels = MaxUsefulLevels(width, height);
    DevicePtr glow = 0;
    int glow_width = 0;
    int glow_height = 0;
    if (settings.glow_intensity > 0.0f && settings.glow_radius > 0.0f) {
        const BlurPlan plan =
            MakeBlurPlan(settings.glow_radius * render.blur_scale * 0.5f, settings.glow_falloff, max_levels);
        if (plan.levels > 0) {
            GpuPyramid pyramid;
            Status status = BuildPyramid(device, buffers, image, width, height, plan.levels, border, true,
                                         MakeGlowThreshold(settings), &pyramid);
            if (status == Status::kOk) status = Collapse(device, buffers, pyramid, plan.weights, plan.levels, border, &glow);
            if (status != Status::kOk) return ToResult(status);
            glow_width = pyramid.width[0];
            glow_height = pyramid.height[0];
            ReleasePyramid(buffers, &pyramid);
        }
    }

    DevicePtr diffusion = 0;
    int diffusion_width = 0;
    int diffusion_height = 0;
    const float diffusion_amount = std::clamp(settings.diffusion, 0.0f, 1.0f);
    if (diffusion_amount > 0.0f && settings.diffusion_radius > 0.0f) {
        const BlurPlan plan =
            MakeBlurPlan(settings.diffusion_radius * render.blur_scale * 0.5f, kDiffusionFalloff, max_levels);
        if (plan.levels > 0) {
            GpuPyramid pyramid;
            Status status = BuildPyramid(device, buffers, image, width, height, plan.levels, border, false,
                                         GlowThreshold(), &pyramid);
            if (status == Status::kOk) {
                status = Collapse(device, buffers, pyramid, plan.weights, plan.levels, border, &diffusion);
            }
            if (status != Status::kOk) return ToResult(status);
            diffusion_width = pyramid.width[0];
            diffusion_height = pyramid.height[0];
            ReleasePyramid(buffers, &pyramid);
        }
    }

    // 4. Composite into the host's frame.
    CompositeParams p;
    p.image = image;
    p.glow = glow;
    p.diffusion = diffusion;
    p.dest = dest.data;
    p.width = width;
    p.height = height;
    p.dest_pitch = dest.pitch;
    p.dest_left = render.dest_left;
    p.dest_top = render.dest_top;
    p.glow_width = glow_width;
    p.glow_height = glow_height;
    p.diffusion_width = diffusion_width;
    p.diffusion_height = diffusion_height;
    p.border = static_cast<int>(border);
    p.intensity = std::max(0.0f, settings.glow_intensity);
    p.diffusion_amount = diffusion_amount;
    p.finish = MakeFinishParams(settings, render.to_full_x, render.to_full_y, !linear);
    if (!Run(device, Kernel::kComposite, width, height, p)) return CosmicResult::kDeviceError;
    if (!device.Finish()) return CosmicResult::kDeviceError;
    return CosmicResult::kOk;
}

}  // namespace cosmic
