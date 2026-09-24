#include "gpu/GpuPipeline.h"

#include <algorithm>
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
        case Kernel::kRowStats: return "CosmicRowStats";
        case Kernel::kWarpGrid: return "CosmicWarpGrid";
        case Kernel::kShape: return "CosmicShape";
        case Kernel::kSmooth: return "CosmicSmooth";
        case Kernel::kRelief: return "CosmicRelief";
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

    // The layer's shape: content bounds and the relief's scale.
    ShapeMeasure shape;
    if (WantsShapeMeasure(settings) && has_source) {
        const int rows = render.source.height;
        const DevicePtr stats = buffers.Allocate(sizeof(RowStats) * static_cast<std::size_t>(rows));
        if (stats == 0) return CosmicResult::kOutOfMemory;
        RowStatsParams p;
        p.source = render.source.data;
        p.out = stats;
        p.width = render.source.width;
        p.height = rows;
        p.pitch = render.source.pitch;
        p.measure_outline = settings.bulge > 0.0f ? 1 : 0;
        p.visible = kVisibleAlpha;
        p.to_full_x = render.to_full_x;
        p.to_full_y = render.to_full_y;
        p.unused = 0.0f;
        if (!Run(device, Kernel::kRowStats, rows, 1, p)) return CosmicResult::kDeviceError;
        // Summed on the host in row order, like the CPU renderer does.
        std::vector<RowStats> found(static_cast<std::size_t>(rows));
        if (!device.Finish() || !device.Download(found.data(), stats, sizeof(RowStats) * found.size())) {
            return CosmicResult::kDeviceError;
        }
        buffers.Release(stats);
        shape = MeasureShape(found.data(), rows);
    }
    const ReferenceBox box =
        BoxFromShape(settings, shape, render.source_left, render.source_top, render.to_full_x, render.to_full_y);

    // Turbulence grid.
    const WarpPlan warp =
        MakeWarpPlan(settings, box, render.to_full_x, render.to_full_y, canvas_left, canvas_top, width, height);
    DevicePtr warp_buffer = 0;
    if (warp.active) {
        warp_buffer = buffers.Allocate(sizeof(float) * 2 * static_cast<std::size_t>(warp.lattice.nx) *
                                       static_cast<std::size_t>(warp.lattice.ny));
        if (warp_buffer == 0) return CosmicResult::kOutOfMemory;
        WarpGridParams p;
        p.out = warp_buffer;
        p.lattice = warp.lattice;
        p.amount = warp.amount;
        p.evolution = warp.evolution;
        p.fbm = warp.fbm;
        p.fbm2 = warp.fbm2;
        if (!Run(device, Kernel::kWarpGrid, warp.lattice.nx, warp.lattice.ny, p)) return CosmicResult::kDeviceError;
    }

    // 0. The Bulge relief. Level 0 of its pyramid is the shape; each level is
    // filtered from the one before into `level`, except level lo, which goes
    // to `kept` when level lo + 1 is needed as well.
    DevicePtr relief = 0;
    ReliefGeometry geometry;
    geometry.canvas_left = canvas_left;
    geometry.canvas_top = canvas_top;
    geometry.width = width;
    geometry.height = height;
    geometry.source_left = render.source_left;
    geometry.source_top = render.source_top;
    geometry.source_width = has_source ? render.source.width : 0;
    geometry.source_height = has_source ? render.source.height : 0;
    geometry.to_full_x = render.to_full_x;
    geometry.to_full_y = render.to_full_y;
    geometry.blur_scale = render.blur_scale;
    const ReliefPlan relief_plan = MakeReliefPlan(settings, box, shape, geometry);
    if (relief_plan.active && has_source) {
        const int dw = relief_plan.domain_width;
        const int dh = relief_plan.domain_height;
        const std::size_t plane = sizeof(float) * static_cast<std::size_t>(dw) * static_cast<std::size_t>(dh);
        const bool two = relief_plan.hi > relief_plan.lo;
        const DevicePtr level = buffers.Allocate(plane);
        const DevicePtr tmp = buffers.Allocate(plane);
        const DevicePtr kept = two ? buffers.Allocate(plane) : 0;
        if (level == 0 || tmp == 0 || (two && kept == 0)) return CosmicResult::kOutOfMemory;

        ShapeParams sp;
        sp.source = render.source.data;
        sp.out = level;
        sp.source_width = render.source.width;
        sp.source_height = render.source.height;
        sp.source_pitch = render.source.pitch;
        sp.source_left = render.source_left;
        sp.source_top = render.source_top;
        sp.width = dw;
        sp.height = dh;
        sp.canvas_left = canvas_left + relief_plan.domain_left;
        sp.canvas_top = canvas_top + relief_plan.domain_top;
        sp.invert = relief_plan.invert;
        if (!Run(device, Kernel::kShape, dw, dh, sp)) return CosmicResult::kDeviceError;

        DevicePtr src = level;
        for (int k = 1; k <= relief_plan.hi; ++k) {
            SmoothParams p;
            p.src = src;
            p.dst = tmp;
            p.width = dw;
            p.height = dh;
            p.step = SmoothLevelStep(k);
            p.vertical = 0;
            p.border = static_cast<int>(border);
            p.unused = 0;
            if (!Run(device, Kernel::kSmooth, dw, dh, p)) return CosmicResult::kDeviceError;
            p.src = tmp;
            p.dst = (k == relief_plan.lo && two) ? kept : level;
            p.vertical = 1;
            if (!Run(device, Kernel::kSmooth, dw, dh, p)) return CosmicResult::kDeviceError;
            src = p.dst;
        }
        buffers.Release(tmp);

        relief = buffers.Images(width, height);
        if (relief == 0) return CosmicResult::kOutOfMemory;
        ReliefParams rp;
        rp.lo = two ? kept : level;
        rp.hi = level;
        rp.out = relief;
        rp.width = width;
        rp.height = height;
        rp.domain_left = relief_plan.domain_left;
        rp.domain_top = relief_plan.domain_top;
        rp.domain_width = dw;
        rp.domain_height = dh;
        rp.mix = relief_plan.mix;
        rp.unused = 0.0f;
        rp.shape = relief_plan.shape;
        if (!Run(device, Kernel::kRelief, width, height, rp)) return CosmicResult::kDeviceError;
        buffers.Release(level);
        buffers.Release(kept);
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
        p.relief = relief;
        p.source_width = render.source.width;
        p.source_height = render.source.height;
        p.source_pitch = render.source.pitch;
        p.source_left = render.source_left;
        p.source_top = render.source_top;
        p.width = width;
        p.height = height;
        p.canvas_left = canvas_left;
        p.canvas_top = canvas_top;
        p.decode_srgb = linear ? 0 : 1;
        p.matte = static_cast<int>(settings.matte);
        p.blend = static_cast<int>(settings.blend);
        p.opacity = std::clamp(settings.opacity, 0.0f, 1.0f);
        p.to_full_x = render.to_full_x;
        p.to_full_y = render.to_full_y;
        p.lattice = warp.lattice;
        p.field = MakeField(settings, box);
        if (!Run(device, Kernel::kBase, width, height, p)) return CosmicResult::kDeviceError;
    }
    buffers.Release(warp_buffer);
    buffers.Release(relief);

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
