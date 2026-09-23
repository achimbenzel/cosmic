// The GPU renderer against the CPU renderer: the CUDA kernels run on the CPU
// through the emulator, driven by the same GpuPipeline the plug-in uses, and
// must produce the CPU's frame to within rounding.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "GpuEmulator.h"
#include "TestSupport.h"
#include "core/CosmicPipeline.h"
#include "core/UiModel.h"
#include "gpu/GpuPipeline.h"

using cosmic::FrameBGRA;
using cosmic::PixelDepth;
using cosmic::PixelF;
using cosmic_test::TestImage;

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool condition, const std::string& what) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("  FAIL: %s\n", what.c_str());
    }
}

TestImage MakeLayer(int w, int h) {
    TestImage image(w, h, PixelDepth::kFloat32);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float d = std::hypot(x + 0.5f - w * 0.35f, y + 0.5f - h * 0.5f);
            float a = std::clamp(h * 0.3f - d + 0.5f, 0.0f, 1.0f);
            const bool bar = x > w * 0.6f && x < w * 0.92f && y > h * 0.3f && y < h * 0.7f;
            if (bar) a = 0.8f;  // partial coverage, to exercise unpremultiplying
            // A coloured layer, so blend modes have something to work with.
            image.SetPixel(x, y, PixelF{a, a * 0.9f, a * (0.3f + 0.5f * x / w), a * 0.6f});
        }
    }
    return image;
}

struct Case {
    std::string name;
    std::function<void(cosmic::UiValues&)> tweak;
    int downsample = 1;
};

// Renders on both paths and returns the largest channel difference.
double Compare(const Case& c, const TestImage& layer_full, int w, int h) {
    cosmic::UiValues ui;
    cosmic::PlaceDefaultPoints(&ui, static_cast<float>(w), static_cast<float>(h));
    c.tweak(ui);
    const int ds = c.downsample;
    const cosmic::CosmicSettings settings = cosmic::SettingsFromUi(ui, static_cast<float>(w), static_cast<float>(h), 3);
    const float blur_scale = 1.0f / ds;
    const int lw = w / ds;
    const int lh = h / ds;
    TestImage layer(lw, lh, PixelDepth::kFloat32);
    for (int y = 0; y < lh; ++y)
        for (int x = 0; x < lw; ++x) layer.SetPixel(x, y, layer_full.GetPixel(x * ds, y * ds));

    const int expansion = (ui.expand_bounds && cosmic::CanExpand(settings))
                              ? static_cast<int>(std::ceil(cosmic::EffectReach(settings, blur_scale)))
                              : 0;
    const int ow = lw + 2 * expansion;
    const int oh = lh + 2 * expansion;

    // CPU.
    TestImage cpu(ow, oh, PixelDepth::kFloat32);
    cosmic::CosmicRender r;
    r.source = layer.View();
    r.dest = cpu.View();
    r.dest_left = -expansion;
    r.dest_top = -expansion;
    r.to_full_x = static_cast<float>(ds);
    r.to_full_y = static_cast<float>(ds);
    r.blur_scale = blur_scale;
    cosmic_test::MallocAllocator allocator;
    cosmic_test::ThreadPoolRunner runner(4);
    Check(cosmic::RenderCosmic(settings, r, allocator, runner) == cosmic::CosmicResult::kOk, c.name + ": CPU renders");

    // GPU, through the emulator, with padded pitches like After Effects' frames.
    cosmic_test::EmulatedDevice device;
    const int src_pitch = lw + 5;
    const int dst_pitch = ow + 3;
    std::vector<FrameBGRA> src(static_cast<std::size_t>(src_pitch) * lh);
    for (int y = 0; y < lh; ++y) {
        for (int x = 0; x < lw; ++x) {
            const PixelF p = layer.GetPixel(x, y);
            src[static_cast<std::size_t>(y) * src_pitch + x] = FrameBGRA{p.b, p.g, p.r, p.a};
        }
    }
    cosmic::DevicePtr src_ptr = 0;
    cosmic::DevicePtr dst_ptr = 0;
    device.Allocate(src.size() * sizeof(FrameBGRA), &src_ptr);
    device.Allocate(static_cast<std::size_t>(dst_pitch) * oh * sizeof(FrameBGRA), &dst_ptr);
    device.Upload(src_ptr, src.data(), src.size() * sizeof(FrameBGRA));

    cosmic::GpuRender g;
    g.source = cosmic::GpuFrame{src_ptr, lw, lh, src_pitch};
    g.dest = cosmic::GpuFrame{dst_ptr, ow, oh, dst_pitch};
    g.dest_left = -expansion;
    g.dest_top = -expansion;
    g.to_full_x = static_cast<float>(ds);
    g.to_full_y = static_cast<float>(ds);
    g.blur_scale = blur_scale;
    g.float_project = true;
    Check(cosmic::RenderCosmicGpu(settings, g, device) == cosmic::CosmicResult::kOk, c.name + ": GPU renders");

    std::vector<FrameBGRA> dst(static_cast<std::size_t>(dst_pitch) * oh);
    device.Download(dst.data(), dst_ptr, dst.size() * sizeof(FrameBGRA));
    device.Free(src_ptr);
    device.Free(dst_ptr);
    Check(device.allocations() == device.frees(), c.name + ": every device buffer is freed");

    double worst = 0.0;
    bool finite = true;
    for (int y = 0; y < oh; ++y) {
        for (int x = 0; x < ow; ++x) {
            const PixelF p = cpu.GetPixel(x, y);
            const FrameBGRA& q = dst[static_cast<std::size_t>(y) * dst_pitch + x];
            if (!std::isfinite(q.a) || !std::isfinite(q.r) || !std::isfinite(q.g) || !std::isfinite(q.b)) finite = false;
            worst = std::max({worst, (double)std::fabs(p.a - q.a), (double)std::fabs(p.r - q.r),
                              (double)std::fabs(p.g - q.g), (double)std::fabs(p.b - q.b)});
        }
    }
    Check(finite, c.name + ": no NaN (nothing read uninitialised device memory)");
    return worst;
}

}  // namespace

int main() {
    const int w = 240;
    const int h = 150;
    const TestImage layer = MakeLayer(w, h);

    std::vector<Case> cases = {
        {"defaults", [](cosmic::UiValues&) {}},
        {"linear_ws", [](cosmic::UiValues& u) { u.working_space = 2; }},
        {"srgb_ws", [](cosmic::UiValues& u) { u.working_space = 3; }},
        {"turbulence", [](cosmic::UiValues& u) { u.turbulence_pct = 20; u.complexity = 4.5f; u.evolution_deg = 40; }},
        {"fine_turbulence", [](cosmic::UiValues& u) { u.turbulence_pct = 10; u.turbulence_size_pct = 5; u.complexity = 6; }},
        {"radial_sphere", [](cosmic::UiValues& u) { u.gradient_type = 2; u.depth_shape = 2; u.depth_pct = 60; }},
        {"conic_mirror", [](cosmic::UiValues& u) { u.gradient_type = 3; u.repeat = 3; u.cycles = 3; }},
        {"diamond_ridge", [](cosmic::UiValues& u) { u.gradient_type = 4; u.depth_shape = 3; }},
        {"reflected_wave", [](cosmic::UiValues& u) { u.gradient_type = 5; u.depth_shape = 4; u.depth_pct = 10; }},
        {"bulge", [](cosmic::UiValues& u) { u.depth_shape = 5; u.depth_pct = 70; u.cycles = 3; u.repeat = 2; }},
        {"pinch", [](cosmic::UiValues& u) { u.depth_shape = 5; u.depth_pct = -90; }},
        {"focus", [](cosmic::UiValues& u) { u.defocus_px = 14; u.focus_radius_pct = 10; }},
        {"focus_hard", [](cosmic::UiValues& u) { u.defocus_px = 40; u.focus_radius_pct = 5; u.focus_falloff_pct = 0; }},
        {"hot_glow", [](cosmic::UiValues& u) { u.glow_intensity_pct = 300; u.glow_threshold = 0.1f; u.protection_pct = 80; }},
        {"wide_glow", [](cosmic::UiValues& u) { u.glow_radius_px = 300; u.glow_falloff = 1.0f; }},
        {"no_glow_no_veil", [](cosmic::UiValues& u) { u.glow_intensity_pct = 0; u.diffusion_pct = 0; }},
        {"multiply", [](cosmic::UiValues& u) { u.blend = 2; }},
        {"screen", [](cosmic::UiValues& u) { u.blend = 3; u.opacity_pct = 60; }},
        {"overlay", [](cosmic::UiValues& u) { u.blend = 4; }},
        {"color", [](cosmic::UiValues& u) { u.blend = 5; u.opacity_pct = 70; }},
        {"grain", [](cosmic::UiValues& u) { u.grain_pct = 30; u.grain_size_px = 2; u.animate_grain = true; }},
        {"layer_fit", [](cosmic::UiValues& u) { u.fit = 2; u.angle_deg = 120; }},
        {"inverted", [](cosmic::UiValues& u) { u.matte = 2; u.defocus_px = 10; }},
        {"full_frame", [](cosmic::UiValues& u) { u.matte = 3; u.turbulence_pct = 10; u.defocus_px = 20; u.focus_radius_pct = 0; }},
        {"no_expand", [](cosmic::UiValues& u) { u.expand_bounds = false; }},
        {"half_res", [](cosmic::UiValues& u) { u.defocus_px = 12; u.turbulence_pct = 8; }, 2},
        {"everything", [](cosmic::UiValues& u) {
             u.turbulence_pct = 12; u.depth_shape = 5; u.defocus_px = 20; u.grain_pct = 10; u.blend = 5;
             u.glow_intensity_pct = 150; u.diffusion_pct = 60; u.working_space = 3; }},
    };

    std::printf("GPU kernels (emulated) against the CPU renderer\n");
    for (const Case& c : cases) {
        const double diff = Compare(c, layer, w, h);
        // Linear light with no sRGB round trips agrees to float rounding; the
        // CPU's sRGB lookup tables differ from the GPU's exact curve by far
        // less than an 8-bit step.
        const double tolerance = 2.0e-3;
        char line[160];
        std::snprintf(line, sizeof(line), "%-16s max difference %.2e", c.name.c_str(), diff);
        std::printf("  %s\n", line);
        Check(diff < tolerance, line);
    }

    std::printf("GPU: running out of device memory is reported, not a crash\n");
    {
        cosmic::UiValues ui;
        cosmic::PlaceDefaultPoints(&ui, static_cast<float>(w), static_cast<float>(h));
        ui.defocus_px = 20.0f;
        const cosmic::CosmicSettings settings = cosmic::SettingsFromUi(ui, static_cast<float>(w), static_cast<float>(h), 0);
        cosmic_test::EmulatedDevice device;
        cosmic::DevicePtr dst = 0;
        device.Allocate(static_cast<std::size_t>(w) * h * sizeof(FrameBGRA), &dst);
        device.SetBudget(static_cast<std::size_t>(w) * h * sizeof(FrameBGRA) + 200000);
        cosmic::GpuRender g;
        g.dest = cosmic::GpuFrame{dst, w, h, w};
        const cosmic::CosmicResult result = cosmic::RenderCosmicGpu(settings, g, device);
        Check(result == cosmic::CosmicResult::kOutOfMemory, "out of memory is reported");
        device.Free(dst);
        Check(device.allocations() == device.frees(), "and everything allocated is freed");
    }

    std::printf("\n%d checks, %d failed\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
