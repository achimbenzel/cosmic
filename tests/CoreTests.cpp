// Unit tests for the gradient core. No framework: each test prints what it
// checks and the process exits non-zero if anything failed.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "TestSupport.h"
#include "core/CosmicPipeline.h"
#include "core/Noise.h"
#include "core/Palette.h"
#include "core/Pyramid.h"
#include "core/UiModel.h"

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

std::string Fmt(const char* format, double a, double b = 0.0) {
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), format, a, b);
    return buffer;
}

// A disc with a soft edge on transparency, standing in for a shape layer.
TestImage MakeDisc(int width, int height, PixelDepth depth) {
    TestImage image(width, height, depth);
    const float cx = width * 0.5f;
    const float cy = height * 0.5f;
    const float r = std::min(width, height) * 0.3f;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float d = std::sqrt((x + 0.5f - cx) * (x + 0.5f - cx) + (y + 0.5f - cy) * (y + 0.5f - cy));
            const float a = std::clamp(r - d + 0.5f, 0.0f, 1.0f);
            image.SetPixel(x, y, PixelF{a, a, a, a});
        }
    }
    return image;
}

struct Output {
    TestImage image;
    int expansion = 0;
    int allocations = 0;
    int frees = 0;
    cosmic::CosmicResult result = cosmic::CosmicResult::kOk;
};

cosmic::UiValues Plain(int width, int height) {
    cosmic::UiValues ui;
    cosmic::PlaceDefaultPoints(&ui, static_cast<float>(width), static_cast<float>(height));
    return ui;
}

// Renders at a given downsample factor: the layer is `width` x `height` at
// full resolution, the buffers 1/downsample of that.
Output Render(const cosmic::UiValues& ui, const TestImage* layer, int width, int height, PixelDepth depth,
              int downsample = 1, bool expand = true, int frame = 0) {
    const cosmic::CosmicSettings settings =
        cosmic::SettingsFromUi(ui, static_cast<float>(width), static_cast<float>(height), frame);
    const float blur_scale = 1.0f / static_cast<float>(downsample);
    const float reach = cosmic::EffectReach(settings, blur_scale);
    Output out;
    out.expansion = (expand && cosmic::CanExpand(settings)) ? static_cast<int>(std::ceil(reach)) : 0;
    const int w = width / downsample;
    const int h = height / downsample;
    out.image.Resize(w + 2 * out.expansion, h + 2 * out.expansion, depth);

    cosmic::CosmicRender render;
    if (layer != nullptr) render.source = layer->View();
    render.dest = out.image.View();
    render.dest_left = -out.expansion;
    render.dest_top = -out.expansion;
    render.to_full_x = static_cast<float>(downsample);
    render.to_full_y = static_cast<float>(downsample);
    render.blur_scale = blur_scale;

    cosmic_test::MallocAllocator allocator;
    cosmic_test::ThreadPoolRunner runner(4);
    out.result = cosmic::RenderCosmic(settings, render, allocator, runner);
    out.allocations = allocator.allocations();
    out.frees = allocator.frees();
    return out;
}

double MaxDifference(const TestImage& a, const TestImage& b) {
    double worst = 0.0;
    for (int y = 0; y < a.View().height; ++y) {
        for (int x = 0; x < a.View().width; ++x) {
            const PixelF p = a.GetPixel(x, y);
            const PixelF q = b.GetPixel(x, y);
            worst = std::max({worst, (double)std::fabs(p.a - q.a), (double)std::fabs(p.r - q.r),
                              (double)std::fabs(p.g - q.g), (double)std::fabs(p.b - q.b)});
        }
    }
    return worst;
}

double MeanDifference(const TestImage& a, const TestImage& b, int inset) {
    double sum = 0.0;
    long count = 0;
    for (int y = inset; y < a.View().height - inset; ++y) {
        for (int x = inset; x < a.View().width - inset; ++x) {
            const PixelF p = a.GetPixel(x, y);
            const PixelF q = b.GetPixel(x, y);
            sum += std::fabs(p.r - q.r) + std::fabs(p.g - q.g) + std::fabs(p.b - q.b) + std::fabs(p.a - q.a);
            count += 4;
        }
    }
    return count > 0 ? sum / count : 0.0;
}

// ---------------------------------------------------------------------------

void TestPaletteEndpoints() {
    std::printf("palette: stops land on their positions in every blend\n");
    cosmic::Rgb stops[cosmic::kStopCount];
    for (int k = 0; k < cosmic::kStopCount; ++k) {
        const auto& c = cosmic::Preset(3).srgb[k];
        stops[k] = cosmic::Rgb{cosmic::SrgbToLinear(c[0] / 255.0f), cosmic::SrgbToLinear(c[1] / 255.0f),
                               cosmic::SrgbToLinear(c[2] / 255.0f)};
    }
    for (int blend = 0; blend < 4; ++blend) {
        cosmic::GradientLut lut;
        lut.Build(stops, static_cast<cosmic::ColorBlend>(blend), false);
        for (int k = 0; k < cosmic::kStopCount; ++k) {
            const cosmic::Rgb s = lut.Sample(k / 4.0f);
            const float err = std::max({std::fabs(s.r - stops[k].r), std::fabs(s.g - stops[k].g),
                                        std::fabs(s.b - stops[k].b)});
            Check(err < 2.0e-3f, Fmt("blend %.0f stop error %.5f", blend, err));
        }
        cosmic::GradientLut reversed;
        reversed.Build(stops, static_cast<cosmic::ColorBlend>(blend), true);
        const cosmic::Rgb r0 = reversed.Sample(0.0f);
        Check(std::fabs(r0.g - stops[4].g) < 2.0e-3f, "reversed palette starts on the last stop");
    }
}

void TestSmoothPaletteDoesNotOvershoot() {
    std::printf("palette: the smooth blend never leaves the range of its neighbouring stops\n");
    for (int p = 0; p < cosmic::PresetCount(); ++p) {
        cosmic::Rgb stops[cosmic::kStopCount];
        float lab_l[cosmic::kStopCount];
        for (int k = 0; k < cosmic::kStopCount; ++k) {
            const auto& c = cosmic::Preset(p).srgb[k];
            stops[k] = cosmic::Rgb{cosmic::SrgbToLinear(c[0] / 255.0f), cosmic::SrgbToLinear(c[1] / 255.0f),
                                   cosmic::SrgbToLinear(c[2] / 255.0f)};
            lab_l[k] = cosmic::LinearSrgbToOklab(stops[k]).l;
        }
        cosmic::GradientLut lut;
        lut.Build(stops, cosmic::ColorBlend::kOklabSmooth, false);
        bool ok = true;
        for (int i = 0; i <= 400; ++i) {
            const float t = i / 400.0f;
            const int seg = std::min(3, static_cast<int>(t * 4.0f));
            const float lo = std::min(lab_l[seg], lab_l[seg + 1]) - 2.0e-3f;
            const float hi = std::max(lab_l[seg], lab_l[seg + 1]) + 2.0e-3f;
            const float l = cosmic::LinearSrgbToOklab(lut.Sample(t)).l;
            if (l < lo || l > hi) ok = false;
        }
        Check(ok, std::string("no lightness overshoot in ") + cosmic::Preset(p).name);
    }
}

void TestPresetPopupString() {
    std::printf("palette: popup string lists every preset and Custom\n");
    const std::string names = cosmic::PresetPopupString();
    int bars = 0;
    for (char c : names) bars += c == '|';
    Check(bars == cosmic::PresetCount(), "one separator per preset");
    Check(names.rfind("|Custom") == names.size() - 7, "Custom is last");
}

void TestNoiseLoops() {
    std::printf("noise: evolution is periodic in a full turn\n");
    cosmic::FbmSettings fbm;
    fbm.octaves = 4.5f;
    fbm.seed = 1234;
    double worst = 0.0;
    double range = 0.0;
    for (int i = 0; i < 200; ++i) {
        const float x = i * 0.137f - 5.0f;
        const float y = i * 0.071f + 3.0f;
        const float phase = i * 0.05f;
        const float a = cosmic::LoopingFbm(x, y, phase, fbm);
        const float b = cosmic::LoopingFbm(x, y, phase + 6.283185307f, fbm);
        worst = std::max(worst, (double)std::fabs(a - b));
        range = std::max(range, (double)std::fabs(a));
    }
    Check(worst < 1.0e-4, Fmt("loop mismatch %.6f", worst));
    Check(range > 0.2 && range < 1.5, Fmt("fbm amplitude %.3f is in a sensible range", range));
}

void TestSeamlessLoopRender() {
    std::printf("render: angle a and a + 360 give the same frame (seamless loop)\n");
    const int w = 240;
    const int h = 135;
    cosmic::UiValues ui = Plain(w, h);
    ui.matte = 3;
    ui.turbulence_pct = 12.0f;
    ui.grain_pct = 0.0f;
    ui.angle_deg = 37.0f;
    const Output a = Render(ui, nullptr, w, h, PixelDepth::kFloat32);
    ui.angle_deg = 397.0f;
    const Output b = Render(ui, nullptr, w, h, PixelDepth::kFloat32);
    const double diff = MaxDifference(a.image, b.image);
    Check(diff < 2.0e-3, Fmt("max difference %.6f", diff));
    ui.angle_deg = 127.0f;
    const Output c = Render(ui, nullptr, w, h, PixelDepth::kFloat32);
    Check(MaxDifference(a.image, c.image) > 0.05, "a quarter turn does change the frame");
}

void TestBitDepthsAgree() {
    std::printf("render: 8, 16 and 32 bpc agree\n");
    const int w = 200;
    const int h = 120;
    cosmic::UiValues ui = Plain(w, h);
    ui.grain_pct = 0.0f;
    ui.working_space = 3;  // sRGB everywhere, so the three depths encode alike
    const TestImage disc8 = MakeDisc(w, h, PixelDepth::kBits8);
    const TestImage disc16 = MakeDisc(w, h, PixelDepth::kBits16);
    const TestImage disc32 = MakeDisc(w, h, PixelDepth::kFloat32);
    const Output a = Render(ui, &disc8, w, h, PixelDepth::kBits8);
    const Output b = Render(ui, &disc16, w, h, PixelDepth::kBits16);
    const Output c = Render(ui, &disc32, w, h, PixelDepth::kFloat32);
    Check(a.result == cosmic::CosmicResult::kOk && b.result == cosmic::CosmicResult::kOk &&
              c.result == cosmic::CosmicResult::kOk,
          "all depths render");
    const double ab = MaxDifference(a.image, b.image);
    const double bc = MaxDifference(b.image, c.image);
    Check(ab < 3.0 / 255.0, Fmt("8 vs 16 bpc max difference %.4f", ab));
    Check(bc < 1.5 / 255.0, Fmt("16 vs 32 bpc max difference %.4f", bc));
}

void TestTransparencyAndPremultiplication() {
    std::printf("render: no light where the matte is empty, and 8-bit stays premultiplied\n");
    const int w = 160;
    const int h = 100;
    cosmic::UiValues ui = Plain(w, h);
    ui.glow_intensity_pct = 0.0f;
    ui.diffusion_pct = 0.0f;
    ui.grain_pct = 0.0f;
    const TestImage disc = MakeDisc(w, h, PixelDepth::kBits8);
    const Output out = Render(ui, &disc, w, h, PixelDepth::kBits8);
    bool empty_ok = true;
    bool premul_ok = true;
    bool alpha_ok = true;
    const cosmic::HostImage& v = out.image.View();
    for (int y = 0; y < v.height; ++y) {
        const cosmic::Pixel8* row = static_cast<const cosmic::Pixel8*>(static_cast<const void*>(v.ConstRow(y)));
        for (int x = 0; x < v.width; ++x) {
            const int sx = x - out.expansion;
            const int sy = y - out.expansion;
            const float src_a = (sx >= 0 && sy >= 0 && sx < w && sy < h) ? disc.GetPixel(sx, sy).a : 0.0f;
            const cosmic::Pixel8& p = row[x];
            if (src_a == 0.0f && (p.a | p.r | p.g | p.b) != 0) empty_ok = false;
            if (p.r > p.a || p.g > p.a || p.b > p.a) premul_ok = false;
            if (std::fabs(p.a / 255.0f - src_a) > 1.0f / 255.0f) alpha_ok = false;
        }
    }
    Check(empty_ok, "transparent source stays transparent without glow");
    Check(premul_ok, "no channel above alpha");
    Check(alpha_ok, "alpha follows the layer's matte");
}

void TestOpacityZeroIsPassThrough() {
    std::printf("render: opacity 0 with no glow returns the layer\n");
    const int w = 120;
    const int h = 80;
    cosmic::UiValues ui = Plain(w, h);
    ui.opacity_pct = 0.0f;
    ui.glow_intensity_pct = 0.0f;
    ui.diffusion_pct = 0.0f;
    ui.grain_pct = 0.0f;
    ui.protection_pct = 0.0f;
    const TestImage disc = MakeDisc(w, h, PixelDepth::kBits16);
    const Output out = Render(ui, &disc, w, h, PixelDepth::kBits16, 1, false);
    const double diff = MaxDifference(out.image, disc);
    Check(diff < 1.5 / 255.0, Fmt("max difference %.5f", diff));
}

void TestGlowConservesLight() {
    std::printf("render: the glow spreads light without making any\n");
    const int w = 200;
    const int h = 200;
    cosmic::UiValues ui = Plain(w, h);
    ui.working_space = 2;  // linear
    ui.diffusion_pct = 0.0f;
    ui.grain_pct = 0.0f;
    ui.protection_pct = 0.0f;
    ui.glow_threshold = 0.0f;
    ui.glow_softness_pct = 0.0f;
    ui.glow_radius_px = 30.0f;
    const TestImage disc = MakeDisc(w, h, PixelDepth::kFloat32);

    ui.glow_intensity_pct = 0.0f;
    const Output base = Render(ui, &disc, w, h, PixelDepth::kFloat32);
    ui.glow_intensity_pct = 100.0f;
    const Output lit = Render(ui, &disc, w, h, PixelDepth::kFloat32);

    auto total = [](const TestImage& img) {
        double sum = 0.0;
        for (int y = 0; y < img.View().height; ++y) {
            for (int x = 0; x < img.View().width; ++x) {
                const PixelF p = img.GetPixel(x, y);
                sum += p.r + p.g + p.b;
            }
        }
        return sum;
    };
    const double before = total(base.image);
    const double after = total(lit.image);
    const double ratio = (after - before) / before;
    Check(std::fabs(ratio - 1.0) < 0.03, Fmt("glow adds %.4f of the source's light (want 1)", ratio));

    // And nothing reaches the edge of the bounds grown for it.
    const TestImage& img = lit.image;
    double edge = 0.0;
    for (int x = 0; x < img.View().width; ++x) {
        edge = std::max(edge, (double)img.GetPixel(x, 0).g);
        edge = std::max(edge, (double)img.GetPixel(x, img.View().height - 1).g);
    }
    Check(edge < 2.0e-3, Fmt("light at the bounds' edge %.5f", edge));
}

void TestResolutionIndependence() {
    std::printf("render: Half resolution matches Full\n");
    const int w = 320;
    const int h = 200;
    cosmic::UiValues ui = Plain(w, h);
    ui.grain_pct = 0.0f;
    ui.defocus_px = 6.0f;
    ui.focus_radius_pct = 10.0f;
    const TestImage disc_full = MakeDisc(w, h, PixelDepth::kFloat32);
    const TestImage disc_half = MakeDisc(w / 2, h / 2, PixelDepth::kFloat32);
    const Output full = Render(ui, &disc_full, w, h, PixelDepth::kFloat32, 1, false);
    const Output half = Render(ui, &disc_half, w, h, PixelDepth::kFloat32, 2, false);

    // Box-filter the full render down to half and compare.
    TestImage reduced(w / 2, h / 2, PixelDepth::kFloat32);
    for (int y = 0; y < h / 2; ++y) {
        for (int x = 0; x < w / 2; ++x) {
            PixelF acc{0, 0, 0, 0};
            for (int j = 0; j < 2; ++j) {
                for (int i = 0; i < 2; ++i) {
                    const PixelF p = full.image.GetPixel(2 * x + i, 2 * y + j);
                    acc.a += p.a * 0.25f;
                    acc.r += p.r * 0.25f;
                    acc.g += p.g * 0.25f;
                    acc.b += p.b * 0.25f;
                }
            }
            reduced.SetPixel(x, y, acc);
        }
    }
    const double mean = MeanDifference(reduced, half.image, 2);
    Check(mean < 0.01, Fmt("mean difference %.5f", mean));
}

void TestContentBounds() {
    std::printf("geometry: content bounds put the palette's ends on the content's edges\n");
    cosmic::UiValues ui = Plain(1000, 500);
    cosmic::CosmicSettings s = cosmic::SettingsFromUi(ui, 1000.0f, 500.0f, 0);
    s.depth = 0.0f;
    cosmic::ReferenceBox box;
    box.x0 = 300.0f;
    box.y0 = 200.0f;
    box.width = 400.0f;
    box.height = 100.0f;
    // Angle 180: the palette runs top to bottom.
    const float top = cosmic::GradientCoordinate(s, box, 500.0f, 200.0f);
    const float bottom = cosmic::GradientCoordinate(s, box, 500.0f, 300.0f);
    const float mid = cosmic::GradientCoordinate(s, box, 500.0f, 250.0f);
    Check(std::fabs(top) < 1.0e-4f, Fmt("top of the content is t = %.5f", top));
    Check(std::fabs(bottom - 1.0f) < 1.0e-4f, Fmt("bottom of the content is t = %.5f", bottom));
    Check(std::fabs(mid - 0.5f) < 1.0e-4f, Fmt("middle is t = %.5f", mid));

    // Found from the layer's pixels: a disc in a larger layer.
    const TestImage disc = MakeDisc(400, 300, PixelDepth::kBits8);
    cosmic::CosmicRender render;
    render.source = disc.View();
    const cosmic::CosmicSettings fit = cosmic::SettingsFromUi(Plain(400, 300), 400.0f, 300.0f, 0);
    const cosmic::ReferenceBox found = cosmic::FindReferenceBox(fit, render);
    Check(std::fabs(found.width - 180.0f) <= 2.0f && std::fabs(found.height - 180.0f) <= 2.0f,
          Fmt("disc bounds %.1f x %.1f (want 180)", found.width, found.height));
    cosmic::UiValues layer_ui = Plain(400, 300);
    layer_ui.fit = 2;
    const cosmic::ReferenceBox layer_box =
        cosmic::FindReferenceBox(cosmic::SettingsFromUi(layer_ui, 400.0f, 300.0f, 0), render);
    Check(layer_box.width == 400.0f && layer_box.height == 300.0f, "Fit: Layer uses the layer");
}

void TestRepeatModes() {
    std::printf("geometry: repeat modes\n");
    cosmic::UiValues ui = Plain(100, 100);
    ui.fit = 2;
    ui.depth_pct = 0.0f;
    ui.cycles = 2.0f;
    cosmic::ReferenceBox box;
    box.width = 100.0f;
    box.height = 100.0f;
    ui.repeat = 1;
    cosmic::CosmicSettings s = cosmic::SettingsFromUi(ui, 100.0f, 100.0f, 0);
    Check(cosmic::GradientCoordinate(s, box, 50.0f, 90.0f) == 1.0f, "None clamps");
    ui.repeat = 2;
    s = cosmic::SettingsFromUi(ui, 100.0f, 100.0f, 0);
    Check(std::fabs(cosmic::GradientCoordinate(s, box, 50.0f, 75.0f) - 0.5f) < 1.0e-4f, "Repeat wraps");
    ui.repeat = 3;
    s = cosmic::SettingsFromUi(ui, 100.0f, 100.0f, 0);
    Check(std::fabs(cosmic::GradientCoordinate(s, box, 50.0f, 75.0f) - 0.5f) < 1.0e-4f &&
              std::fabs(cosmic::GradientCoordinate(s, box, 50.0f, 50.0f) - 1.0f) < 1.0e-4f,
          "Mirror folds back");
}

void TestBulge() {
    std::printf("geometry: bulge magnifies inside its radius and leaves the rest alone\n");
    cosmic::UiValues ui = Plain(1000, 1000);
    ui.fit = 2;
    ui.angle_deg = 90.0f;  // left to right
    ui.depth_shape = 5;
    ui.depth_x = 500.0f;
    ui.depth_y = 500.0f;
    ui.depth_radius_pct = 20.0f;  // 200 px
    cosmic::ReferenceBox box;
    box.width = 1000.0f;
    box.height = 1000.0f;
    auto slope = [&](float depth_pct, float x) {
        ui.depth_pct = depth_pct;
        const cosmic::CosmicSettings s = cosmic::SettingsFromUi(ui, 1000.0f, 1000.0f, 0);
        return cosmic::GradientCoordinate(s, box, x + 1.0f, 500.0f) - cosmic::GradientCoordinate(s, box, x - 1.0f, 500.0f);
    };
    const float plain = slope(0.0f, 500.0f);
    Check(slope(60.0f, 500.0f) < plain * 0.6f, "positive depth magnifies the centre");
    Check(slope(-60.0f, 500.0f) > plain * 1.4f, "negative depth pinches the centre");
    Check(std::fabs(slope(60.0f, 800.0f) - plain) < 1.0e-5f, "outside the radius nothing moves");

    // One-to-one at any strength: the coordinate never runs backwards.
    bool monotonic = true;
    for (float depth : {-1000.0f, -120.0f, 95.0f, 1000.0f}) {
        ui.depth_pct = depth;
        const cosmic::CosmicSettings s = cosmic::SettingsFromUi(ui, 1000.0f, 1000.0f, 0);
        float previous = -1.0f;
        for (int x = 250; x <= 750; ++x) {
            const float t = cosmic::GradientCoordinate(s, box, static_cast<float>(x), 500.0f);
            if (t < previous - 1.0e-6f) monotonic = false;
            previous = t;
        }
    }
    Check(monotonic, "no fold-over at extreme depths");
}

void TestDefaults() {
    std::printf("defaults: turbulence does not follow the angle unless asked to\n");
    cosmic::UiValues ui = Plain(100, 100);
    ui.angle_deg = 123.0f;
    ui.evolution_deg = 10.0f;
    const cosmic::CosmicSettings s = cosmic::SettingsFromUi(ui, 100.0f, 100.0f, 0);
    Check(std::fabs(s.evolution - 10.0f * 3.14159265f / 180.0f) < 1.0e-5f, "evolution is the Evolution control");
    ui.evolve_with_angle = true;
    const cosmic::CosmicSettings looped = cosmic::SettingsFromUi(ui, 100.0f, 100.0f, 0);
    Check(std::fabs(looped.evolution - 133.0f * 3.14159265f / 180.0f) < 1.0e-5f, "Loop With Angle adds the angle");
}

void TestFullFrameEdges() {
    std::printf("render: a full-frame blur carries the edge on instead of darkening it\n");
    const int w = 200;
    const int h = 120;
    cosmic::UiValues ui = Plain(w, h);
    ui.matte = 3;
    cosmic::ApplyPreset(&ui, 13);
    for (int k = 0; k < cosmic::kStopCount; ++k) ui.colors[k][0] = ui.colors[k][1] = ui.colors[k][2] = 128;
    ui.grain_pct = 0.0f;
    ui.glow_intensity_pct = 0.0f;
    ui.diffusion_pct = 100.0f;
    ui.diffusion_radius_px = 80.0f;
    ui.defocus_px = 30.0f;
    ui.focus_radius_pct = 0.0f;
    const Output out = Render(ui, nullptr, w, h, PixelDepth::kFloat32);
    const PixelF corner = out.image.GetPixel(0, 0);
    const PixelF centre = out.image.GetPixel(w / 2, h / 2);
    Check(std::fabs(corner.g - centre.g) < 1.0e-3f && corner.a > 0.999f,
          Fmt("flat grey stays flat to the corner (%.4f vs %.4f)", corner.g, centre.g));
}

void TestMemoryBalanceAndOptions() {
    std::printf("render: every option renders and frees what it allocates\n");
    const int w = 180;
    const int h = 100;
    const TestImage disc = MakeDisc(w, h, PixelDepth::kBits8);
    bool ok = true;
    bool balanced = true;
    for (int type = 1; type <= 5; ++type) {
        for (int matte = 1; matte <= 3; ++matte) {
            for (int blend = 1; blend <= 5; ++blend) {
                cosmic::UiValues ui = Plain(w, h);
                ui.gradient_type = type;
                ui.matte = matte;
                ui.blend = blend;
                ui.depth_shape = 1 + (type + blend) % 5;
                ui.color_blend = 1 + (matte + blend) % 4;
                ui.repeat = 1 + type % 3;
                ui.defocus_px = (blend % 2) ? 5.0f : 0.0f;
                ui.animate_grain = true;
                const Output out = Render(ui, &disc, w, h, PixelDepth::kBits8, 1, true, blend);
                ok = ok && out.result == cosmic::CosmicResult::kOk;
                balanced = balanced && out.allocations == out.frees;
            }
        }
    }
    Check(ok, "all combinations render");
    Check(balanced, "allocations and frees balance");

    std::printf("render: an empty layer and a 1x1 layer are fine\n");
    TestImage empty(64, 64, PixelDepth::kBits8);
    cosmic::UiValues ui = Plain(64, 64);
    ui.defocus_px = 20.0f;
    Check(Render(ui, &empty, 64, 64, PixelDepth::kBits8).result == cosmic::CosmicResult::kOk, "empty layer");
    TestImage dot(1, 1, PixelDepth::kFloat32);
    dot.SetPixel(0, 0, PixelF{1, 1, 1, 1});
    Check(Render(Plain(1, 1), &dot, 1, 1, PixelDepth::kFloat32).result == cosmic::CosmicResult::kOk, "1x1 layer");
}

void TestNoBanding() {
    std::printf("render: 8-bit output is dithered\n");
    const int w = 1024;
    const int h = 64;
    cosmic::UiValues ui = Plain(w, h);
    ui.fit = 2;
    ui.matte = 3;
    ui.angle_deg = 90.0f;  // left to right
    ui.depth_pct = 0.0f;
    ui.turbulence_pct = 0.0f;
    ui.grain_pct = 0.0f;
    ui.glow_intensity_pct = 0.0f;
    ui.diffusion_pct = 0.0f;
    // A grey ramp that stays clear of 0 and 255, where dither is clamped:
    // 175 levels over 1024 px, so undithered it would step every 6 px.
    for (int k = 0; k < cosmic::kStopCount; ++k) {
        const std::uint8_t v = static_cast<std::uint8_t>(40 + k * 175 / 4);
        ui.colors[k][0] = ui.colors[k][1] = ui.colors[k][2] = v;
    }
    ui.color_blend = 4;  // sRGB: even steps in the output encoding
    const Output out = Render(ui, nullptr, w, h, PixelDepth::kBits8);
    const int run = cosmic_test::LongestFlatRun(out.image, h / 2);
    Check(run < 12, Fmt("longest flat run %.0f px", run));
}

void TestPyramidSigma() {
    std::printf("pyramid: level blur grows by an octave per level\n");
    for (int k = 2; k <= 8; ++k) {
        const float ratio = cosmic::PyramidLevelSigma(k) / cosmic::PyramidLevelSigma(k - 1);
        Check(ratio > 1.9f && ratio < 2.2f, Fmt("level %.0f ratio %.3f", k, ratio));
    }
}

}  // namespace

int main() {
    TestPaletteEndpoints();
    TestSmoothPaletteDoesNotOvershoot();
    TestPresetPopupString();
    TestNoiseLoops();
    TestPyramidSigma();
    TestContentBounds();
    TestRepeatModes();
    TestBulge();
    TestDefaults();
    TestSeamlessLoopRender();
    TestBitDepthsAgree();
    TestTransparencyAndPremultiplication();
    TestOpacityZeroIsPassThrough();
    TestGlowConservesLight();
    TestResolutionIndependence();
    TestNoBanding();
    TestFullFrameEdges();
    TestMemoryBalanceAndOptions();
    std::printf("\n%d checks, %d failed\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
