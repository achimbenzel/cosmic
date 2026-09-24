// Renders reference scenes through the gradient core and writes PNGs, so the
// look can be judged without After Effects.
//
// Usage: cosmic_preview <output directory> [layer.pgm]
//
// With a layer (an 8-bit binary PGM, used as the alpha of a white layer), only
// the Bulge scenes are rendered, on that layer.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "TestSupport.h"
#include "core/CosmicPipeline.h"
#include "core/UiModel.h"

using cosmic::PixelDepth;
using cosmic::PixelF;
using cosmic_test::TestImage;

namespace {

// 5x7 glyphs for the letters of the test word.
const char* Glyph(char c) {
    switch (c) {
        case 'C': return ".###.#...##....#....#....#...#.###.";
        case 'O': return ".###.#...##...##...##...##...#.###.";
        case 'S': return ".###.#...##.....###.....##...#.###.";
        case 'M': return "#...###.###.#.##.#.##...##...##...#";
        case 'I': return ".###...#....#....#....#....#...###.";
        default: return "...................................";
    }
}

// White "COSMIC" on transparency, anti-aliased by supersampling rounded cells,
// standing in for a text layer.
TestImage MakeTextLayer(int width, int height, PixelDepth depth) {
    TestImage image(width, height, depth);
    const char* word = "COSMIC";
    const int letters = 6;
    const float cell = std::min(width / (letters * 6.0f + 1.0f), height / 11.0f);
    const float text_w = cell * (letters * 6 - 1);
    const float text_h = cell * 7;
    const float x0 = (width - text_w) * 0.5f;
    const float y0 = (height - text_h) * 0.5f;
    constexpr int kSamples = 4;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int hits = 0;
            for (int sy = 0; sy < kSamples; ++sy) {
                for (int sx = 0; sx < kSamples; ++sx) {
                    const float px = x + (sx + 0.5f) / kSamples - x0;
                    const float py = y + (sy + 0.5f) / kSamples - y0;
                    if (px < 0 || py < 0) continue;
                    const int gx = static_cast<int>(px / cell);
                    const int gy = static_cast<int>(py / cell);
                    const int letter = gx / 6;
                    const int col = gx % 6;
                    if (letter >= letters || col >= 5 || gy >= 7) continue;
                    if (Glyph(word[letter])[gy * 5 + col] != '#') continue;
                    // Round each cell a little so the glyphs are not pure squares.
                    const float fx = px / cell - gx - 0.5f;
                    const float fy = py / cell - gy - 0.5f;
                    if (fx * fx + fy * fy <= 0.55f) ++hits;
                }
            }
            const float a = hits / static_cast<float>(kSamples * kSamples);
            image.SetPixel(x, y, PixelF{a, a, a, a});
        }
    }
    return image;
}

// An 8-bit binary PGM as the alpha of a white layer.
bool LoadPgm(const std::string& path, TestImage* out) {
    std::ifstream in(path, std::ios::binary);
    std::string magic;
    int width = 0;
    int height = 0;
    int max_value = 0;
    in >> magic >> width >> height >> max_value;
    in.get();
    if (!in || magic != "P5" || width <= 0 || height <= 0 || max_value != 255) return false;
    std::vector<unsigned char> bytes(static_cast<std::size_t>(width) * height);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in) return false;
    out->Resize(width, height, PixelDepth::kBits8);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float a = bytes[static_cast<std::size_t>(y) * width + x] / 255.0f;
            out->SetPixel(x, y, PixelF{a, a, a, a});
        }
    }
    return true;
}

struct Rendered {
    TestImage image;
    double milliseconds = 0.0;
};

// Renders `ui` onto a layer, growing the bounds the way the plug-in does.
Rendered Render(const cosmic::UiValues& ui, const TestImage* layer, int width, int height, PixelDepth depth) {
    cosmic::CosmicSettings settings = cosmic::SettingsFromUi(ui, static_cast<float>(width), static_cast<float>(height), 0);
    const float reach = cosmic::EffectReach(settings, 1.0f);
    const int expansion = (ui.expand_bounds && cosmic::CanExpand(settings)) ? static_cast<int>(std::ceil(reach)) : 0;

    Rendered out;
    out.image.Resize(width + 2 * expansion, height + 2 * expansion, depth);
    cosmic::CosmicRender render;
    if (layer != nullptr) render.source = layer->View();
    render.dest = out.image.View();
    render.dest_left = -expansion;
    render.dest_top = -expansion;

    cosmic_test::MallocAllocator allocator;
    cosmic_test::ThreadPoolRunner runner(static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
    const auto start = std::chrono::steady_clock::now();
    const cosmic::CosmicResult result = cosmic::RenderCosmic(settings, render, allocator, runner);
    out.milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (result != cosmic::CosmicResult::kOk) std::printf("render failed (%d)\n", static_cast<int>(result));
    if (allocator.allocations() != allocator.frees()) std::printf("LEAK: %d allocations, %d frees\n",
                                                                  allocator.allocations(), allocator.frees());
    return out;
}

// Full-frame gradient on a layer of the given size, with pixel distances
// scaled so a small tile looks like the same design at 1080p.
cosmic::UiValues FullFrame(int width, int height) {
    cosmic::UiValues ui;
    cosmic::PlaceDefaultPoints(&ui, static_cast<float>(width), static_cast<float>(height));
    ui.matte = 3;
    const float s = height / 1080.0f;
    ui.glow_radius_px *= s;
    ui.diffusion_radius_px *= s;
    ui.grain_size_px = std::max(0.6f, ui.grain_size_px * s);
    return ui;
}

void Blit(TestImage& sheet, const TestImage& tile, int x0, int y0) {
    for (int y = 0; y < tile.View().height; ++y) {
        for (int x = 0; x < tile.View().width; ++x) {
            const int sx = x0 + x;
            const int sy = y0 + y;
            if (sx < 0 || sy < 0 || sx >= sheet.View().width || sy >= sheet.View().height) continue;
            sheet.SetPixel(sx, sy, tile.GetPixel(x, y));
        }
    }
}

template <typename Fn>
void Sheet(const std::string& path, int columns, int count, int tile_w, int tile_h, const Fn& make) {
    const int rows = (count + columns - 1) / columns;
    const int gap = 6;
    TestImage sheet(columns * tile_w + (columns + 1) * gap, rows * tile_h + (rows + 1) * gap, PixelDepth::kBits8);
    for (int i = 0; i < count; ++i) {
        cosmic::UiValues ui = FullFrame(tile_w, tile_h);
        make(i, ui);
        const Rendered r = Render(ui, nullptr, tile_w, tile_h, PixelDepth::kBits8);
        Blit(sheet, r.image, gap + (i % columns) * (tile_w + gap), gap + (i / columns) * (tile_h + gap));
    }
    cosmic_test::WritePng(path, sheet);
    std::printf("wrote %s\n", path.c_str());
}

struct Scene {
    const char* name;
    void (*tweak)(cosmic::UiValues&);
};

// Bulge: the glass relief raised from the layer's shape, and what shapes it.
const Scene kBulgeScenes[] = {
    {"bulge_off", [](cosmic::UiValues&) {}},
    {"bulge_100", [](cosmic::UiValues& ui) { ui.bulge_pct = 100.0f; }},
    {"bulge_50", [](cosmic::UiValues& ui) { ui.bulge_pct = 50.0f; }},
    {"bulge_200", [](cosmic::UiValues& ui) { ui.bulge_pct = 200.0f; }},
    {"bulge_cushion",
     [](cosmic::UiValues& ui) {
         ui.bulge_pct = 100.0f;
         ui.rounding_pct = 0.0f;
     }},
    {"bulge_narrow",
     [](cosmic::UiValues& ui) {
         ui.bulge_pct = 100.0f;
         ui.softness_pct = 40.0f;
     }},
    {"bulge_wide",
     [](cosmic::UiValues& ui) {
         ui.bulge_pct = 100.0f;
         ui.softness_pct = 250.0f;
     }},
    {"bulge_no_light",
     [](cosmic::UiValues& ui) {
         ui.bulge_pct = 100.0f;
         ui.contrast_pct = 0.0f;
     }},
    {"bulge_sunset_radial",
     [](cosmic::UiValues& ui) {
         cosmic::ApplyPreset(&ui, 4);
         ui.gradient_type = 2;
         ui.size_pct = 120.0f;
         ui.bulge_pct = 100.0f;
     }},
    {"bulge_nebula_turbulent",
     [](cosmic::UiValues& ui) {
         cosmic::ApplyPreset(&ui, 2);
         ui.bulge_pct = 100.0f;
         ui.turbulence_pct = 12.0f;
     }},
};

void RenderScenes(const std::string& dir, const TestImage& layer, const Scene* scenes, int count) {
    const int w = layer.View().width;
    const int h = layer.View().height;
    for (int i = 0; i < count; ++i) {
        cosmic::UiValues ui;
        cosmic::PlaceDefaultPoints(&ui, static_cast<float>(w), static_cast<float>(h));
        const float s = h / 1080.0f;
        ui.glow_radius_px *= s;
        ui.diffusion_radius_px *= s;
        scenes[i].tweak(ui);
        const Rendered r = Render(ui, &layer, w, h, PixelDepth::kBits8);
        const std::string path = dir + "/" + scenes[i].name + ".png";
        cosmic_test::WritePng(path, r.image);
        std::printf("wrote %s (%dx%d, %.1f ms)\n", path.c_str(), r.image.View().width, r.image.View().height,
                    r.milliseconds);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    if (argc > 2) {
        TestImage layer;
        if (!LoadPgm(argv[2], &layer)) {
            std::printf("could not read %s (want an 8-bit binary PGM)\n", argv[2]);
            return 1;
        }
        RenderScenes(dir, layer, kBulgeScenes, static_cast<int>(sizeof(kBulgeScenes) / sizeof(kBulgeScenes[0])));
        return 0;
    }
    const int w = 960;
    const int h = 540;

    // Every palette, full frame.
    Sheet(dir + "/sheet_palettes.png", 4, cosmic::PresetCount(), 320, 180,
          [](int i, cosmic::UiValues& ui) { cosmic::ApplyPreset(&ui, i + 1); });

    // Gradient types.
    Sheet(dir + "/sheet_types.png", 3, 6, 320, 180, [](int i, cosmic::UiValues& ui) {
        cosmic::ApplyPreset(&ui, 2);
        ui.depth_pct = 0.0f;
        ui.gradient_type = i < 5 ? i + 1 : 3;
        if (i == 5) ui.repeat = 3, ui.cycles = 2.0f;  // conic, mirrored twice
    });

    // Depth shapes, on the default linear gradient: dome, sphere, ridge, wave,
    // then the lens magnifying and pinching (negative depth).
    Sheet(dir + "/sheet_depth.png", 3, 6, 320, 180, [](int i, cosmic::UiValues& ui) {
        cosmic::ApplyPreset(&ui, 4);
        ui.depth_shape = i < 5 ? i + 1 : 5;
        ui.depth_pct = i == 3 ? 12.0f : 45.0f;
        if (i >= 4) {
            ui.gradient_type = 1;
            ui.cycles = 4.0f;
            ui.repeat = 3;
            ui.depth_pct = i == 4 ? 70.0f : -80.0f;
            ui.depth_y = ui.center_y;
            ui.depth_radius_pct = 30.0f;
        }
    });

    // Turbulence and the loop: evolution a quarter turn apart, then the
    // full turn that must match the first.
    Sheet(dir + "/sheet_turbulence.png", 3, 6, 320, 180, [](int i, cosmic::UiValues& ui) {
        cosmic::ApplyPreset(&ui, 3);
        ui.turbulence_pct = i < 3 ? 4.0f + 6.0f * i : 12.0f;
        ui.complexity = i < 3 ? 3.0f : 5.0f;
        ui.angle_deg = 180.0f + (i >= 3 ? (i - 3) * 120.0f : 0.0f);
    });

    // Repeat modes and cycles.
    Sheet(dir + "/sheet_repeat.png", 3, 3, 320, 180, [](int i, cosmic::UiValues& ui) {
        cosmic::ApplyPreset(&ui, 7);
        ui.depth_pct = 20.0f;
        ui.cycles = 3.0f;
        ui.repeat = i + 1;
    });

    // Text layer: defaults, focus, glow off, blend modes.
    const TestImage text = MakeTextLayer(w, h, PixelDepth::kBits8);
    auto text_ui = [&]() {
        cosmic::UiValues ui;
        cosmic::PlaceDefaultPoints(&ui, static_cast<float>(w), static_cast<float>(h));
        const float s = h / 1080.0f;
        ui.glow_radius_px *= s;
        ui.diffusion_radius_px *= s;
        return ui;
    };
    struct Case {
        const char* name;
        void (*tweak)(cosmic::UiValues&);
    };
    const Case cases[] = {
        {"text_default", [](cosmic::UiValues&) {}},
        {"text_sunset", [](cosmic::UiValues& ui) { cosmic::ApplyPreset(&ui, 4); }},
        {"text_nebula_radial",
         [](cosmic::UiValues& ui) {
             cosmic::ApplyPreset(&ui, 2);
             ui.gradient_type = 2;
             ui.size_pct = 140.0f;
         }},
        {"text_focus",
         [](cosmic::UiValues& ui) {
             cosmic::ApplyPreset(&ui, 6);
             ui.focus_x = 200.0f;
             ui.defocus_px = 18.0f;
             ui.focus_radius_pct = 15.0f;
         }},
        {"text_no_glow",
         [](cosmic::UiValues& ui) {
             ui.glow_intensity_pct = 0.0f;
             ui.diffusion_pct = 0.0f;
         }},
        {"text_hot_glow",
         [](cosmic::UiValues& ui) {
             cosmic::ApplyPreset(&ui, 5);
             ui.glow_intensity_pct = 250.0f;
             ui.glow_threshold = 0.2f;
         }},
        {"text_hot_glow_unprotected",
         [](cosmic::UiValues& ui) {
             cosmic::ApplyPreset(&ui, 5);
             ui.glow_intensity_pct = 250.0f;
             ui.glow_threshold = 0.2f;
             ui.protection_pct = 0.0f;
         }},
        {"text_inverted",
         [](cosmic::UiValues& ui) {
             cosmic::ApplyPreset(&ui, 1);
             ui.matte = 2;
         }},
    };
    for (const Case& c : cases) {
        cosmic::UiValues ui = text_ui();
        c.tweak(ui);
        const Rendered r = Render(ui, &text, w, h, PixelDepth::kBits8);
        const std::string path = dir + "/" + c.name + ".png";
        cosmic_test::WritePng(path, r.image);
        std::printf("wrote %s (%dx%d, %.1f ms)\n", path.c_str(), r.image.View().width, r.image.View().height,
                    r.milliseconds);
    }

    RenderScenes(dir, text, kBulgeScenes, static_cast<int>(sizeof(kBulgeScenes) / sizeof(kBulgeScenes[0])));

    // Timing at 1080p and 4K, full frame, defaults.
    for (int scale = 1; scale <= 2; ++scale) {
        const int fw = 1920 * scale;
        const int fh = 1080 * scale;
        cosmic::UiValues ui;
        cosmic::PlaceDefaultPoints(&ui, static_cast<float>(fw), static_cast<float>(fh));
        ui.matte = 3;
        const Rendered r = Render(ui, nullptr, fw, fh, PixelDepth::kBits8);
        std::printf("full frame %dx%d defaults: %.1f ms\n", fw, fh, r.milliseconds);
        if (scale == 1) cosmic_test::WritePng(dir + "/hero_1080.png", r.image);
    }
    return 0;
}
