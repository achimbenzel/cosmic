#include "CosmicPipeline.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "ImageF.h"
#include "Noise.h"
#include "Plan.h"
#include "Pyramid.h"
#include "Transfer.h"

namespace cosmic {
namespace {

// ---------------------------------------------------------------------------
// Turbulence: displacement vectors on a lattice fixed in noise space,
// reconstructed with the cubic B-spline. The noise is smooth at the scale it
// is asked for, so evaluating it at every pixel would buy nothing but time.
// ---------------------------------------------------------------------------

class WarpGrid {
public:
    bool Active() const { return active_; }
    const WarpLattice& Lattice() const { return lattice_; }

    void Build(const WarpPlan& plan, TaskRunner& runner) {
        active_ = plan.active;
        if (!active_) return;
        lattice_ = plan.lattice;
        const int nx = lattice_.nx;
        data_.assign(static_cast<std::size_t>(nx) * static_cast<std::size_t>(lattice_.ny) * 2, 0.0f);
        ParallelRows(runner, lattice_.ny, [&](int begin, int end, int) {
            for (int j = begin; j < end; ++j) {
                for (int i = 0; i < nx; ++i) {
                    WarpNode(i, j, lattice_, plan.amount, plan.evolution, plan.fbm, plan.fbm2,
                             &data_[(static_cast<std::size_t>(j) * nx + i) * 2]);
                }
            }
        });
    }

    // Blends the four grid rows around grid coordinate gy into `row`
    // (nx * 2 floats).
    void Row(float gy, float* row) const {
        const float fl = std::floor(gy);
        float w[4];
        BsplineWeights(gy - fl, w);
        const int j0 = static_cast<int>(fl) - 1;
        const int floats = lattice_.nx * 2;
        std::fill(row, row + floats, 0.0f);
        for (int t = 0; t < 4; ++t) {
            const int j = std::clamp(j0 + t, 0, lattice_.ny - 1);
            const float* in = &data_[static_cast<std::size_t>(j) * floats];
            for (int i = 0; i < floats; ++i) row[i] += in[i] * w[t];
        }
    }

    void At(const float* row, float gx, float* wx, float* wy) const {
        const float fl = std::floor(gx);
        float w[4];
        BsplineWeights(gx - fl, w);
        const int i0 = static_cast<int>(fl) - 1;
        float ax = 0.0f;
        float ay = 0.0f;
        for (int t = 0; t < 4; ++t) {
            const int i = std::clamp(i0 + t, 0, lattice_.nx - 1);
            ax += row[i * 2] * w[t];
            ay += row[i * 2 + 1] * w[t];
        }
        *wx = ax;
        *wy = ay;
    }

    int RowFloats() const { return lattice_.nx * 2; }

private:
    bool active_ = false;
    WarpLattice lattice_;
    std::vector<float> data_;
};

// One pass of a level of the relief's pyramid over row y: SmoothAt at every
// pixel, with the pixels whose taps are all inside summed without the border
// checks, in the same order, so the result is the same to the bit.
void SmoothRow(const float* src, int w, int h, int y, int step, bool vertical, BorderMode border, float* out) {
    const float t0 = SmoothTap(0);
    const float t1 = SmoothTap(1);
    const float t2 = SmoothTap(2);
    const int reach = 2 * step;
    if (vertical) {
        if (y - reach < 0 || y + reach >= h) {
            for (int x = 0; x < w; ++x) out[x] = SmoothAt(src, w, h, x, y, step, 1, border);
            return;
        }
        const float* r0 = src + static_cast<std::size_t>(y - reach) * w;
        const float* r1 = src + static_cast<std::size_t>(y - step) * w;
        const float* r2 = src + static_cast<std::size_t>(y) * w;
        const float* r3 = src + static_cast<std::size_t>(y + step) * w;
        const float* r4 = src + static_cast<std::size_t>(y + reach) * w;
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            acc += r0[x] * t0;
            acc += r1[x] * t1;
            acc += r2[x] * t2;
            acc += r3[x] * t1;
            acc += r4[x] * t0;
            out[x] = acc;
        }
        return;
    }
    const float* row = src + static_cast<std::size_t>(y) * w;
    const int inner_begin = std::min(reach, w);
    const int inner_end = std::max(inner_begin, w - reach);
    for (int x = 0; x < inner_begin; ++x) out[x] = SmoothAt(src, w, h, x, y, step, 0, border);
    for (int x = inner_begin; x < inner_end; ++x) {
        float acc = 0.0f;
        acc += row[x - reach] * t0;
        acc += row[x - step] * t1;
        acc += row[x] * t2;
        acc += row[x + step] * t1;
        acc += row[x + reach] * t0;
        out[x] = acc;
    }
    for (int x = inner_end; x < w; ++x) out[x] = SmoothAt(src, w, h, x, y, step, 0, border);
}

// The layer's alpha for MeasureRow.
struct HostAlpha {
    const HostImage* image;
    float At(int x, int y) const {
        if (x < 0 || y < 0 || x >= image->width || y >= image->height) return 0.0f;
        return Clampf(ReadHostPixel(*image, image->ConstRow(y), x).a, 0.0f, 1.0f);
    }
};

ShapeMeasure MeasureSource(const CosmicRender& render, bool outline, TaskRunner* runner) {
    const HostImage& source = render.source;
    if (source.Empty()) return ShapeMeasure();
    std::vector<RowStats> rows(static_cast<std::size_t>(source.height));
    const HostAlpha alpha{&source};
    auto measure = [&](int begin, int end, int) {
        for (int y = begin; y < end; ++y) {
            rows[static_cast<std::size_t>(y)] = MeasureRow(alpha, y, source.width, kVisibleAlpha, outline ? 1 : 0,
                                                           render.to_full_x, render.to_full_y);
        }
    };
    if (runner != nullptr) {
        ParallelRows(*runner, source.height, measure);
    } else {
        measure(0, source.height, 0);
    }
    return MeasureShape(rows.data(), source.height);
}

// ---------------------------------------------------------------------------
// Pixel helpers
// ---------------------------------------------------------------------------

inline float Quantize(float value, float max_value, float dither) {
    const float scaled = value * max_value + dither;
    if (!(scaled > 0.0f)) return 0.0f;
    if (scaled >= max_value) return max_value;
    return std::floor(scaled + 0.5f);
}

void StorePixel(void* row, int x, const PixelF& p, PixelDepth depth, float dither) {
    switch (depth) {
        case PixelDepth::kBits8: {
            Pixel8& out = static_cast<Pixel8*>(row)[x];
            out.a = static_cast<std::uint8_t>(Quantize(p.a, kMaxChannel8, 0.0f));
            out.r = static_cast<std::uint8_t>(Quantize(p.r, kMaxChannel8, dither));
            out.g = static_cast<std::uint8_t>(Quantize(p.g, kMaxChannel8, dither));
            out.b = static_cast<std::uint8_t>(Quantize(p.b, kMaxChannel8, dither));
            // Premultiplied data must never have a channel above alpha.
            out.r = std::min(out.r, out.a);
            out.g = std::min(out.g, out.a);
            out.b = std::min(out.b, out.a);
            break;
        }
        case PixelDepth::kBits16: {
            Pixel16& out = static_cast<Pixel16*>(row)[x];
            out.a = static_cast<std::uint16_t>(Quantize(p.a, kMaxChannel16, 0.0f));
            out.r = static_cast<std::uint16_t>(Quantize(p.r, kMaxChannel16, dither));
            out.g = static_cast<std::uint16_t>(Quantize(p.g, kMaxChannel16, dither));
            out.b = static_cast<std::uint16_t>(Quantize(p.b, kMaxChannel16, dither));
            out.r = std::min(out.r, out.a);
            out.g = std::min(out.g, out.a);
            out.b = std::min(out.b, out.a);
            break;
        }
        case PixelDepth::kFloat32:
        default:
            static_cast<PixelF*>(row)[x] = p;
            break;
    }
}

const TransferFunction& WorkingTransfer(WorkingSpace space, PixelDepth depth) {
    return IsLinearWorkingSpace(space, depth == PixelDepth::kFloat32) ? TransferFunction::Identity()
                                                                       : TransferFunction::Srgb();
}

}  // namespace

float GradientCoordinate(const CosmicSettings& settings, const ReferenceBox& box, float x, float y) {
    return FieldValue(MakeField(settings, box), x, y);
}

ReferenceBox FindReferenceBox(const CosmicSettings& settings, const CosmicRender& render) {
    if (!WantsContentBounds(settings) || render.source.Empty()) return LayerBox(settings);
    return BoxFromShape(settings, MeasureSource(render, false, nullptr), render.source_left, render.source_top,
                        render.to_full_x, render.to_full_y);
}

float EffectReach(const CosmicSettings& s, float blur_scale) {
    float spread = 0.0f;
    if (s.glow_intensity > 0.0f && s.glow_radius > 0.0f) {
        spread = std::max(spread,
                          MakeBlurPlan(s.glow_radius * blur_scale * 0.5f, s.glow_falloff, kMaxPyramidLevels).reach);
    }
    if (s.diffusion > 0.0f && s.diffusion_radius > 0.0f) {
        spread = std::max(
            spread, MakeBlurPlan(s.diffusion_radius * blur_scale * 0.5f, kDiffusionFalloff, kMaxPyramidLevels).reach);
    }
    const float defocus = s.defocus > 0.0f ? DefocusReach(s.defocus * blur_scale) : 0.0f;
    // The glow and the veil spread the already defocused image, and blurs in
    // sequence add like Gaussians: in quadrature.
    return std::sqrt(spread * spread + defocus * defocus);
}

CosmicResult RenderCosmic(const CosmicSettings& settings, const CosmicRender& render, Allocator& allocator,
                          TaskRunner& runner) {
    const HostImage& dest = render.dest;
    if (dest.Empty()) return CosmicResult::kInvalidArguments;
    if (!(render.to_full_x > 0.0f) || !(render.to_full_y > 0.0f) || !(render.blur_scale > 0.0f)) {
        return CosmicResult::kInvalidArguments;
    }

    const TransferFunction& transfer = WorkingTransfer(settings.working_space, dest.depth);

    GradientLut lut;
    lut.Build(settings.stops, settings.color_blend, settings.reverse);
    const HostImage& source = render.source;
    ShapeMeasure shape;
    if (WantsShapeMeasure(settings) && !source.Empty()) {
        shape = MeasureSource(render, settings.bulge > 0.0f, &runner);
    }
    const ReferenceBox box =
        BoxFromShape(settings, shape, render.source_left, render.source_top, render.to_full_x, render.to_full_y);
    const Field field = MakeField(settings, box);

    // The canvas is the output. Where the matte fills the layer edge to edge,
    // the blurs carry the edge on instead of fading into a black border.
    const int width = dest.width;
    const int height = dest.height;
    const int canvas_left = render.dest_left;
    const int canvas_top = render.dest_top;
    const BorderMode border = CanExpand(settings) ? BorderMode::kZero : BorderMode::kClamp;

    WarpGrid warp;
    warp.Build(MakeWarpPlan(settings, box, render.to_full_x, render.to_full_y, canvas_left, canvas_top, width, height),
               runner);

    // --- 0. The Bulge relief: lookup offsets and palette shifts --------------
    OwnedImageF relief;
    ReliefGeometry geometry;
    geometry.canvas_left = canvas_left;
    geometry.canvas_top = canvas_top;
    geometry.width = width;
    geometry.height = height;
    geometry.source_left = render.source_left;
    geometry.source_top = render.source_top;
    geometry.source_width = source.width;
    geometry.source_height = source.height;
    geometry.to_full_x = render.to_full_x;
    geometry.to_full_y = render.to_full_y;
    geometry.blur_scale = render.blur_scale;
    const ReliefPlan relief_plan = MakeReliefPlan(settings, box, shape, geometry);
    if (relief_plan.active && !source.Empty()) {
        // Level 0 is the shape over the relief's domain; each level is
        // filtered from the one before into `level`, except level lo, which
        // goes to `kept` when level lo + 1 is needed as well.
        const int dw = relief_plan.domain_width;
        const int dh = relief_plan.domain_height;
        const int domain_x = canvas_left + relief_plan.domain_left;  // layer pixels
        const int domain_y = canvas_top + relief_plan.domain_top;
        const std::size_t count = static_cast<std::size_t>(dw) * static_cast<std::size_t>(dh);
        OwnedFloats level;
        OwnedFloats tmp;
        OwnedFloats kept;
        if (!level.Allocate(allocator, count) || !tmp.Allocate(allocator, count)) return CosmicResult::kOutOfMemory;
        if (relief_plan.hi > relief_plan.lo && !kept.Allocate(allocator, count)) return CosmicResult::kOutOfMemory;
        const int invert = relief_plan.invert;
        const float outside = ShapeValue(0.0f, invert);
        {
            float* shape_data = level.Data();
            ParallelRows(runner, dh, [&](int begin, int end, int) {
                for (int y = begin; y < end; ++y) {
                    float* out = shape_data + static_cast<std::size_t>(y) * dw;
                    std::fill(out, out + dw, outside);
                    const int sy = domain_y + y - render.source_top;
                    if (sy < 0 || sy >= source.height) continue;
                    const void* row = source.ConstRow(sy);
                    const int first = std::max(0, render.source_left - domain_x);
                    const int last = std::min(dw, render.source_left + source.width - domain_x);
                    for (int x = first; x < last; ++x) {
                        out[x] = ShapeValue(ReadHostPixel(source, row, domain_x + x - render.source_left).a, invert);
                    }
                }
            });
        }
        const float* src = level.Data();
        for (int k = 1; k <= relief_plan.hi; ++k) {
            const int step = SmoothLevelStep(k);
            float* t = tmp.Data();
            ParallelRows(runner, dh, [&](int begin, int end, int) {
                for (int y = begin; y < end; ++y) {
                    SmoothRow(src, dw, dh, y, step, false, border, t + static_cast<std::size_t>(y) * dw);
                }
            });
            float* dst = (k == relief_plan.lo && relief_plan.hi > relief_plan.lo) ? kept.Data() : level.Data();
            ParallelRows(runner, dh, [&](int begin, int end, int) {
                for (int y = begin; y < end; ++y) {
                    SmoothRow(t, dw, dh, y, step, true, border, dst + static_cast<std::size_t>(y) * dw);
                }
            });
            src = dst;
        }
        tmp.Release();

        const float* lo = relief_plan.hi > relief_plan.lo ? kept.Data() : level.Data();
        const float* hi = level.Data();
        if (!relief.Allocate(allocator, width, height)) return CosmicResult::kOutOfMemory;
        ImageF& out_image = relief.View();
        ParallelRows(runner, height, [&](int begin, int end, int) {
            for (int cy = begin; cy < end; ++cy) {
                PixelF* out = out_image.Row(cy);
                for (int cx = 0; cx < width; ++cx) {
                    float dx = 0.0f;
                    float dy = 0.0f;
                    float lift = 0.0f;
                    ReliefPixel(lo, hi, relief_plan.mix, dw, dh, cx - relief_plan.domain_left,
                                cy - relief_plan.domain_top, relief_plan.shape, &dx, &dy, &lift);
                    out[cx] = PixelF{lift, dx, dy, 0.0f};
                }
            }
        });
    }

    // --- 1. The gradient, matted and blended with the layer ----------------
    OwnedImageF base;
    if (!base.Allocate(allocator, width, height)) return CosmicResult::kOutOfMemory;
    {
        ImageF& out_image = base.View();
        const float opacity = std::clamp(settings.opacity, 0.0f, 1.0f);
        const WarpLattice& lattice = warp.Lattice();
        const bool has_relief = relief.Valid();
        ParallelRows(runner, height, [&](int begin, int end, int) {
            std::vector<PixelF> src_row(static_cast<std::size_t>(width));
            std::vector<float> warp_row(warp.Active() ? static_cast<std::size_t>(warp.RowFloats()) : 0);
            for (int cy = begin; cy < end; ++cy) {
                const int ly = canvas_top + cy;
                const int sy = ly - render.source_top;
                std::fill(src_row.begin(), src_row.end(), PixelF{0.0f, 0.0f, 0.0f, 0.0f});
                if (!source.Empty() && sy >= 0 && sy < source.height) {
                    const void* row = source.ConstRow(sy);
                    const int first = std::max(0, render.source_left - canvas_left);
                    const int last = std::min(width, render.source_left + source.width - canvas_left);
                    for (int cx = first; cx < last; ++cx) {
                        src_row[static_cast<std::size_t>(cx)] = LinearizePremultiplied(
                            ReadHostPixel(source, row, canvas_left + cx - render.source_left), transfer,
                            transfer.IsIdentity());
                    }
                }
                if (warp.Active()) {
                    warp.Row(WarpGridCoordinate(cy, canvas_top, render.to_full_y, lattice.anchor_y, lattice.scale,
                                                lattice.origin_j),
                             warp_row.data());
                }
                const PixelF* relief_row = has_relief ? relief.View().Row(cy) : nullptr;

                const float fy = (static_cast<float>(ly) + 0.5f) * render.to_full_y;
                PixelF* out = out_image.Row(cy);
                for (int cx = 0; cx < width; ++cx) {
                    const PixelF& src = src_row[static_cast<std::size_t>(cx)];
                    const float matte = MatteValue(settings.matte, src.a);
                    if (matte <= kTransparent) {
                        const float keep = 1.0f - opacity;
                        out[cx] = PixelF{src.a * keep, src.r * keep, src.g * keep, src.b * keep};
                        continue;
                    }
                    float x = (static_cast<float>(canvas_left + cx) + 0.5f) * render.to_full_x;
                    float y = fy;
                    if (warp.Active()) {
                        float wx = 0.0f;
                        float wy = 0.0f;
                        warp.At(warp_row.data(),
                                WarpGridCoordinate(cx, canvas_left, render.to_full_x, lattice.anchor_x, lattice.scale,
                                                   lattice.origin_i),
                                &wx, &wy);
                        x += wx;
                        y += wy;
                    }
                    float lift = 0.0f;
                    if (relief_row != nullptr) {
                        const PixelF& rv = relief_row[cx];
                        x += rv.r;
                        y += rv.g;
                        lift = rv.a;
                    }
                    Rgb color = lut.Sample(FieldValue(field, x, y, lift));
                    if (settings.blend != BlendMode::kNormal) {
                        const float inv = src.a > kTransparent ? 1.0f / src.a : 0.0f;
                        color = BlendColor(settings.blend, Rgb{src.r * inv, src.g * inv, src.b * inv}, color,
                                           TransferFunction::Srgb());
                    }
                    const float keep = 1.0f - opacity;
                    const float m = matte * opacity;
                    out[cx] = PixelF{m + src.a * keep, color.r * m + src.r * keep, color.g * m + src.g * keep,
                                     color.b * m + src.b * keep};
                }
            }
        });
    }

    relief.Release();

    // --- 2. Focus: sharp inside the radius, defocused beyond ----------------
    OwnedImageF focused;
    const float defocus = settings.defocus * render.blur_scale;
    if (defocus > 0.05f) {
        const int levels = std::min(LevelsForSigma(defocus), MaxUsefulLevels(width, height));
        Pyramid pyramid;
        const ImageF& b = base.View();
        if (!pyramid.Build(
                allocator, runner, width, height, levels,
                [&](int y, PixelF* out) {
                    const PixelF* in = b.Row(y);
                    std::copy(in, in + width, out);
                },
                border)) {
            return CosmicResult::kOutOfMemory;
        }
        if (!focused.Allocate(allocator, width, height)) return CosmicResult::kOutOfMemory;
        ImageF& f = focused.View();
        const int top = pyramid.Count();
        float sigmas[kMaxPyramidLevels + 2];
        for (int k = 0; k <= top; ++k) sigmas[k] = PyramidLevelSigma(k);
        sigmas[top + 1] = sigmas[top];
        std::vector<BsplineRowSampler> samplers;
        samplers.reserve(static_cast<std::size_t>(top));
        for (int k = 1; k <= top; ++k) samplers.emplace_back(pyramid.Level(k), k, width, border);
        const float falloff = std::max(0.0f, settings.focus_falloff);

        // Row by row: work out which levels each pixel blends, reconstruct
        // just the spans of those levels the row needs, then mix. A pixel's
        // blur is a mix of the two levels around its sigma.
        ParallelRows(runner, height, [&](int begin, int end, int) {
            constexpr unsigned char kSharp = 0xFF;
            std::vector<unsigned char> level_of(static_cast<std::size_t>(width));
            std::vector<float> mix_of(static_cast<std::size_t>(width));
            std::vector<PixelF> scratch(static_cast<std::size_t>(width));
            std::vector<std::vector<PixelF>> rows(static_cast<std::size_t>(top) + 1);
            for (auto& row : rows) row.resize(static_cast<std::size_t>(width));
            int span_begin[kMaxPyramidLevels + 1];
            int span_end[kMaxPyramidLevels + 1];

            for (int cy = begin; cy < end; ++cy) {
                const float y = (static_cast<float>(canvas_top + cy) + 0.5f) * render.to_full_y - settings.focus_y;
                for (int k = 0; k <= top; ++k) {
                    span_begin[k] = width;
                    span_end[k] = 0;
                }
                for (int cx = 0; cx < width; ++cx) {
                    const float x =
                        (static_cast<float>(canvas_left + cx) + 0.5f) * render.to_full_x - settings.focus_x;
                    const float d = std::sqrt(x * x + y * y) - settings.focus_radius;
                    float amount;
                    if (d <= 0.0f) {
                        amount = 0.0f;
                    } else if (falloff <= 0.0f) {
                        amount = 1.0f;
                    } else {
                        amount = Smoothstep01(d / falloff);
                    }
                    const float sigma = defocus * amount;
                    if (sigma <= 0.02f || top == 0) {
                        level_of[cx] = kSharp;
                        continue;
                    }
                    int k = 0;
                    while (k < top && sigmas[k + 1] <= sigma) ++k;
                    const int hi = std::min(k + 1, top);
                    level_of[cx] = static_cast<unsigned char>(k);
                    mix_of[cx] = k >= top ? 1.0f : (sigma - sigmas[k]) / (sigmas[k + 1] - sigmas[k]);
                    for (int level : {k, hi}) {
                        if (level < 1) continue;
                        span_begin[level] = std::min(span_begin[level], cx);
                        span_end[level] = std::max(span_end[level], cx + 1);
                    }
                }
                for (int k = 1; k <= top; ++k) {
                    if (span_begin[k] < span_end[k]) {
                        samplers[static_cast<std::size_t>(k - 1)].SampleRow(cy, scratch.data(), rows[k].data(),
                                                                            span_begin[k], span_end[k]);
                    }
                }

                const PixelF* in = b.Row(cy);
                PixelF* out = f.Row(cy);
                for (int cx = 0; cx < width; ++cx) {
                    const int k = level_of[cx];
                    if (k == kSharp) {
                        out[cx] = in[cx];
                        continue;
                    }
                    const int hi = std::min(k + 1, top);
                    const PixelF& lo = k == 0 ? in[cx] : rows[k][cx];
                    const PixelF& up = rows[hi][cx];
                    const float t = mix_of[cx];
                    out[cx] = PixelF{lo.a + (up.a - lo.a) * t, lo.r + (up.r - lo.r) * t, lo.g + (up.g - lo.g) * t,
                                     lo.b + (up.b - lo.b) * t};
                }
            }
        });
        base.Release();
    }
    const ImageF& image = focused.Valid() ? focused.View() : base.View();

    // --- 3. Glow and diffusion, each a collapsed pyramid at half resolution ---
    const int max_levels = MaxUsefulLevels(width, height);

    OwnedImageF glow;
    if (settings.glow_intensity > 0.0f && settings.glow_radius > 0.0f) {
        const BlurPlan plan =
            MakeBlurPlan(settings.glow_radius * render.blur_scale * 0.5f, settings.glow_falloff, max_levels);
        if (plan.levels > 0) {
            const GlowThreshold threshold = MakeGlowThreshold(settings);
            Pyramid pyramid;
            if (!pyramid.Build(
                    allocator, runner, width, height, plan.levels,
                    [&](int y, PixelF* out) {
                        const PixelF* in = image.Row(y);
                        for (int x = 0; x < width; ++x) out[x] = ExtractHighlight(in[x], threshold);
                    },
                    border)) {
                return CosmicResult::kOutOfMemory;
            }
            if (!CollapsePyramid(allocator, runner, pyramid, plan.weights, 1, plan.levels, &glow, border)) {
                return CosmicResult::kOutOfMemory;
            }
        }
    }

    OwnedImageF diffusion;
    const float diffusion_amount = std::clamp(settings.diffusion, 0.0f, 1.0f);
    if (diffusion_amount > 0.0f && settings.diffusion_radius > 0.0f) {
        const BlurPlan plan =
            MakeBlurPlan(settings.diffusion_radius * render.blur_scale * 0.5f, kDiffusionFalloff, max_levels);
        if (plan.levels > 0) {
            Pyramid pyramid;
            if (!pyramid.Build(
                    allocator, runner, width, height, plan.levels,
                    [&](int y, PixelF* out) {
                        const PixelF* in = image.Row(y);
                        std::copy(in, in + width, out);
                    },
                    border)) {
                return CosmicResult::kOutOfMemory;
            }
            if (!CollapsePyramid(allocator, runner, pyramid, plan.weights, 1, plan.levels, &diffusion, border)) {
                return CosmicResult::kOutOfMemory;
            }
        }
    }

    // --- 4. Composite into the host buffer -----------------------------------
    const FinishParams finish =
        MakeFinishParams(settings, render.to_full_x, render.to_full_y, !transfer.IsIdentity());
    const TransferFunction& srgb = TransferFunction::Srgb();
    const float intensity = std::max(0.0f, settings.glow_intensity);
    const bool dither = dest.depth != PixelDepth::kFloat32;

    const BsplineRowSampler* glow_sampler = nullptr;
    const BsplineRowSampler* diffusion_sampler = nullptr;
    // Constructed only when their image exists.
    std::vector<BsplineRowSampler> samplers;
    samplers.reserve(2);
    if (glow.Valid()) {
        samplers.emplace_back(glow.View(), 1, width, border);
        glow_sampler = &samplers.back();
    }
    if (diffusion.Valid()) {
        samplers.emplace_back(diffusion.View(), 1, width, border);
        diffusion_sampler = &samplers.back();
    }

    ParallelRows(runner, dest.height, [&](int begin, int end, int) {
        std::vector<PixelF> glow_row(glow_sampler ? static_cast<std::size_t>(width) : 0);
        std::vector<PixelF> diffusion_row(diffusion_sampler ? static_cast<std::size_t>(width) : 0);
        std::vector<PixelF> scratch(static_cast<std::size_t>(width));
        for (int y = begin; y < end; ++y) {
            const int cy = y;
            if (glow_sampler) glow_sampler->SampleRow(cy, scratch.data(), glow_row.data());
            if (diffusion_sampler) diffusion_sampler->SampleRow(cy, scratch.data(), diffusion_row.data());
            const PixelF* in = image.Row(cy);
            void* out_row = dest.Row(y);
            const int ly = render.dest_top + y;
            for (int x = 0; x < dest.width; ++x) {
                const int cx = x;
                PixelF c = in[cx];
                if (diffusion_sampler) {
                    // The veil spreads light outwards and dims what it takes it
                    // from, but never thins the shape itself: text under a
                    // diffusion filter is softer, not see-through.
                    const PixelF& d = diffusion_row[static_cast<std::size_t>(cx)];
                    c.a = std::max(c.a, c.a + (d.a - c.a) * diffusion_amount);
                    c.r += (d.r - c.r) * diffusion_amount;
                    c.g += (d.g - c.g) * diffusion_amount;
                    c.b += (d.b - c.b) * diffusion_amount;
                }
                if (glow_sampler) {
                    const PixelF& g = glow_row[static_cast<std::size_t>(cx)];
                    const float ga = std::clamp(g.a * intensity, 0.0f, 1.0f);
                    const float ca = std::clamp(c.a, 0.0f, 1.0f);
                    c.r += g.r * intensity;
                    c.g += g.g * intensity;
                    c.b += g.b * intensity;
                    c.a = ca + ga * (1.0f - ca);
                }
                const int lx = render.dest_left + x;
                const PixelF encoded = FinishPixel(c, finish, srgb, lx, ly);
                // Triangular dither of one step decorrelates the quantisation error
                // from the signal, so an 8-bit ramp shows noise instead of bands.
                const float d = dither ? TriangularNoise(lx, ly, 0x51ed270bu) : 0.0f;
                StorePixel(out_row, x, encoded, dest.depth, d);
            }
        }
    });

    return CosmicResult::kOk;
}

}  // namespace cosmic
