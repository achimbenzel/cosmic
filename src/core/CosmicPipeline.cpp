#include "CosmicPipeline.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "ImageF.h"
#include "Noise.h"
#include "Pyramid.h"
#include "Transfer.h"

namespace cosmic {
namespace {

constexpr float kOpaque = 0.999f;
constexpr float kTransparent = 1.0e-6f;
constexpr float kTwoPi = 6.28318530717958647f;

// A full-frame matte is padded by this much at most, so blurs near the frame
// edge see the gradient carry on rather than a black border. Beyond it the
// widest glow levels are already too faint to shade the edge visibly.
constexpr int kMaxInternalPad = 512;

// Glow, diffusion and defocus bounds: measured, a glow falls below a
// thousandth of its value next to the source within 3.0 to 3.6 sigma of its
// widest level, so this keeps everything visible inside the bounds without
// rendering far more empty canvas than needed.
constexpr float kReachSigmas = 3.4f;
constexpr float kReachMargin = 8.0f;

// Diffusion is a veil rather than a bloom: its levels are weighted a little
// towards the fine end, like the scatter of a mist filter.
constexpr float kDiffusionFalloff = 2.3f;

inline float Fract(float v) { return v - std::floor(v); }

inline float Smoothstep(float x) {
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;
    return x * x * (3.0f - 2.0f * x);
}

// ---------------------------------------------------------------------------
// Blur plans shared by the glow, the diffusion and the bounds.
// ---------------------------------------------------------------------------

struct BlurPlan {
    int levels = 0;                               // pyramid levels 1..levels
    float weights[kMaxPyramidLevels + 1] = {};    // indexed by level
    float reach = 0.0f;                           // render px
};

// Spreads a blur of the given sigma over pyramid levels whose weights follow a
// 1/r^falloff point spread: a Gaussian of sigma s has its peak at 1/s^2, so a
// level weighted s^(2 - falloff) contributes s^-falloff at its own scale. The
// top level fades in with the fractional part of the size, so animating the
// radius is continuous. Weights sum to one: the blur moves light, it does not
// make any.
BlurPlan MakeBlurPlan(float sigma, float falloff, int max_levels) {
    BlurPlan plan;
    if (!(sigma > 0.0f) || max_levels <= 0) return plan;
    const float first = PyramidLevelSigma(1);
    float position = sigma > first ? 1.0f + std::log2(sigma / first) : 1.0f;
    int top = static_cast<int>(std::ceil(position - 1.0e-4f));
    top = std::clamp(top, 1, std::min(max_levels, kMaxPyramidLevels));
    const float top_weight = std::clamp(position - static_cast<float>(top - 1), 0.05f, 1.0f);

    float sum = 0.0f;
    for (int k = 1; k <= top; ++k) {
        float w = std::pow(PyramidLevelSigma(k), 2.0f - falloff);
        if (k == top) w *= top_weight;
        plan.weights[k] = w;
        sum += w;
    }
    for (int k = 1; k <= top; ++k) plan.weights[k] /= sum;
    plan.levels = top;
    plan.reach = kReachSigmas * PyramidLevelSigma(top) + kReachMargin;
    return plan;
}

// Levels needed so the coarsest one is at least `sigma` wide.
int LevelsForSigma(float sigma) {
    int k = 0;
    while (k < kMaxPyramidLevels && PyramidLevelSigma(k) < sigma) ++k;
    return k;
}

// ---------------------------------------------------------------------------
// Gradient field
// ---------------------------------------------------------------------------

struct Field {
    GradientType type = GradientType::kLinear;
    float cx = 0.0f, cy = 0.0f;
    float dx = 0.0f, dy = 1.0f;  // along the gradient
    float ex = 1.0f, ey = 0.0f;  // across it
    float inv_span = 1.0f;
    float inv_half_span = 2.0f;
    float cycles = 1.0f;
    float offset = 0.0f;
    RepeatMode repeat = RepeatMode::kNone;

    DepthShape depth_shape = DepthShape::kDome;
    float depth = 0.0f;
    float depth_x = 0.0f, depth_y = 0.0f;
    float inv_depth_radius = 1.0f;
};

// Where the point controls land on the reference box: the layer's frame is
// mapped proportionally onto it, so a control at the layer's centre sits at
// the centre of the content.
struct BoxMapping {
    float sx = 1.0f, sy = 1.0f, ox = 0.0f, oy = 0.0f;
    float X(float x) const { return ox + x * sx; }
    float Y(float y) const { return oy + y * sy; }
};

BoxMapping MakeBoxMapping(const CosmicSettings& s, const ReferenceBox& box) {
    BoxMapping m;
    if (s.layer_width > 0.0f && s.layer_height > 0.0f) {
        m.sx = box.width / s.layer_width;
        m.sy = box.height / s.layer_height;
    }
    m.ox = box.x0;
    m.oy = box.y0;
    return m;
}

Field MakeField(const CosmicSettings& s, const ReferenceBox& box) {
    Field f;
    const BoxMapping map = MakeBoxMapping(s, box);
    const float w = std::max(1.0f, box.width);
    const float h = std::max(1.0f, box.height);
    f.type = s.type;
    f.cx = map.X(s.center_x);
    f.cy = map.Y(s.center_y);
    // After Effects angles: 0 is up, clockwise positive, y down.
    f.dx = std::sin(s.angle);
    f.dy = -std::cos(s.angle);
    f.ex = -f.dy;
    f.ey = f.dx;

    // Size 1 makes one palette length span the box: like CSS, a linear ramp
    // covers the box's extent along its own direction, a radial one reaches
    // the corners.
    const float along = std::fabs(w * f.dx) + std::fabs(h * f.dy);
    const float across = std::fabs(w * f.ex) + std::fabs(h * f.ey);
    float extent;
    switch (s.type) {
        case GradientType::kRadial: extent = std::sqrt(w * w + h * h); break;
        case GradientType::kDiamond: extent = along + across; break;
        case GradientType::kLinear:
        case GradientType::kReflected:
        case GradientType::kConic:
        default: extent = along; break;
    }
    const float span = std::max(1.0e-3f, s.size * extent);
    f.inv_span = 1.0f / span;
    f.inv_half_span = 2.0f / span;
    f.cycles = s.cycles;
    f.offset = s.offset;
    f.repeat = s.repeat;
    f.depth_shape = s.depth_shape;
    f.depth = s.depth;
    f.depth_x = map.X(s.depth_x);
    f.depth_y = map.Y(s.depth_y);
    f.inv_depth_radius = 1.0f / std::max(1.0e-3f, s.depth_radius * std::max(w, h));
    return f;
}

inline float FieldValue(const Field& f, float x, float y) {
    const float px = x - f.cx;
    const float py = y - f.cy;
    const float u = px * f.dx + py * f.dy;
    const float v = px * f.ex + py * f.ey;

    float t;
    switch (f.type) {
        case GradientType::kRadial:
            t = std::sqrt(px * px + py * py) * f.inv_half_span;
            break;
        case GradientType::kConic:
            // Starts at the angle's direction and sweeps clockwise.
            t = std::atan2(v, u) * (1.0f / kTwoPi);
            if (t < 0.0f) t += 1.0f;
            break;
        case GradientType::kDiamond:
            t = (std::fabs(u) + std::fabs(v)) * f.inv_half_span;
            break;
        case GradientType::kReflected:
            t = std::fabs(u) * f.inv_half_span;
            break;
        case GradientType::kLinear:
        default:
            t = u * f.inv_span + 0.5f;
            break;
    }

    if (f.depth != 0.0f) {
        const float qx = (x - f.depth_x) * f.inv_depth_radius;
        const float qy = (y - f.depth_y) * f.inv_depth_radius;
        float h = 0.0f;
        switch (f.depth_shape) {
            case DepthShape::kSphere: {
                const float r2 = qx * qx + qy * qy;
                h = r2 < 1.0f ? std::sqrt(1.0f - r2) : 0.0f;
                break;
            }
            case DepthShape::kRidge: {
                const float w = qx * f.ex + qy * f.ey;
                const float w2 = w * w;
                h = w2 < 1.0f ? (1.0f - w2) * (1.0f - w2) : 0.0f;
                break;
            }
            case DepthShape::kWave:
                h = std::sin(kTwoPi * (qx * f.ex + qy * f.ey));
                break;
            case DepthShape::kDome:
            default: {
                const float r2 = qx * qx + qy * qy;
                h = r2 < 1.0f ? (1.0f - r2) * (1.0f - r2) : 0.0f;
                break;
            }
        }
        t += f.depth * h;
    }

    t = t * f.cycles + f.offset;
    switch (f.repeat) {
        case RepeatMode::kRepeat:
            return Fract(t);
        case RepeatMode::kMirror: {
            const float m = Fract(t * 0.5f) * 2.0f;
            return m <= 1.0f ? m : 2.0f - m;
        }
        case RepeatMode::kNone:
        default:
            // A conic gradient has no ends to clamp to.
            if (f.type == GradientType::kConic) return Fract(t);
            return std::clamp(t, 0.0f, 1.0f);
    }
}

// ---------------------------------------------------------------------------
// Turbulence: a coarse grid of displacement vectors, reconstructed with the
// cubic B-spline. The noise is smooth at the scale it is asked for, so
// evaluating it at every pixel would buy nothing but time.
// ---------------------------------------------------------------------------

class WarpGrid {
public:
    bool Active() const { return active_; }

    void Build(const CosmicSettings& s, const ReferenceBox& box, const CosmicRender& r, int canvas_left,
               int canvas_top, int width, int height, TaskRunner& runner) {
        const float shorter = std::max(1.0f, std::min(box.width, box.height));
        const float turbulence = s.turbulence * shorter;
        const float turbulence_size = s.turbulence_size * shorter;
        active_ = turbulence > 0.0f && turbulence_size > 0.0f;
        if (!active_) return;
        const float size_render = turbulence_size * r.blur_scale;
        const float finest = size_render / std::ldexp(1.0f, static_cast<int>(std::ceil(s.complexity)) - 1);
        step_ = std::clamp(static_cast<int>(finest * 0.25f), 1, 16);
        nx_ = width / step_ + 5;
        ny_ = height / step_ + 5;
        data_.assign(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_) * 2, 0.0f);

        FbmSettings fbm;
        fbm.octaves = s.complexity;
        fbm.seed = Hash32(s.seed * 0x9e3779b9u + 0x85ebca6bu);
        FbmSettings fbm2 = fbm;
        fbm2.seed = Hash32(fbm.seed ^ 0x5bd1e995u);
        const float inv_size = 1.0f / turbulence_size;
        // Fractal noise sits well inside [-1, 1]; this brings its typical
        // swing up to about the displacement asked for.
        const float amount = turbulence * 1.6f;

        ParallelRows(runner, ny_, [&](int begin, int end, int) {
            for (int j = begin; j < end; ++j) {
                const float py = (static_cast<float>(canvas_top + (j - 2) * step_) + 0.5f) * r.to_full_y;
                for (int i = 0; i < nx_; ++i) {
                    const float px = (static_cast<float>(canvas_left + (i - 2) * step_) + 0.5f) * r.to_full_x;
                    const float qx = px * inv_size;
                    const float qy = py * inv_size;
                    float* out = &data_[(static_cast<std::size_t>(j) * nx_ + i) * 2];
                    out[0] = amount * LoopingFbm(qx, qy, s.evolution, fbm);
                    out[1] = amount * LoopingFbm(qx + 31.416f, qy - 47.853f, s.evolution, fbm2);
                }
            }
        });
    }

    // Blends the four grid rows around canvas row y into `row` (nx * 2 floats).
    void Row(int y, float* row) const {
        const float s = static_cast<float>(y) / static_cast<float>(step_);
        const float fl = std::floor(s);
        float w[4];
        BsplineWeights(s - fl, w);
        const int j0 = static_cast<int>(fl) - 1 + 2;
        std::fill(row, row + nx_ * 2, 0.0f);
        for (int t = 0; t < 4; ++t) {
            const int j = std::clamp(j0 + t, 0, ny_ - 1);
            const float* in = &data_[static_cast<std::size_t>(j) * nx_ * 2];
            for (int i = 0; i < nx_ * 2; ++i) row[i] += in[i] * w[t];
        }
    }

    void At(const float* row, int x, float* wx, float* wy) const {
        const float s = static_cast<float>(x) / static_cast<float>(step_);
        const float fl = std::floor(s);
        float w[4];
        BsplineWeights(s - fl, w);
        const int i0 = static_cast<int>(fl) - 1 + 2;
        float ax = 0.0f;
        float ay = 0.0f;
        for (int t = 0; t < 4; ++t) {
            const int i = std::min(i0 + t, nx_ - 1);
            ax += row[i * 2] * w[t];
            ay += row[i * 2 + 1] * w[t];
        }
        *wx = ax;
        *wy = ay;
    }

    int RowFloats() const { return nx_ * 2; }

private:
    bool active_ = false;
    int step_ = 1;
    int nx_ = 0;
    int ny_ = 0;
    std::vector<float> data_;
};

// ---------------------------------------------------------------------------
// Pixel helpers
// ---------------------------------------------------------------------------

inline PixelF LinearizePremultiplied(const PixelF& p, const TransferFunction& transfer) {
    if (transfer.IsIdentity()) return p;
    if (p.a >= kOpaque) return PixelF{p.a, transfer.Decode(p.r), transfer.Decode(p.g), transfer.Decode(p.b)};
    if (p.a <= kTransparent) return PixelF{p.a, 0.0f, 0.0f, 0.0f};
    const float inv = 1.0f / p.a;
    return PixelF{p.a, transfer.Decode(p.r * inv) * p.a, transfer.Decode(p.g * inv) * p.a,
                  transfer.Decode(p.b * inv) * p.a};
}

inline float ScreenChannel(float a, float b) {
    constexpr float kKnee = 0.75f;
    return a + b - SoftSaturate(a, kKnee) * SoftSaturate(b, kKnee);
}

inline float OverlayChannel(float base, float top) {
    const TransferFunction& srgb = TransferFunction::Srgb();
    const float a = std::clamp(srgb.Encode(base), 0.0f, 1.0f);
    const float b = std::clamp(srgb.Encode(top), 0.0f, 1.0f);
    const float o = a < 0.5f ? 2.0f * a * b : 1.0f - 2.0f * (1.0f - a) * (1.0f - b);
    return srgb.Decode(o);
}

// The gradient's colour once combined with the layer's own (unpremultiplied,
// linear) colour.
inline Rgb BlendColor(BlendMode mode, const Rgb& s, const Rgb& g) {
    switch (mode) {
        case BlendMode::kMultiply:
            return Rgb{s.r * g.r, s.g * g.g, s.b * g.b};
        case BlendMode::kScreen:
            return Rgb{ScreenChannel(s.r, g.r), ScreenChannel(s.g, g.g), ScreenChannel(s.b, g.b)};
        case BlendMode::kOverlay:
            return Rgb{OverlayChannel(s.r, g.r), OverlayChannel(s.g, g.g), OverlayChannel(s.b, g.b)};
        case BlendMode::kColor: {
            // The layer's lightness with the gradient's hue and chroma: keeps
            // the shading of text bevels and footage.
            const Lab ls = LinearSrgbToOklab(Rgb{std::max(0.0f, s.r), std::max(0.0f, s.g), std::max(0.0f, s.b)});
            const Lab lg = LinearSrgbToOklab(g);
            const Rgb out = OklabToLinearSrgb(Lab{ls.l, lg.a, lg.b});
            return Rgb{std::max(0.0f, out.r), std::max(0.0f, out.g), std::max(0.0f, out.b)};
        }
        case BlendMode::kNormal:
        default:
            return g;
    }
}

inline float MatteValue(MatteMode mode, float source_alpha) {
    switch (mode) {
        case MatteMode::kInvertedAlpha: return 1.0f - std::clamp(source_alpha, 0.0f, 1.0f);
        case MatteMode::kFullFrame: return 1.0f;
        case MatteMode::kLayerAlpha:
        default: return std::clamp(source_alpha, 0.0f, 1.0f);
    }
}

// Soft-knee highlight isolation on the pixel's own (unpremultiplied)
// brightness, so an anti-aliased edge emits in proportion to its coverage.
struct GlowThreshold {
    float level = 0.0f;
    float knee = 0.0f;
};

inline PixelF ExtractHighlight(const PixelF& p, const GlowThreshold& threshold) {
    if (p.a <= kTransparent) return PixelF{0.0f, 0.0f, 0.0f, 0.0f};
    const float inv = p.a >= kOpaque ? 1.0f : 1.0f / p.a;
    const float level = std::max(p.r, std::max(p.g, p.b)) * inv;
    if (level <= 0.0f) return PixelF{0.0f, 0.0f, 0.0f, 0.0f};
    float above = level - threshold.level;
    if (threshold.knee > 0.0f) {
        float soft = std::clamp(above + threshold.knee, 0.0f, 2.0f * threshold.knee);
        soft = soft * soft / (4.0f * threshold.knee);
        above = std::max(soft, above);
    }
    if (above <= 0.0f) return PixelF{0.0f, 0.0f, 0.0f, 0.0f};
    const float c = above / level;
    return PixelF{p.a * c, p.r * c, p.g * c, p.b * c};
}

// Film grain: a hashed value per grain cell, interpolated, so grain larger
// than a pixel is soft rather than blocky. Keyed to full-resolution layer
// coordinates, so it holds still under downsampling and moving bounds.
inline float GrainValue(float x, float y, float inv_size, std::uint32_t seed) {
    const float gx = x * inv_size;
    const float gy = y * inv_size;
    const float fx = std::floor(gx);
    const float fy = std::floor(gy);
    const int ix = static_cast<int>(fx);
    const int iy = static_cast<int>(fy);
    const float tx = Smoothstep(gx - fx);
    const float ty = Smoothstep(gy - fy);
    const float n00 = TriangularNoise(ix, iy, seed);
    const float n10 = TriangularNoise(ix + 1, iy, seed);
    const float n01 = TriangularNoise(ix, iy + 1, seed);
    const float n11 = TriangularNoise(ix + 1, iy + 1, seed);
    const float top = n00 + (n10 - n00) * tx;
    const float bottom = n01 + (n11 - n01) * tx;
    // Interpolation lowers the variance; this brings it back to about that of
    // a single cell.
    return (top + (bottom - top) * ty) * 1.35f;
}

inline float Quantize(float value, float max_value, float dither) {
    const float scaled = value * max_value + dither;
    if (!(scaled > 0.0f)) return 0.0f;
    if (scaled >= max_value) return max_value;
    return std::floor(scaled + 0.5f);
}

struct OutputContext {
    const TransferFunction* transfer = nullptr;  // working space encoding
    PixelDepth depth = PixelDepth::kBits8;
    float protection_knee = 1.0f;
    bool protect = false;
    float grain = 0.0f;
    float grain_inv_size = 1.0f;
    std::uint32_t grain_seed = 0;
    float to_full_x = 1.0f;
    float to_full_y = 1.0f;
};

// Linear premultiplied light to the working space's encoding, with highlight
// protection and grain. Returns a premultiplied pixel.
inline PixelF FinishPixel(const PixelF& c, const OutputContext& ctx, int lx, int ly) {
    const float a = std::clamp(c.a, 0.0f, 1.0f);
    if (a <= kTransparent) return PixelF{0.0f, 0.0f, 0.0f, 0.0f};
    const float inv = 1.0f / a;
    float r = std::max(0.0f, c.r * inv);
    float g = std::max(0.0f, c.g * inv);
    float b = std::max(0.0f, c.b * inv);

    if (ctx.protect) {
        // Rolls the brightest channel off and scales the others with it, so a
        // colour approaching the top of the range keeps its hue instead of
        // clipping channel by channel towards white.
        const float m = std::max(r, std::max(g, b));
        if (m > ctx.protection_knee) {
            const float scale = SoftSaturate(m, ctx.protection_knee) / m;
            r *= scale;
            g *= scale;
            b *= scale;
        }
    }

    const TransferFunction& srgb = TransferFunction::Srgb();
    const bool encode_srgb = !ctx.transfer->IsIdentity();
    if (ctx.grain > 0.0f) {
        // Grain lives in perceptual units, strongest in the midtones like film.
        float er = srgb.Encode(r);
        float eg = srgb.Encode(g);
        float eb = srgb.Encode(b);
        const float mean = std::clamp((er + eg + eb) * (1.0f / 3.0f), 0.0f, 1.0f);
        const float response = 0.35f + 2.6f * mean * (1.0f - mean);
        const float n = GrainValue((static_cast<float>(lx) + 0.5f) * ctx.to_full_x,
                                   (static_cast<float>(ly) + 0.5f) * ctx.to_full_y, ctx.grain_inv_size,
                                   ctx.grain_seed) *
                        ctx.grain * 0.18f * response;
        er = std::max(0.0f, er + n);
        eg = std::max(0.0f, eg + n);
        eb = std::max(0.0f, eb + n);
        if (encode_srgb) return PixelF{a, er * a, eg * a, eb * a};
        return PixelF{a, srgb.Decode(er) * a, srgb.Decode(eg) * a, srgb.Decode(eb) * a};
    }

    if (encode_srgb) return PixelF{a, srgb.Encode(r) * a, srgb.Encode(g) * a, srgb.Encode(b) * a};
    return PixelF{a, r * a, g * a, b * a};
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
    switch (space) {
        case WorkingSpace::kLinear: return TransferFunction::Identity();
        case WorkingSpace::kSrgb: return TransferFunction::Srgb();
        case WorkingSpace::kAuto:
        default:
            // The usual setups: 32 bpc projects are linearised, 8 and 16 are not.
            return depth == PixelDepth::kFloat32 ? TransferFunction::Identity() : TransferFunction::Srgb();
    }
}

}  // namespace

float GradientCoordinate(const CosmicSettings& settings, const ReferenceBox& box, float x, float y) {
    return FieldValue(MakeField(settings, box), x, y);
}

ReferenceBox FindReferenceBox(const CosmicSettings& settings, const CosmicRender& render) {
    ReferenceBox layer;
    layer.width = settings.layer_width;
    layer.height = settings.layer_height;
    const HostImage& source = render.source;
    if (settings.fit != FitMode::kContentBounds || settings.matte != MatteMode::kLayerAlpha || source.Empty()) {
        return layer;
    }

    // Anything below half an 8-bit step is treated as empty, so faint
    // anti-aliasing or a soft shadow does not stretch the box.
    constexpr float kVisible = 0.5f / 255.0f;
    int x0 = source.width;
    int x1 = -1;
    int y0 = source.height;
    int y1 = -1;
    for (int y = 0; y < source.height; ++y) {
        const void* row = source.ConstRow(y);
        int first = -1;
        for (int x = 0; x < source.width; ++x) {
            if (ReadHostPixel(source, row, x).a > kVisible) {
                first = x;
                break;
            }
        }
        if (first < 0) continue;
        int last = first;
        for (int x = source.width - 1; x > first; --x) {
            if (ReadHostPixel(source, row, x).a > kVisible) {
                last = x;
                break;
            }
        }
        x0 = std::min(x0, first);
        x1 = std::max(x1, last);
        y0 = std::min(y0, y);
        y1 = y;
    }
    if (x1 < x0 || y1 < y0) return layer;

    ReferenceBox box;
    box.x0 = static_cast<float>(x0 + render.source_left) * render.to_full_x;
    box.y0 = static_cast<float>(y0 + render.source_top) * render.to_full_y;
    box.width = static_cast<float>(x1 - x0 + 1) * render.to_full_x;
    box.height = static_cast<float>(y1 - y0 + 1) * render.to_full_y;
    return box;
}

float EffectReach(const CosmicSettings& s, float blur_scale) {
    float reach = 0.0f;
    if (s.glow_intensity > 0.0f && s.glow_radius > 0.0f) {
        reach = std::max(reach, MakeBlurPlan(s.glow_radius * blur_scale * 0.5f, s.glow_falloff, kMaxPyramidLevels).reach);
    }
    if (s.diffusion > 0.0f && s.diffusion_radius > 0.0f) {
        reach = std::max(
            reach, MakeBlurPlan(s.diffusion_radius * blur_scale * 0.5f, kDiffusionFalloff, kMaxPyramidLevels).reach);
    }
    if (s.defocus > 0.0f) {
        const int levels = LevelsForSigma(s.defocus * blur_scale);
        reach += kReachSigmas * PyramidLevelSigma(levels) + kReachMargin;
    }
    return reach;
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
    const ReferenceBox box = FindReferenceBox(settings, render);
    const Field field = MakeField(settings, box);

    // Canvas: the output, padded where the matte carries on past the layer.
    const float reach = EffectReach(settings, render.blur_scale);
    const int pad = CanExpand(settings) ? 0 : std::min(kMaxInternalPad, static_cast<int>(std::ceil(reach)));
    const int width = dest.width + 2 * pad;
    const int height = dest.height + 2 * pad;
    const int canvas_left = render.dest_left - pad;
    const int canvas_top = render.dest_top - pad;

    WarpGrid warp;
    warp.Build(settings, box, render, canvas_left, canvas_top, width, height, runner);

    // --- 1. The gradient, matted and blended with the layer ----------------
    OwnedImageF base;
    if (!base.Allocate(allocator, width, height)) return CosmicResult::kOutOfMemory;
    {
        ImageF& out_image = base.View();
        const HostImage& source = render.source;
        const float opacity = std::clamp(settings.opacity, 0.0f, 1.0f);
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
                            ReadHostPixel(source, row, canvas_left + cx - render.source_left), transfer);
                    }
                }
                if (warp.Active()) warp.Row(cy, warp_row.data());

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
                        warp.At(warp_row.data(), cx, &wx, &wy);
                        x += wx;
                        y += wy;
                    }
                    Rgb color = lut.Sample(FieldValue(field, x, y));
                    if (settings.blend != BlendMode::kNormal) {
                        const float inv = src.a > kTransparent ? 1.0f / src.a : 0.0f;
                        color = BlendColor(settings.blend, Rgb{src.r * inv, src.g * inv, src.b * inv}, color);
                    }
                    const float keep = 1.0f - opacity;
                    const float m = matte * opacity;
                    out[cx] = PixelF{m + src.a * keep, color.r * m + src.r * keep, color.g * m + src.g * keep,
                                     color.b * m + src.b * keep};
                }
            }
        });
    }

    // --- 2. Focus: sharp inside the radius, defocused beyond ----------------
    OwnedImageF focused;
    const float defocus = settings.defocus * render.blur_scale;
    if (defocus > 0.05f) {
        const int levels = std::min(LevelsForSigma(defocus), MaxUsefulLevels(width, height));
        Pyramid pyramid;
        const ImageF& b = base.View();
        if (!pyramid.Build(allocator, runner, width, height, levels, [&](int y, PixelF* out) {
                const PixelF* in = b.Row(y);
                std::copy(in, in + width, out);
            })) {
            return CosmicResult::kOutOfMemory;
        }
        if (!focused.Allocate(allocator, width, height)) return CosmicResult::kOutOfMemory;
        ImageF& f = focused.View();
        const int top = pyramid.Count();
        float sigmas[kMaxPyramidLevels + 1];
        for (int k = 0; k <= top; ++k) sigmas[k] = PyramidLevelSigma(k);
        const float falloff = std::max(0.0f, settings.focus_falloff);
        ParallelRows(runner, height, [&](int begin, int end, int) {
            for (int cy = begin; cy < end; ++cy) {
                const float y = (static_cast<float>(canvas_top + cy) + 0.5f) * render.to_full_y - settings.focus_y;
                const PixelF* in = b.Row(cy);
                PixelF* out = f.Row(cy);
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
                        amount = Smoothstep(d / falloff);
                    }
                    const float sigma = defocus * amount;
                    if (sigma <= 0.02f || top == 0) {
                        out[cx] = in[cx];
                        continue;
                    }
                    int k = 0;
                    while (k < top && sigmas[k + 1] <= sigma) ++k;
                    if (k >= top) {
                        out[cx] = SampleLevel(pyramid.Level(top), top, static_cast<float>(cx), static_cast<float>(cy));
                        continue;
                    }
                    const float t = (sigma - sigmas[k]) / (sigmas[k + 1] - sigmas[k]);
                    const PixelF lo = k == 0 ? in[cx]
                                             : SampleLevel(pyramid.Level(k), k, static_cast<float>(cx),
                                                           static_cast<float>(cy));
                    const PixelF hi =
                        SampleLevel(pyramid.Level(k + 1), k + 1, static_cast<float>(cx), static_cast<float>(cy));
                    out[cx] = PixelF{lo.a + (hi.a - lo.a) * t, lo.r + (hi.r - lo.r) * t, lo.g + (hi.g - lo.g) * t,
                                     lo.b + (hi.b - lo.b) * t};
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
            GlowThreshold threshold;
            threshold.level = std::max(0.0f, settings.glow_threshold);
            threshold.knee = std::clamp(settings.glow_softness, 0.0f, 1.0f) * std::max(threshold.level, 0.05f);
            Pyramid pyramid;
            if (!pyramid.Build(allocator, runner, width, height, plan.levels, [&](int y, PixelF* out) {
                    const PixelF* in = image.Row(y);
                    for (int x = 0; x < width; ++x) out[x] = ExtractHighlight(in[x], threshold);
                })) {
                return CosmicResult::kOutOfMemory;
            }
            if (!CollapsePyramid(allocator, runner, pyramid, plan.weights, 1, plan.levels, &glow)) {
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
            if (!pyramid.Build(allocator, runner, width, height, plan.levels, [&](int y, PixelF* out) {
                    const PixelF* in = image.Row(y);
                    std::copy(in, in + width, out);
                })) {
                return CosmicResult::kOutOfMemory;
            }
            if (!CollapsePyramid(allocator, runner, pyramid, plan.weights, 1, plan.levels, &diffusion)) {
                return CosmicResult::kOutOfMemory;
            }
        }
    }

    // --- 4. Composite into the host buffer -----------------------------------
    OutputContext ctx;
    ctx.transfer = &transfer;
    ctx.depth = dest.depth;
    ctx.protect = settings.highlight_protection > 0.0f;
    ctx.protection_knee = 1.0f - 0.6f * std::clamp(settings.highlight_protection, 0.0f, 1.0f);
    ctx.grain = std::clamp(settings.grain, 0.0f, 1.0f);
    ctx.grain_inv_size = 1.0f / std::max(0.05f, settings.grain_size);
    ctx.grain_seed = Hash32(settings.grain_seed ^ 0x27d4eb2fu);
    ctx.to_full_x = render.to_full_x;
    ctx.to_full_y = render.to_full_y;
    const float intensity = std::max(0.0f, settings.glow_intensity);
    const bool dither = dest.depth != PixelDepth::kFloat32;

    const BsplineRowSampler* glow_sampler = nullptr;
    const BsplineRowSampler* diffusion_sampler = nullptr;
    // Constructed only when their image exists.
    std::vector<BsplineRowSampler> samplers;
    samplers.reserve(2);
    if (glow.Valid()) {
        samplers.emplace_back(glow.View(), 1, width);
        glow_sampler = &samplers.back();
    }
    if (diffusion.Valid()) {
        samplers.emplace_back(diffusion.View(), 1, width);
        diffusion_sampler = &samplers.back();
    }

    ParallelRows(runner, dest.height, [&](int begin, int end, int) {
        std::vector<PixelF> glow_row(glow_sampler ? static_cast<std::size_t>(width) : 0);
        std::vector<PixelF> diffusion_row(diffusion_sampler ? static_cast<std::size_t>(width) : 0);
        std::vector<PixelF> scratch(static_cast<std::size_t>(width));
        for (int y = begin; y < end; ++y) {
            const int cy = y + pad;
            if (glow_sampler) glow_sampler->SampleRow(cy, scratch.data(), glow_row.data());
            if (diffusion_sampler) diffusion_sampler->SampleRow(cy, scratch.data(), diffusion_row.data());
            const PixelF* in = image.Row(cy);
            void* out_row = dest.Row(y);
            const int ly = render.dest_top + y;
            for (int x = 0; x < dest.width; ++x) {
                const int cx = x + pad;
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
                const PixelF encoded = FinishPixel(c, ctx, lx, ly);
                // Triangular dither of one step decorrelates the quantisation error
                // from the signal, so an 8-bit ramp shows noise instead of bands.
                const float d = dither ? TriangularNoise(lx, ly, 0x51ed270bu) : 0.0f;
                StorePixel(out_row, x, encoded, ctx.depth, d);
            }
        }
    });

    return CosmicResult::kOk;
}

}  // namespace cosmic
