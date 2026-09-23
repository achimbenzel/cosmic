#pragma once

// Per-pixel maths shared by the CPU renderer and the CUDA kernels. This header
// compiles both as ordinary C++ and under NVRTC, so it uses only C maths
// functions and nothing from the standard library. Anything the two renderers
// must agree on lives here, so they cannot drift apart.

#if defined(__CUDACC__)
#define COSMIC_HD __host__ __device__ inline
#else
#include <math.h>
#define COSMIC_HD inline
#endif

namespace cosmic {

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

COSMIC_HD float Minf(float a, float b) { return a < b ? a : b; }
COSMIC_HD float Maxf(float a, float b) { return a > b ? a : b; }
COSMIC_HD float Clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
COSMIC_HD int Clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
COSMIC_HD float Fract(float v) { return v - floorf(v); }

COSMIC_HD float Smoothstep01(float x) {
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;
    return x * x * (3.0f - 2.0f * x);
}

COSMIC_HD int FastFloor(float v) {
    const int i = static_cast<int>(v);
    return v < static_cast<float>(i) ? i - 1 : i;
}

constexpr float kOpaque = 0.999f;
constexpr float kTransparent = 1.0e-6f;
constexpr float kTwoPi = 6.28318530717958647f;

// ---------------------------------------------------------------------------
// Enums. Their values are popup indices minus one, so they are part of the
// saved project format: append only.
// ---------------------------------------------------------------------------

enum class GradientType { kLinear = 0, kRadial = 1, kConic = 2, kDiamond = 3, kReflected = 4 };
enum class RepeatMode { kNone = 0, kRepeat = 1, kMirror = 2 };
enum class DepthShape { kDome = 0, kSphere = 1, kRidge = 2, kWave = 3, kBulge = 4 };
enum class MatteMode { kLayerAlpha = 0, kInvertedAlpha = 1, kFullFrame = 2 };
enum class BlendMode { kNormal = 0, kMultiply = 1, kScreen = 2, kOverlay = 3, kColor = 4 };

// What a blur sees outside the image: nothing (a layer on transparency), or
// the edge carried on (a frame filled edge to edge).
enum class BorderMode { kZero = 0, kClamp = 1 };

// ---------------------------------------------------------------------------
// Pixels and colour
// ---------------------------------------------------------------------------

// Channel order matches After Effects' ARGB pixel layout.
struct PixelF {
    float a, r, g, b;
};

struct Rgb {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

struct Lab {
    float l = 0.0f;
    float a = 0.0f;
    float b = 0.0f;
};

// sRGB transfer functions, extended symmetrically so HDR and negative values
// survive a round trip.
COSMIC_HD float SrgbToLinear(float c) {
    const float s = c < 0.0f ? -1.0f : 1.0f;
    const float a = fabsf(c);
    if (a <= 0.04045f) return s * a / 12.92f;
    return s * powf((a + 0.055f) / 1.055f, 2.4f);
}

COSMIC_HD float LinearToSrgb(float c) {
    const float s = c < 0.0f ? -1.0f : 1.0f;
    const float a = fabsf(c);
    if (a <= 0.0031308f) return s * a * 12.92f;
    return s * (1.055f * powf(a, 1.0f / 2.4f) - 0.055f);
}

// The exact transfer, for code that cannot use the CPU's lookup tables. The
// CPU passes its TransferFunction instead; both have Encode/Decode.
struct ExactSrgb {
    COSMIC_HD float Encode(float linear) const { return LinearToSrgb(linear); }
    COSMIC_HD float Decode(float encoded) const { return SrgbToLinear(encoded); }
};

// Oklab (Björn Ottosson, 2020). Blending in it keeps the lightness of a ramp
// even and avoids the grey dip that linear RGB puts between complementary
// colours, which is most of what makes a gradient read as "designed".
COSMIC_HD Lab LinearSrgbToOklab(const Rgb& c) {
    const float l = 0.4122214708f * c.r + 0.5363325363f * c.g + 0.0514459929f * c.b;
    const float m = 0.2119034982f * c.r + 0.6806995451f * c.g + 0.1073969566f * c.b;
    const float s = 0.0883024619f * c.r + 0.2817188376f * c.g + 0.6299787005f * c.b;
    const float l_ = cbrtf(l);
    const float m_ = cbrtf(m);
    const float s_ = cbrtf(s);
    Lab out;
    out.l = 0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_;
    out.a = 1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_;
    out.b = 0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_;
    return out;
}

COSMIC_HD Rgb OklabToLinearSrgb(const Lab& c) {
    const float l_ = c.l + 0.3963377774f * c.a + 0.2158037573f * c.b;
    const float m_ = c.l - 0.1055613458f * c.a - 0.0638541728f * c.b;
    const float s_ = c.l - 0.0894841775f * c.a - 1.2914855480f * c.b;
    const float l = l_ * l_ * l_;
    const float m = m_ * m_ * m_;
    const float s = s_ * s_ * s_;
    Rgb out;
    out.r = 4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    out.g = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
    out.b = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
    return out;
}

// Smooth shoulder: identity below `knee`, asymptotic to 1 above it, C1 at the
// join.
COSMIC_HD float SoftSaturate(float x, float knee) {
    if (!(x > knee)) return x < 0.0f ? 0.0f : x;
    const float head = 1.0f - knee;
    return knee + head * (1.0f - expf(-(x - knee) / head));
}

// Working-space pixel to linear light, premultiplied in and out.
template <typename Srgb>
COSMIC_HD PixelF LinearizePremultiplied(const PixelF& p, const Srgb& srgb, bool identity) {
    if (identity) return p;
    PixelF out;
    out.a = p.a;
    if (p.a >= kOpaque) {
        out.r = srgb.Decode(p.r);
        out.g = srgb.Decode(p.g);
        out.b = srgb.Decode(p.b);
        return out;
    }
    if (p.a <= kTransparent) {
        out.r = out.g = out.b = 0.0f;
        return out;
    }
    const float inv = 1.0f / p.a;
    out.r = srgb.Decode(p.r * inv) * p.a;
    out.g = srgb.Decode(p.g * inv) * p.a;
    out.b = srgb.Decode(p.b * inv) * p.a;
    return out;
}

COSMIC_HD float ScreenChannel(float a, float b) {
    const float kKnee = 0.75f;
    return a + b - SoftSaturate(a, kKnee) * SoftSaturate(b, kKnee);
}

template <typename Srgb>
COSMIC_HD float OverlayChannel(float base, float top, const Srgb& srgb) {
    const float a = Clampf(srgb.Encode(base), 0.0f, 1.0f);
    const float b = Clampf(srgb.Encode(top), 0.0f, 1.0f);
    const float o = a < 0.5f ? 2.0f * a * b : 1.0f - 2.0f * (1.0f - a) * (1.0f - b);
    return srgb.Decode(o);
}

// The gradient's colour once combined with the layer's own (unpremultiplied,
// linear) colour.
template <typename Srgb>
COSMIC_HD Rgb BlendColor(BlendMode mode, const Rgb& s, const Rgb& g, const Srgb& srgb) {
    Rgb out;
    switch (mode) {
        case BlendMode::kMultiply:
            out.r = s.r * g.r;
            out.g = s.g * g.g;
            out.b = s.b * g.b;
            return out;
        case BlendMode::kScreen:
            out.r = ScreenChannel(s.r, g.r);
            out.g = ScreenChannel(s.g, g.g);
            out.b = ScreenChannel(s.b, g.b);
            return out;
        case BlendMode::kOverlay:
            out.r = OverlayChannel(s.r, g.r, srgb);
            out.g = OverlayChannel(s.g, g.g, srgb);
            out.b = OverlayChannel(s.b, g.b, srgb);
            return out;
        case BlendMode::kColor: {
            // The layer's lightness with the gradient's hue and chroma: keeps
            // the shading of text bevels and footage.
            Rgb clamped;
            clamped.r = Maxf(0.0f, s.r);
            clamped.g = Maxf(0.0f, s.g);
            clamped.b = Maxf(0.0f, s.b);
            const Lab ls = LinearSrgbToOklab(clamped);
            Lab lg = LinearSrgbToOklab(g);
            lg.l = ls.l;
            const Rgb mixed = OklabToLinearSrgb(lg);
            out.r = Maxf(0.0f, mixed.r);
            out.g = Maxf(0.0f, mixed.g);
            out.b = Maxf(0.0f, mixed.b);
            return out;
        }
        case BlendMode::kNormal:
        default:
            return g;
    }
}

COSMIC_HD float MatteValue(MatteMode mode, float source_alpha) {
    switch (mode) {
        case MatteMode::kInvertedAlpha: return 1.0f - Clampf(source_alpha, 0.0f, 1.0f);
        case MatteMode::kFullFrame: return 1.0f;
        case MatteMode::kLayerAlpha:
        default: return Clampf(source_alpha, 0.0f, 1.0f);
    }
}

// ---------------------------------------------------------------------------
// Noise
// ---------------------------------------------------------------------------

// Integer hash (lowbias32 by Chris Wellons): cheap, and good enough that
// neighbouring pixels show no visible pattern.
COSMIC_HD unsigned int Hash32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

COSMIC_HD unsigned int Hash2(int x, int y, unsigned int seed) {
    return Hash32(static_cast<unsigned int>(x) * 0x8da6b343u ^ Hash32(static_cast<unsigned int>(y) ^ seed));
}

// Uniform in [0, 1).
COSMIC_HD float HashToUnit(unsigned int h) {
    return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}

// Triangular distribution in (-1, 1): the sum of two uniforms. Used for dither
// (it decorrelates the quantisation error from the signal) and for grain.
COSMIC_HD float TriangularNoise(int x, int y, unsigned int seed) {
    const unsigned int h = Hash2(x, y, seed);
    return HashToUnit(h) + HashToUnit(Hash32(h ^ 0x9e3779b9u)) - 1.0f;
}

// One corner of a 4D simplex. The gradient is one of the 32 edge midpoints of
// a hypercube: `gi >> 3` picks the axis that is zero, and the low three bits
// the signs of the other three, in axis order. That is the same table as
// Gustavson's reference, computed rather than looked up, so the GPU needs no
// constant memory.
COSMIC_HD float SimplexCorner(float x, float y, float z, float w, int gi) {
    float t = 0.6f - x * x - y * y - z * z - w * w;
    if (t <= 0.0f) return 0.0f;
    t *= t;
    const int zero = gi >> 3;
    const float s0 = (gi & 4) ? -1.0f : 1.0f;
    const float s1 = (gi & 2) ? -1.0f : 1.0f;
    const float s2 = (gi & 1) ? -1.0f : 1.0f;
    float dot;
    switch (zero) {
        case 0: dot = s0 * y + s1 * z + s2 * w; break;
        case 1: dot = s0 * x + s1 * z + s2 * w; break;
        case 2: dot = s0 * x + s1 * y + s2 * w; break;
        default: dot = s0 * x + s1 * y + s2 * z; break;
    }
    return t * t * dot;
}

COSMIC_HD int SimplexGradientIndex(int i, int j, int k, int l, unsigned int seed) {
    unsigned int h = static_cast<unsigned int>(i) * 0x8da6b343u;
    h ^= static_cast<unsigned int>(j) * 0xd8163841u;
    h ^= static_cast<unsigned int>(k) * 0xcb1ab31fu;
    h ^= static_cast<unsigned int>(l) * 0x165667b1u;
    return static_cast<int>(Hash32(h ^ seed) & 31u);
}

// 4D simplex noise (after Stefan Gustavson's public domain reference), in
// roughly [-1, 1]. Four dimensions let two of them trace a circle, so the
// noise evolves and returns exactly to where it started: a seamless loop.
COSMIC_HD float Simplex4(float x, float y, float z, float w, unsigned int seed) {
    const float kF4 = 0.309016994374947f;  // (sqrt(5) - 1) / 4
    const float kG4 = 0.138196601125011f;  // (5 - sqrt(5)) / 20
    const float s = (x + y + z + w) * kF4;
    const int i = FastFloor(x + s);
    const int j = FastFloor(y + s);
    const int k = FastFloor(z + s);
    const int l = FastFloor(w + s);
    const float t = static_cast<float>(i + j + k + l) * kG4;
    const float x0 = x - (static_cast<float>(i) - t);
    const float y0 = y - (static_cast<float>(j) - t);
    const float z0 = z - (static_cast<float>(k) - t);
    const float w0 = w - (static_cast<float>(l) - t);

    // Which of the 24 simplices we are in follows from the order of the
    // offsets' magnitudes.
    int rank_x = 0, rank_y = 0, rank_z = 0, rank_w = 0;
    if (x0 > y0) ++rank_x; else ++rank_y;
    if (x0 > z0) ++rank_x; else ++rank_z;
    if (x0 > w0) ++rank_x; else ++rank_w;
    if (y0 > z0) ++rank_y; else ++rank_z;
    if (y0 > w0) ++rank_y; else ++rank_w;
    if (z0 > w0) ++rank_z; else ++rank_w;

    const int i1 = rank_x >= 3, j1 = rank_y >= 3, k1 = rank_z >= 3, l1 = rank_w >= 3;
    const int i2 = rank_x >= 2, j2 = rank_y >= 2, k2 = rank_z >= 2, l2 = rank_w >= 2;
    const int i3 = rank_x >= 1, j3 = rank_y >= 1, k3 = rank_z >= 1, l3 = rank_w >= 1;

    const float x1 = x0 - i1 + kG4, y1 = y0 - j1 + kG4, z1 = z0 - k1 + kG4, w1 = w0 - l1 + kG4;
    const float x2 = x0 - i2 + 2.0f * kG4, y2 = y0 - j2 + 2.0f * kG4, z2 = z0 - k2 + 2.0f * kG4,
                w2 = w0 - l2 + 2.0f * kG4;
    const float x3 = x0 - i3 + 3.0f * kG4, y3 = y0 - j3 + 3.0f * kG4, z3 = z0 - k3 + 3.0f * kG4,
                w3 = w0 - l3 + 3.0f * kG4;
    const float x4 = x0 - 1.0f + 4.0f * kG4, y4 = y0 - 1.0f + 4.0f * kG4, z4 = z0 - 1.0f + 4.0f * kG4,
                w4 = w0 - 1.0f + 4.0f * kG4;

    float n = SimplexCorner(x0, y0, z0, w0, SimplexGradientIndex(i, j, k, l, seed));
    n += SimplexCorner(x1, y1, z1, w1, SimplexGradientIndex(i + i1, j + j1, k + k1, l + l1, seed));
    n += SimplexCorner(x2, y2, z2, w2, SimplexGradientIndex(i + i2, j + j2, k + k2, l + l2, seed));
    n += SimplexCorner(x3, y3, z3, w3, SimplexGradientIndex(i + i3, j + j3, k + k3, l + l3, seed));
    n += SimplexCorner(x4, y4, z4, w4, SimplexGradientIndex(i + 1, j + 1, k + 1, l + 1, seed));
    return 27.0f * n;
}

struct FbmSettings {
    float octaves = 3.0f;      // fractional: the last octave fades in
    float loop_radius = 0.6f;  // how far one full evolution turn travels
    unsigned int seed = 0;
};

// Fractal sum of Simplex4. `phase` is the evolution angle in radians; phase and
// phase + 2*pi give identical results.
COSMIC_HD float LoopingFbm(float x, float y, float phase, const FbmSettings& settings) {
    const float cz = cosf(phase) * settings.loop_radius;
    const float cw = sinf(phase) * settings.loop_radius;
    float octaves = Clampf(settings.octaves, 1.0f, 10.0f);

    float sum = 0.0f;
    float norm = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    for (int o = 0; octaves > 0.0f; ++o) {
        const float weight = octaves >= 1.0f ? 1.0f : octaves;
        // A per-octave shift keeps the octaves from sharing a lattice origin,
        // which would line their features up into a visible grid.
        const float shift = static_cast<float>(o) * 19.19f;
        const float n = Simplex4(x * frequency + shift, y * frequency - shift, cz * frequency, cw * frequency,
                                 settings.seed + static_cast<unsigned int>(o) * 0x632be5abu);
        sum += n * amplitude * weight;
        norm += amplitude * weight;
        amplitude *= 0.5f;
        frequency *= 2.0f;
        octaves -= 1.0f;
    }
    return norm > 0.0f ? sum / norm : 0.0f;
}

// ---------------------------------------------------------------------------
// Turbulence grid
// ---------------------------------------------------------------------------

// The displacement at grid node (i, j). Nodes start two steps before the
// canvas, so the cubic reconstruction has neighbours at the edges.
COSMIC_HD void WarpNode(int i, int j, int step, int canvas_left, int canvas_top, float to_full_x, float to_full_y,
                        float inv_size, float amount, float evolution, const FbmSettings& fbm,
                        const FbmSettings& fbm2, float* out) {
    const float py = (static_cast<float>(canvas_top + (j - 2) * step) + 0.5f) * to_full_y;
    const float px = (static_cast<float>(canvas_left + (i - 2) * step) + 0.5f) * to_full_x;
    const float qx = px * inv_size;
    const float qy = py * inv_size;
    out[0] = amount * LoopingFbm(qx, qy, evolution, fbm);
    out[1] = amount * LoopingFbm(qx + 31.416f, qy - 47.853f, evolution, fbm2);
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

// Cubic B-spline weights for a sample `f` of the way between taps 1 and 2 of
// taps 0..3. C2 continuous, so nothing reconstructed from a coarse level shows
// a kink, and it never rings.
COSMIC_HD void BsplineWeights(float f, float* w) {
    const float f2 = f * f;
    const float f3 = f2 * f;
    const float g = 1.0f - f;
    w[0] = g * g * g * (1.0f / 6.0f);
    w[1] = (3.0f * f3 - 6.0f * f2 + 4.0f) * (1.0f / 6.0f);
    w[2] = (-3.0f * f3 + 3.0f * f2 + 3.0f * f + 1.0f) * (1.0f / 6.0f);
    w[3] = f3 * (1.0f / 6.0f);
}

// Displacement at canvas pixel (x, y), reconstructed from the 4 x 4 nearest
// grid nodes with the cubic B-spline: each column vertically, then across, the
// same order the CPU's row-wise version sums in.
COSMIC_HD void SampleWarp(const float* grid, int nx, int ny, int step, int x, int y, float* out_x, float* out_y) {
    const float sy = static_cast<float>(y) / static_cast<float>(step);
    const float fly = floorf(sy);
    float wy[4];
    BsplineWeights(sy - fly, wy);
    const int j0 = static_cast<int>(fly) + 1;
    const float sx = static_cast<float>(x) / static_cast<float>(step);
    const float flx = floorf(sx);
    float wx[4];
    BsplineWeights(sx - flx, wx);
    const int i0 = static_cast<int>(flx) + 1;
    float ax = 0.0f;
    float ay = 0.0f;
    for (int s = 0; s < 4; ++s) {
        const int i = i0 + s < nx - 1 ? i0 + s : nx - 1;
        float cx = 0.0f;
        float cy = 0.0f;
        for (int t = 0; t < 4; ++t) {
            const int j = Clampi(j0 + t, 0, ny - 1);
            cx += grid[(j * nx + i) * 2] * wy[t];
            cy += grid[(j * nx + i) * 2 + 1] * wy[t];
        }
        ax += cx * wx[s];
        ay += cy * wx[s];
    }
    *out_x = ax;
    *out_y = ay;
}

// Binomial (1 5 10 10 5 1) / 32, the pyramid's reduce filter: variance 1.25 in
// the finer level's pixels, centred between the pair of pixels it merges.
COSMIC_HD float ReduceTap(int m) {
    switch (m) {
        case 0:
        case 5: return 1.0f / 32.0f;
        case 1:
        case 4: return 5.0f / 32.0f;
        default: return 10.0f / 32.0f;
    }
}

// Where a tap outside [0, size) reads from, or -1 for "nothing there".
COSMIC_HD int BorderIndex(int i, int size, BorderMode border) {
    if (i >= 0 && i < size) return i;
    if (border == BorderMode::kZero) return -1;
    return i < 0 ? 0 : size - 1;
}

// ---------------------------------------------------------------------------
// Gradient field
// ---------------------------------------------------------------------------

// Everything FieldValue needs, resolved to pixels on the host.
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

// Bulge: a lens over the depth centre that magnifies the field inside the
// radius (or pinches it, for negative depth), leaving it untouched at the rim.
// The strength is limited to keep the mapping one-to-one: past those limits
// the field would fold over itself.
COSMIC_HD void BulgePoint(const Field& f, float* x, float* y) {
    const float qx = (*x - f.depth_x) * f.inv_depth_radius;
    const float qy = (*y - f.depth_y) * f.inv_depth_radius;
    const float r2 = qx * qx + qy * qy;
    if (!(r2 < 1.0f)) return;
    const float strength = Clampf(f.depth, -1.2f, 0.95f);
    const float falloff = (1.0f - r2) * (1.0f - r2);
    const float scale = 1.0f - strength * falloff;
    *x = f.depth_x + (*x - f.depth_x) * scale;
    *y = f.depth_y + (*y - f.depth_y) * scale;
}

// The palette coordinate at a full-resolution position (after any turbulence
// displacement), with repeat applied.
COSMIC_HD float FieldValue(const Field& f, float x, float y) {
    if (f.depth != 0.0f && f.depth_shape == DepthShape::kBulge) BulgePoint(f, &x, &y);

    const float px = x - f.cx;
    const float py = y - f.cy;
    const float u = px * f.dx + py * f.dy;
    const float v = px * f.ex + py * f.ey;

    float t;
    switch (f.type) {
        case GradientType::kRadial:
            t = sqrtf(px * px + py * py) * f.inv_half_span;
            break;
        case GradientType::kConic:
            // Starts at the angle's direction and sweeps clockwise.
            t = atan2f(v, u) * (1.0f / kTwoPi);
            if (t < 0.0f) t += 1.0f;
            break;
        case GradientType::kDiamond:
            t = (fabsf(u) + fabsf(v)) * f.inv_half_span;
            break;
        case GradientType::kReflected:
            t = fabsf(u) * f.inv_half_span;
            break;
        case GradientType::kLinear:
        default:
            t = u * f.inv_span + 0.5f;
            break;
    }

    if (f.depth != 0.0f && f.depth_shape != DepthShape::kBulge) {
        const float qx = (x - f.depth_x) * f.inv_depth_radius;
        const float qy = (y - f.depth_y) * f.inv_depth_radius;
        float h = 0.0f;
        switch (f.depth_shape) {
            case DepthShape::kSphere: {
                const float r2 = qx * qx + qy * qy;
                h = r2 < 1.0f ? sqrtf(1.0f - r2) : 0.0f;
                break;
            }
            case DepthShape::kRidge: {
                const float w = qx * f.ex + qy * f.ey;
                const float w2 = w * w;
                h = w2 < 1.0f ? (1.0f - w2) * (1.0f - w2) : 0.0f;
                break;
            }
            case DepthShape::kWave:
                h = sinf(kTwoPi * (qx * f.ex + qy * f.ey));
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
            return Clampf(t, 0.0f, 1.0f);
    }
}

// Palette lookup table: kLutSize linear-light colours, t in [0, 1].
constexpr int kLutSize = 2048;

COSMIC_HD Rgb SampleLut(const Rgb* table, float t) {
    if (!(t > 0.0f)) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    const float scaled = t * static_cast<float>(kLutSize - 1);
    int i = static_cast<int>(scaled);
    if (i > kLutSize - 2) i = kLutSize - 2;
    const float f = scaled - static_cast<float>(i);
    const Rgb& a = table[i];
    const Rgb& b = table[i + 1];
    Rgb out;
    out.r = a.r + (b.r - a.r) * f;
    out.g = a.g + (b.g - a.g) * f;
    out.b = a.b + (b.b - a.b) * f;
    return out;
}

// ---------------------------------------------------------------------------
// Glow, grain and output
// ---------------------------------------------------------------------------

// Soft-knee highlight isolation on the pixel's own (unpremultiplied)
// brightness, so an anti-aliased edge emits in proportion to its coverage.
struct GlowThreshold {
    float level = 0.0f;
    float knee = 0.0f;
};

COSMIC_HD PixelF ExtractHighlight(const PixelF& p, const GlowThreshold& threshold) {
    PixelF zero;
    zero.a = zero.r = zero.g = zero.b = 0.0f;
    if (p.a <= kTransparent) return zero;
    const float inv = p.a >= kOpaque ? 1.0f : 1.0f / p.a;
    const float level = Maxf(p.r, Maxf(p.g, p.b)) * inv;
    if (level <= 0.0f) return zero;
    float above = level - threshold.level;
    if (threshold.knee > 0.0f) {
        float soft = Clampf(above + threshold.knee, 0.0f, 2.0f * threshold.knee);
        soft = soft * soft / (4.0f * threshold.knee);
        above = Maxf(soft, above);
    }
    if (above <= 0.0f) return zero;
    const float c = above / level;
    PixelF out;
    out.a = p.a * c;
    out.r = p.r * c;
    out.g = p.g * c;
    out.b = p.b * c;
    return out;
}

// Film grain: a hashed value per grain cell, interpolated, so grain larger
// than a pixel is soft rather than blocky. Keyed to full-resolution layer
// coordinates, so it holds still under downsampling and moving bounds.
COSMIC_HD float GrainValue(float x, float y, float inv_size, unsigned int seed) {
    const float gx = x * inv_size;
    const float gy = y * inv_size;
    const float fx = floorf(gx);
    const float fy = floorf(gy);
    const int ix = static_cast<int>(fx);
    const int iy = static_cast<int>(fy);
    const float tx = Smoothstep01(gx - fx);
    const float ty = Smoothstep01(gy - fy);
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

struct FinishParams {
    int protect = 0;
    float protection_knee = 1.0f;
    float grain = 0.0f;
    float grain_inv_size = 1.0f;
    unsigned int grain_seed = 0;
    int encode_srgb = 0;  // the working space is sRGB rather than linear
    float to_full_x = 1.0f;
    float to_full_y = 1.0f;
};

// Linear premultiplied light to the working space's encoding, with highlight
// protection and grain. `lx`, `ly` are the pixel in layer render coordinates.
// Returns a premultiplied pixel.
template <typename Srgb>
COSMIC_HD PixelF FinishPixel(const PixelF& c, const FinishParams& p, const Srgb& srgb, int lx, int ly) {
    PixelF out;
    const float a = Clampf(c.a, 0.0f, 1.0f);
    if (a <= kTransparent) {
        out.a = out.r = out.g = out.b = 0.0f;
        return out;
    }
    const float inv = 1.0f / a;
    float r = Maxf(0.0f, c.r * inv);
    float g = Maxf(0.0f, c.g * inv);
    float b = Maxf(0.0f, c.b * inv);

    if (p.protect) {
        // Rolls the brightest channel off and scales the others with it, so a
        // colour approaching the top of the range keeps its hue instead of
        // clipping channel by channel towards white.
        const float m = Maxf(r, Maxf(g, b));
        if (m > p.protection_knee) {
            const float scale = SoftSaturate(m, p.protection_knee) / m;
            r *= scale;
            g *= scale;
            b *= scale;
        }
    }

    out.a = a;
    if (p.grain > 0.0f) {
        // Grain lives in perceptual units, strongest in the midtones like film.
        float er = srgb.Encode(r);
        float eg = srgb.Encode(g);
        float eb = srgb.Encode(b);
        const float mean = Clampf((er + eg + eb) * (1.0f / 3.0f), 0.0f, 1.0f);
        const float response = 0.35f + 2.6f * mean * (1.0f - mean);
        const float n = GrainValue((static_cast<float>(lx) + 0.5f) * p.to_full_x,
                                   (static_cast<float>(ly) + 0.5f) * p.to_full_y, p.grain_inv_size, p.grain_seed) *
                        p.grain * 0.18f * response;
        er = Maxf(0.0f, er + n);
        eg = Maxf(0.0f, eg + n);
        eb = Maxf(0.0f, eb + n);
        if (p.encode_srgb) {
            out.r = er * a;
            out.g = eg * a;
            out.b = eb * a;
        } else {
            out.r = srgb.Decode(er) * a;
            out.g = srgb.Decode(eg) * a;
            out.b = srgb.Decode(eb) * a;
        }
        return out;
    }

    if (p.encode_srgb) {
        out.r = srgb.Encode(r) * a;
        out.g = srgb.Encode(g) * a;
        out.b = srgb.Encode(b) * a;
    } else {
        out.r = r * a;
        out.g = g * a;
        out.b = b * a;
    }
    return out;
}

}  // namespace cosmic
