// CUDA kernels for Cosmic Gradient.
//
// tools/build_ptx.py compiles this file to PTX with NVRTC; the plug-in embeds
// the PTX and the NVIDIA driver compiles it for whatever GPU is installed. The
// tests compile the same file as C++ and run it through an emulator, so every
// kernel here is exercised on the CPU against the CPU renderer.
//
// Each kernel mirrors one stage of RenderCosmic in core/CosmicPipeline.cpp and
// sums in the same order, so the two agree to rounding.

#include "core/Shared.h"
#include "gpu/KernelParams.h"

#if defined(__CUDACC__)
#define COSMIC_KERNEL(name, Params) extern "C" __global__ void name(const Params p)
#define COSMIC_X static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)
#define COSMIC_Y static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y)
#else
#define COSMIC_KERNEL(name, Params) inline void name(const Params& p, int cosmic_x, int cosmic_y)
#define COSMIC_X cosmic_x
#define COSMIC_Y cosmic_y
#endif

namespace cosmic {
namespace kernels {

template <typename T>
COSMIC_HD T* Ptr(DevicePtr p) {
    return reinterpret_cast<T*>(p);
}

COSMIC_HD PixelF ZeroPixel() {
    PixelF p;
    p.a = p.r = p.g = p.b = 0.0f;
    return p;
}

COSMIC_HD void Madd(PixelF& acc, const PixelF& p, float w) {
    acc.a += p.a * w;
    acc.r += p.r * w;
    acc.g += p.g * w;
    acc.b += p.b * w;
}

// Samples an image that lives on a grid 2^scale_log2 coarser than the canvas
// at canvas pixel (x, y) with the cubic B-spline: each column vertically, then
// across, as BsplineRowSampler does.
COSMIC_HD PixelF SampleScaled(const PixelF* image, int w, int h, int scale_log2, int x, int y, BorderMode border) {
    const float inv = 1.0f / static_cast<float>(1 << scale_log2);
    const float sy = (static_cast<float>(y) + 0.5f) * inv - 0.5f;
    const float fly = floorf(sy);
    float wy[4];
    BsplineWeights(sy - fly, wy);
    const int j0 = static_cast<int>(fly) - 1;
    const float sx = (static_cast<float>(x) + 0.5f) * inv - 0.5f;
    const float flx = floorf(sx);
    float wx[4];
    BsplineWeights(sx - flx, wx);
    const int i0 = static_cast<int>(flx) - 1;

    PixelF acc = ZeroPixel();
    for (int s = 0; s < 4; ++s) {
        const int i = BorderIndex(i0 + s, w, border);
        if (i < 0) continue;
        PixelF column = ZeroPixel();
        for (int t = 0; t < 4; ++t) {
            const int j = BorderIndex(j0 + t, h, border);
            if (j < 0) continue;
            Madd(column, image[j * w + i], wy[t]);
        }
        Madd(acc, column, wx[s]);
    }
    return acc;
}

// The layer's alpha for MeasureRow.
struct FrameAlpha {
    const FrameBGRA* data;
    int width;
    int height;
    int pitch;
    COSMIC_HD float At(int x, int y) const {
        if (x < 0 || y < 0 || x >= width || y >= height) return 0.0f;
        return Clampf(data[y * pitch + x].a, 0.0f, 1.0f);
    }
};

COSMIC_HD PixelF FromFrame(const FrameBGRA& f) {
    PixelF p;
    p.a = f.a;
    p.r = f.r;
    p.g = f.g;
    p.b = f.b;
    return p;
}

}  // namespace kernels
}  // namespace cosmic

// ---------------------------------------------------------------------------
// The layer's shape, measured row by row: one thread per row.
// ---------------------------------------------------------------------------
COSMIC_KERNEL(CosmicRowStats, cosmic::RowStatsParams) {
    using namespace cosmic;
    const int y = COSMIC_X;
    (void)COSMIC_Y;  // a one-dimensional launch
    if (y >= p.height) return;
    kernels::FrameAlpha alpha;
    alpha.data = kernels::Ptr<const FrameBGRA>(p.source);
    alpha.width = p.width;
    alpha.height = p.height;
    alpha.pitch = p.pitch;
    kernels::Ptr<RowStats>(p.out)[y] =
        MeasureRow(alpha, y, p.width, p.visible, p.measure_outline, p.to_full_x, p.to_full_y);
}

// ---------------------------------------------------------------------------
// Turbulence grid nodes.
// ---------------------------------------------------------------------------
COSMIC_KERNEL(CosmicWarpGrid, cosmic::WarpGridParams) {
    using namespace cosmic;
    const int i = COSMIC_X;
    const int j = COSMIC_Y;
    if (i >= p.lattice.nx || j >= p.lattice.ny) return;
    WarpNode(i, j, p.lattice, p.amount, p.evolution, p.fbm, p.fbm2,
             kernels::Ptr<float>(p.out) + (j * p.lattice.nx + i) * 2);
}

// ---------------------------------------------------------------------------
// The shape the Bulge relief is raised from, on the canvas.
// ---------------------------------------------------------------------------
COSMIC_KERNEL(CosmicShape, cosmic::ShapeParams) {
    using namespace cosmic;
    const int cx = COSMIC_X;
    const int cy = COSMIC_Y;
    if (cx >= p.width || cy >= p.height) return;
    float a = 0.0f;
    const int sy = p.canvas_top + cy - p.source_top;
    const int sx = p.canvas_left + cx - p.source_left;
    if (sy >= 0 && sy < p.source_height && sx >= 0 && sx < p.source_width) {
        a = kernels::Ptr<const FrameBGRA>(p.source)[sy * p.source_pitch + sx].a;
    }
    kernels::Ptr<float>(p.out)[cy * p.width + cx] = ShapeValue(a, p.invert);
}

// ---------------------------------------------------------------------------
// One pass of a level of the relief's pyramid.
// ---------------------------------------------------------------------------
COSMIC_KERNEL(CosmicSmooth, cosmic::SmoothParams) {
    using namespace cosmic;
    const int x = COSMIC_X;
    const int y = COSMIC_Y;
    if (x >= p.width || y >= p.height) return;
    kernels::Ptr<float>(p.dst)[y * p.width + x] =
        SmoothAt(kernels::Ptr<const float>(p.src), p.width, p.height, x, y, p.step, p.vertical,
                 static_cast<BorderMode>(p.border));
}

// ---------------------------------------------------------------------------
// The Bulge relief: lookup offsets and palette shifts.
// ---------------------------------------------------------------------------
COSMIC_KERNEL(CosmicRelief, cosmic::ReliefParams) {
    using namespace cosmic;
    const int cx = COSMIC_X;
    const int cy = COSMIC_Y;
    if (cx >= p.width || cy >= p.height) return;
    float dx = 0.0f;
    float dy = 0.0f;
    float lift = 0.0f;
    ReliefPixel(kernels::Ptr<const float>(p.lo), kernels::Ptr<const float>(p.hi), p.mix, p.domain_width,
                p.domain_height, cx - p.domain_left, cy - p.domain_top, p.shape, &dx, &dy, &lift);
    PixelF o;
    o.a = lift;
    o.r = dx;
    o.g = dy;
    o.b = 0.0f;
    kernels::Ptr<PixelF>(p.out)[cy * p.width + cx] = o;
}

// ---------------------------------------------------------------------------
// The gradient, matted and blended with the layer, in linear light.
// ---------------------------------------------------------------------------
COSMIC_KERNEL(CosmicBase, cosmic::BaseParams) {
    using namespace cosmic;
    const int cx = COSMIC_X;
    const int cy = COSMIC_Y;
    if (cx >= p.width || cy >= p.height) return;
    const ExactSrgb srgb;

    const int ly = p.canvas_top + cy;
    PixelF src = kernels::ZeroPixel();
    if (p.source != 0) {
        const int sy = ly - p.source_top;
        const int sx = p.canvas_left + cx - p.source_left;
        if (sy >= 0 && sy < p.source_height && sx >= 0 && sx < p.source_width) {
            const FrameBGRA f = kernels::Ptr<const FrameBGRA>(p.source)[sy * p.source_pitch + sx];
            src = LinearizePremultiplied(kernels::FromFrame(f), srgb, p.decode_srgb == 0);
        }
    }

    PixelF* out = kernels::Ptr<PixelF>(p.out) + cy * p.width + cx;
    const float matte = MatteValue(static_cast<MatteMode>(p.matte), src.a);
    const float keep = 1.0f - p.opacity;
    if (matte <= kTransparent) {
        PixelF o;
        o.a = src.a * keep;
        o.r = src.r * keep;
        o.g = src.g * keep;
        o.b = src.b * keep;
        *out = o;
        return;
    }

    float x = (static_cast<float>(p.canvas_left + cx) + 0.5f) * p.to_full_x;
    float y = (static_cast<float>(ly) + 0.5f) * p.to_full_y;
    if (p.warp != 0) {
        const WarpLattice& l = p.lattice;
        float wx = 0.0f;
        float wy = 0.0f;
        SampleWarp(kernels::Ptr<const float>(p.warp), l.nx, l.ny,
                   WarpGridCoordinate(cx, p.canvas_left, p.to_full_x, l.anchor_x, l.scale, l.origin_i),
                   WarpGridCoordinate(cy, p.canvas_top, p.to_full_y, l.anchor_y, l.scale, l.origin_j), &wx, &wy);
        x += wx;
        y += wy;
    }
    float lift = 0.0f;
    if (p.relief != 0) {
        const PixelF rv = kernels::Ptr<const PixelF>(p.relief)[cy * p.width + cx];
        x += rv.r;
        y += rv.g;
        lift = rv.a;
    }
    Rgb color = SampleLut(kernels::Ptr<const Rgb>(p.lut), FieldValue(p.field, x, y, lift));
    if (static_cast<BlendMode>(p.blend) != BlendMode::kNormal) {
        const float inv = src.a > kTransparent ? 1.0f / src.a : 0.0f;
        Rgb s;
        s.r = src.r * inv;
        s.g = src.g * inv;
        s.b = src.b * inv;
        color = BlendColor(static_cast<BlendMode>(p.blend), s, color, srgb);
    }
    const float m = matte * p.opacity;
    PixelF o;
    o.a = m + src.a * keep;
    o.r = color.r * m + src.r * keep;
    o.g = color.g * m + src.g * keep;
    o.b = color.b * m + src.b * keep;
    *out = o;
}

// ---------------------------------------------------------------------------
// Pyramid reduce, horizontal then vertical.
// ---------------------------------------------------------------------------
COSMIC_KERNEL(CosmicReduceH, cosmic::ReduceParams) {
    using namespace cosmic;
    const int i = COSMIC_X;
    const int y = COSMIC_Y;
    if (i >= p.dst_width || y >= p.src_height) return;
    const PixelF* row = kernels::Ptr<const PixelF>(p.src) + y * p.src_width;
    const BorderMode border = static_cast<BorderMode>(p.border);
    const int first = 2 * i - 2;
    PixelF acc = kernels::ZeroPixel();
    for (int m = 0; m < 6; ++m) {
        const int x = BorderIndex(first + m, p.src_width, border);
        if (x < 0) continue;
        const PixelF v = p.extract ? ExtractHighlight(row[x], p.threshold) : row[x];
        kernels::Madd(acc, v, ReduceTap(m));
    }
    kernels::Ptr<PixelF>(p.dst)[y * p.dst_width + i] = acc;
}

COSMIC_KERNEL(CosmicReduceV, cosmic::ReduceParams) {
    using namespace cosmic;
    const int i = COSMIC_X;
    const int j = COSMIC_Y;
    if (i >= p.dst_width || j >= p.dst_height) return;
    const PixelF* src = kernels::Ptr<const PixelF>(p.src);
    const BorderMode border = static_cast<BorderMode>(p.border);
    const int first = 2 * j - 2;
    PixelF acc = kernels::ZeroPixel();
    for (int m = 0; m < 6; ++m) {
        const int y = BorderIndex(first + m, p.src_height, border);
        if (y < 0) continue;
        kernels::Madd(acc, src[y * p.dst_width + i], ReduceTap(m));
    }
    kernels::Ptr<PixelF>(p.dst)[j * p.dst_width + i] = acc;
}

// ---------------------------------------------------------------------------
// Focus: each pixel mixes the two pyramid levels around its blur.
// ---------------------------------------------------------------------------
COSMIC_KERNEL(CosmicFocus, cosmic::FocusParams) {
    using namespace cosmic;
    const int cx = COSMIC_X;
    const int cy = COSMIC_Y;
    if (cx >= p.width || cy >= p.height) return;
    const PixelF* base = kernels::Ptr<const PixelF>(p.base);
    PixelF* out = kernels::Ptr<PixelF>(p.out) + cy * p.width + cx;

    const float y = (static_cast<float>(p.canvas_top + cy) + 0.5f) * p.to_full_y - p.focus_y;
    const float x = (static_cast<float>(p.canvas_left + cx) + 0.5f) * p.to_full_x - p.focus_x;
    const float d = sqrtf(x * x + y * y) - p.focus_radius;
    float amount;
    if (d <= 0.0f) {
        amount = 0.0f;
    } else if (p.falloff <= 0.0f) {
        amount = 1.0f;
    } else {
        amount = Smoothstep01(d / p.falloff);
    }
    const float sigma = p.defocus * amount;
    if (sigma <= 0.02f || p.top == 0) {
        *out = base[cy * p.width + cx];
        return;
    }
    int k = 0;
    while (k < p.top && p.sigmas[k + 1] <= sigma) ++k;
    const int hi = k + 1 < p.top ? k + 1 : p.top;
    const float t = k >= p.top ? 1.0f : (sigma - p.sigmas[k]) / (p.sigmas[k + 1] - p.sigmas[k]);
    const BorderMode border = static_cast<BorderMode>(p.border);
    const PixelF lo = k == 0 ? base[cy * p.width + cx]
                             : kernels::SampleScaled(kernels::Ptr<const PixelF>(p.levels[k - 1]), p.level_width[k - 1],
                                                     p.level_height[k - 1], k, cx, cy, border);
    const PixelF up = kernels::SampleScaled(kernels::Ptr<const PixelF>(p.levels[hi - 1]), p.level_width[hi - 1],
                                            p.level_height[hi - 1], hi, cx, cy, border);
    PixelF o;
    o.a = lo.a + (up.a - lo.a) * t;
    o.r = lo.r + (up.r - lo.r) * t;
    o.g = lo.g + (up.g - lo.g) * t;
    o.b = lo.b + (up.b - lo.b) * t;
    *out = o;
}

// ---------------------------------------------------------------------------
// One step of collapsing a pyramid.
// ---------------------------------------------------------------------------
COSMIC_KERNEL(CosmicCollapse, cosmic::CollapseParams) {
    using namespace cosmic;
    const int x = COSMIC_X;
    const int y = COSMIC_Y;
    if (x >= p.width || y >= p.height) return;
    const PixelF* level = kernels::Ptr<const PixelF>(p.level);
    PixelF* out = kernels::Ptr<PixelF>(p.out) + y * p.width + x;
    if (p.coarse == 0) {
        const PixelF in = level[y * p.width + x];
        PixelF o;
        o.a = in.a * p.weight;
        o.r = in.r * p.weight;
        o.g = in.g * p.weight;
        o.b = in.b * p.weight;
        *out = o;
        return;
    }
    PixelF o = kernels::SampleScaled(kernels::Ptr<const PixelF>(p.coarse), p.coarse_width, p.coarse_height, 1, x, y,
                                     static_cast<BorderMode>(p.border));
    if (p.weight != 0.0f) kernels::Madd(o, level[y * p.width + x], p.weight);
    *out = o;
}

// ---------------------------------------------------------------------------
// Diffusion, glow and the finish, into After Effects' frame.
// ---------------------------------------------------------------------------
COSMIC_KERNEL(CosmicComposite, cosmic::CompositeParams) {
    using namespace cosmic;
    const int x = COSMIC_X;
    const int y = COSMIC_Y;
    if (x >= p.width || y >= p.height) return;
    const BorderMode border = static_cast<BorderMode>(p.border);
    PixelF c = kernels::Ptr<const PixelF>(p.image)[y * p.width + x];
    if (p.diffusion != 0) {
        // The veil spreads light outwards and dims what it takes it from, but
        // never thins the shape itself.
        const PixelF d = kernels::SampleScaled(kernels::Ptr<const PixelF>(p.diffusion), p.diffusion_width,
                                               p.diffusion_height, 1, x, y, border);
        const float amount = p.diffusion_amount;
        c.a = Maxf(c.a, c.a + (d.a - c.a) * amount);
        c.r += (d.r - c.r) * amount;
        c.g += (d.g - c.g) * amount;
        c.b += (d.b - c.b) * amount;
    }
    if (p.glow != 0) {
        const PixelF g = kernels::SampleScaled(kernels::Ptr<const PixelF>(p.glow), p.glow_width, p.glow_height, 1, x,
                                               y, border);
        const float ga = Clampf(g.a * p.intensity, 0.0f, 1.0f);
        const float ca = Clampf(c.a, 0.0f, 1.0f);
        c.r += g.r * p.intensity;
        c.g += g.g * p.intensity;
        c.b += g.b * p.intensity;
        c.a = ca + ga * (1.0f - ca);
    }
    const ExactSrgb srgb;
    const PixelF o = FinishPixel(c, p.finish, srgb, p.dest_left + x, p.dest_top + y);
    FrameBGRA f;
    f.b = o.b;
    f.g = o.g;
    f.r = o.r;
    f.a = o.a;
    kernels::Ptr<FrameBGRA>(p.dest)[y * p.dest_pitch + x] = f;
}
