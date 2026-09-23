#include "Pyramid.h"

#include <algorithm>
#include <cmath>

namespace cosmic {
namespace {

inline void Madd(PixelF& acc, const PixelF& p, float w) {
    acc.a += p.a * w;
    acc.r += p.r * w;
    acc.g += p.g * w;
    acc.b += p.b * w;
}

// dst[i] gathers src[2i - 2 .. 2i + 3] with the binomial ReduceTap weights.
void ReduceRow(const PixelF* src, int width, PixelF* dst, int half_width, BorderMode border) {
    for (int i = 0; i < half_width; ++i) {
        const int first = 2 * i - 2;
        PixelF acc{0.0f, 0.0f, 0.0f, 0.0f};
        if (first >= 0 && first + 5 < width) {
            const PixelF* s = src + first;
            for (int m = 0; m < 6; ++m) Madd(acc, s[m], ReduceTap(m));
        } else {
            for (int m = 0; m < 6; ++m) {
                const int x = BorderIndex(first + m, width, border);
                if (x >= 0) Madd(acc, src[x], ReduceTap(m));
            }
        }
        dst[i] = acc;
    }
}

// Vertical half of the reduce: tmp holds `height` horizontally reduced rows.
void ReduceColumns(const ImageF& tmp, int height, ImageF& dst, TaskRunner& runner, BorderMode border) {
    ParallelRows(runner, dst.height, [&](int begin, int end, int) {
        for (int j = begin; j < end; ++j) {
            PixelF* out = dst.Row(j);
            std::fill(out, out + dst.width, PixelF{0.0f, 0.0f, 0.0f, 0.0f});
            const int first = 2 * j - 2;
            for (int m = 0; m < 6; ++m) {
                const int y = BorderIndex(first + m, height, border);
                if (y < 0) continue;
                const PixelF* in = tmp.Row(y);
                const float w = ReduceTap(m);
                for (int i = 0; i < dst.width; ++i) Madd(out[i], in[i], w);
            }
        }
    });
}

}  // namespace

float PyramidLevelSigma(int level) {
    if (level <= 0) return 0.0f;
    const float p = std::ldexp(1.0f, 2 * level);  // 4^level
    return std::sqrt(1.25f * (p - 1.0f) / 3.0f + p / 3.0f);
}

int MaxUsefulLevels(int width, int height) {
    int levels = 0;
    while ((width > 1 || height > 1) && levels < kMaxPyramidLevels) {
        width = (width + 1) / 2;
        height = (height + 1) / 2;
        ++levels;
    }
    return levels;
}

bool Pyramid::Build(Allocator& allocator, TaskRunner& runner, int width, int height, int levels,
                    const RowFn& level0_row, BorderMode border) {
    count_ = 0;
    levels = std::min(levels, kMaxPyramidLevels);
    if (levels <= 0 || width <= 0 || height <= 0) return true;

    // Scratch for the horizontal pass, sized for the first (largest) level.
    const int first_half_width = (width + 1) / 2;
    OwnedImageF tmp;
    if (!tmp.Allocate(allocator, first_half_width, height)) return false;

    int w = width;
    int h = height;
    for (int k = 1; k <= levels; ++k) {
        const int hw = (w + 1) / 2;
        const int hh = (h + 1) / 2;
        if (!levels_[k - 1].Allocate(allocator, hw, hh)) return false;

        ImageF tmp_view = tmp.View();
        tmp_view.width = hw;
        tmp_view.height = h;

        if (k == 1) {
            ParallelRows(runner, h, [&](int begin, int end, int) {
                std::vector<PixelF> row(static_cast<std::size_t>(w));
                for (int y = begin; y < end; ++y) {
                    level0_row(y, row.data());
                    ReduceRow(row.data(), w, tmp_view.Row(y), hw, border);
                }
            });
        } else {
            const ImageF& src = levels_[k - 2].View();
            ParallelRows(runner, h, [&](int begin, int end, int) {
                for (int y = begin; y < end; ++y) ReduceRow(src.Row(y), w, tmp_view.Row(y), hw, border);
            });
        }
        ReduceColumns(tmp_view, h, levels_[k - 1].View(), runner, border);
        count_ = k;
        w = hw;
        h = hh;
    }
    return true;
}

BsplineRowSampler::BsplineRowSampler(const ImageF& source, int scale_log2, int target_width, BorderMode border)
    : source_(source), inv_scale_(std::ldexp(1.0f, -scale_log2)), target_width_(target_width), border_(border) {
    tap_index_.resize(static_cast<std::size_t>(target_width) * 4);
    tap_weight_.resize(static_cast<std::size_t>(target_width) * 4);
    for (int x = 0; x < target_width; ++x) {
        const float s = (static_cast<float>(x) + 0.5f) * inv_scale_ - 0.5f;
        const float fl = std::floor(s);
        float w[4];
        BsplineWeights(s - fl, w);
        const int i0 = static_cast<int>(fl) - 1;
        for (int t = 0; t < 4; ++t) {
            // A tap with nothing under it keeps a valid, monotonic index and
            // contributes nothing, so a range of columns maps to a range of taps.
            const int i = BorderIndex(i0 + t, source.width, border);
            tap_index_[static_cast<std::size_t>(x) * 4 + t] = i >= 0 ? i : Clampi(i0 + t, 0, source.width - 1);
            tap_weight_[static_cast<std::size_t>(x) * 4 + t] = i >= 0 ? w[t] : 0.0f;
        }
    }
}

void BsplineRowSampler::SampleRow(int y, PixelF* scratch, PixelF* out, int x_begin, int x_end) const {
    if (x_end < 0 || x_end > target_width_) x_end = target_width_;
    if (x_begin < 0) x_begin = 0;
    if (x_begin >= x_end) return;

    const float s = (static_cast<float>(y) + 0.5f) * inv_scale_ - 0.5f;
    const float fl = std::floor(s);
    float w[4];
    BsplineWeights(s - fl, w);
    const int j0 = static_cast<int>(fl) - 1;

    // Only the source columns the requested range reads.
    const int* idx = tap_index_.data();
    const int src_begin = idx[x_begin * 4];
    const int src_end = idx[(x_end - 1) * 4 + 3] + 1;

    std::fill(scratch + src_begin, scratch + src_end, PixelF{0.0f, 0.0f, 0.0f, 0.0f});
    for (int t = 0; t < 4; ++t) {
        const int j = BorderIndex(j0 + t, source_.height, border_);
        if (j < 0) continue;
        const PixelF* row = source_.Row(j);
        for (int i = src_begin; i < src_end; ++i) Madd(scratch[i], row[i], w[t]);
    }

    const float* wt = tap_weight_.data();
    for (int x = x_begin; x < x_end; ++x) {
        const int* ix = idx + x * 4;
        const float* wx = wt + x * 4;
        PixelF acc{0.0f, 0.0f, 0.0f, 0.0f};
        Madd(acc, scratch[ix[0]], wx[0]);
        Madd(acc, scratch[ix[1]], wx[1]);
        Madd(acc, scratch[ix[2]], wx[2]);
        Madd(acc, scratch[ix[3]], wx[3]);
        out[x] = acc;
    }
}

PixelF SampleLevel(const ImageF& level, int k, float x0, float y0, BorderMode border) {
    const float inv = std::ldexp(1.0f, -k);
    const float u = (x0 + 0.5f) * inv - 0.5f;
    const float v = (y0 + 0.5f) * inv - 0.5f;
    const float fu = std::floor(u);
    const float fv = std::floor(v);
    float wx[4];
    float wy[4];
    BsplineWeights(u - fu, wx);
    BsplineWeights(v - fv, wy);
    const int i0 = static_cast<int>(fu) - 1;
    const int j0 = static_cast<int>(fv) - 1;

    PixelF acc{0.0f, 0.0f, 0.0f, 0.0f};
    const bool interior = i0 >= 0 && j0 >= 0 && i0 + 3 < level.width && j0 + 3 < level.height;
    for (int t = 0; t < 4; ++t) {
        const int j = interior ? j0 + t : BorderIndex(j0 + t, level.height, border);
        if (j < 0) continue;
        const PixelF* row = level.Row(j);
        PixelF line{0.0f, 0.0f, 0.0f, 0.0f};
        if (interior) {
            Madd(line, row[i0], wx[0]);
            Madd(line, row[i0 + 1], wx[1]);
            Madd(line, row[i0 + 2], wx[2]);
            Madd(line, row[i0 + 3], wx[3]);
        } else {
            for (int s = 0; s < 4; ++s) {
                const int i = BorderIndex(i0 + s, level.width, border);
                if (i >= 0) Madd(line, row[i], wx[s]);
            }
        }
        Madd(acc, line, wy[t]);
    }
    return acc;
}

bool CollapsePyramid(Allocator& allocator, TaskRunner& runner, const Pyramid& pyramid, const float* weights,
                     int first, int last, OwnedImageF* out, BorderMode border) {
    if (first < 1 || last > pyramid.Count() || first > last) return false;

    OwnedImageF acc;
    {
        const ImageF& top = pyramid.Level(last);
        if (!acc.Allocate(allocator, top.width, top.height)) return false;
        ImageF& a = acc.View();
        const float w = weights[last];
        ParallelRows(runner, top.height, [&](int begin, int end, int) {
            for (int y = begin; y < end; ++y) {
                const PixelF* in = top.Row(y);
                PixelF* o = a.Row(y);
                for (int x = 0; x < top.width; ++x) {
                    o[x] = PixelF{in[x].a * w, in[x].r * w, in[x].g * w, in[x].b * w};
                }
            }
        });
    }

    for (int k = last - 1; k >= first; --k) {
        const ImageF& level = pyramid.Level(k);
        OwnedImageF next;
        if (!next.Allocate(allocator, level.width, level.height)) return false;
        ImageF& n = next.View();
        const ImageF& coarse = acc.View();
        const BsplineRowSampler sampler(coarse, 1, level.width, border);
        const float w = weights[k];
        ParallelRows(runner, level.height, [&](int begin, int end, int) {
            std::vector<PixelF> scratch(static_cast<std::size_t>(coarse.width));
            for (int y = begin; y < end; ++y) {
                PixelF* o = n.Row(y);
                sampler.SampleRow(y, scratch.data(), o);
                if (w != 0.0f) {
                    const PixelF* in = level.Row(y);
                    for (int x = 0; x < level.width; ++x) Madd(o[x], in[x], w);
                }
            }
        });
        // Hand the new accumulator over without copying; the old one is freed
        // when `next` goes out of scope.
        acc.Swap(next);
    }

    out->Swap(acc);
    return true;
}

}  // namespace cosmic
