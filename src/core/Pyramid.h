#pragma once

#include <functional>
#include <vector>

#include "Allocator.h"
#include "ImageF.h"
#include "TaskRunner.h"

namespace cosmic {

constexpr int kMaxPyramidLevels = 14;

// Cubic B-spline weights for a sample `f` of the way between taps 1 and 2 of
// taps 0..3. C2 continuous, so nothing reconstructed from a coarse level shows
// a kink, and it never rings.
inline void BsplineWeights(float f, float (&w)[4]) {
    const float f2 = f * f;
    const float f3 = f2 * f;
    const float g = 1.0f - f;
    w[0] = g * g * g * (1.0f / 6.0f);
    w[1] = (3.0f * f3 - 6.0f * f2 + 4.0f) * (1.0f / 6.0f);
    w[2] = (-3.0f * f3 + 3.0f * f2 + 3.0f * f + 1.0f) * (1.0f / 6.0f);
    w[3] = f3 * (1.0f / 6.0f);
}

// Blur, in level-0 pixels, of level k when it is reconstructed at level 0 with
// the cubic B-spline. Level 0 itself is exact (0).
float PyramidLevelSigma(int level);

// Downsampling pyramid. Level 0 is supplied row by row by the caller and not
// stored; levels 1..count are halved in each direction with a six-tap binomial
// filter centred between the pairs it merges, so level k's grid is exactly
// level 0's scaled by 2^-k. Everything outside the image is zero.
class Pyramid {
public:
    using RowFn = std::function<void(int y, PixelF* out)>;

    // `level0_row` fills row y of the level-0 image (width values) and must be
    // safe to call from several threads.
    bool Build(Allocator& allocator, TaskRunner& runner, int width, int height, int levels,
               const RowFn& level0_row);

    int Count() const { return count_; }
    const ImageF& Level(int k) const { return levels_[k - 1].View(); }

private:
    OwnedImageF levels_[kMaxPyramidLevels];
    int count_ = 0;
};

// How many halvings bring a width x height image down to about one pixel.
int MaxUsefulLevels(int width, int height);

// Weighted sum of pyramid levels first..last, reconstructed onto the grid of
// level `first`. `weights[k]` is the weight of level k.
bool CollapsePyramid(Allocator& allocator, TaskRunner& runner, const Pyramid& pyramid, const float* weights,
                     int first, int last, OwnedImageF* out);

// Evaluates an image that lives on a grid 2^scale_log2 coarser than the target
// at every pixel of a target row, with the cubic B-spline.
class BsplineRowSampler {
public:
    BsplineRowSampler(const ImageF& source, int scale_log2, int target_width);

    // `scratch` must hold source.width pixels.
    void SampleRow(int y, PixelF* scratch, PixelF* out) const;

private:
    const ImageF& source_;
    float inv_scale_ = 1.0f;
    int target_width_ = 0;
    std::vector<int> tap_index_;    // 4 per target column
    std::vector<float> tap_weight_; // 4 per target column
};

// Samples level k (k >= 1) at a level-0 pixel centre, zero outside.
PixelF SampleLevel(const ImageF& level, int k, float x0, float y0);

}  // namespace cosmic
