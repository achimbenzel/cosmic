#pragma once

#include <functional>
#include <vector>

#include "Allocator.h"
#include "Shared.h"
#include "ImageF.h"
#include "TaskRunner.h"

namespace cosmic {

constexpr int kMaxPyramidLevels = 14;

// Blur, in level-0 pixels, of level k when it is reconstructed at level 0 with
// the cubic B-spline. Level 0 itself is exact (0).
float PyramidLevelSigma(int level);

// Downsampling pyramid. Level 0 is supplied row by row by the caller and not
// stored; levels 1..count are halved in each direction with a six-tap binomial
// filter centred between the pairs it merges, so level k's grid is exactly
// level 0's scaled by 2^-k. Outside the image is zero, or the edge carried on.
class Pyramid {
public:
    using RowFn = std::function<void(int y, PixelF* out)>;

    // `level0_row` fills row y of the level-0 image (width values) and must be
    // safe to call from several threads.
    bool Build(Allocator& allocator, TaskRunner& runner, int width, int height, int levels,
               const RowFn& level0_row, BorderMode border = BorderMode::kZero);

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
                     int first, int last, OwnedImageF* out, BorderMode border = BorderMode::kZero);

// Evaluates an image that lives on a grid 2^scale_log2 coarser than the target
// at every pixel of a target row, with the cubic B-spline.
class BsplineRowSampler {
public:
    BsplineRowSampler(const ImageF& source, int scale_log2, int target_width,
                      BorderMode border = BorderMode::kZero);

    // `scratch` must hold source.width pixels. Fills out[x_begin, x_end); the
    // default is the whole row.
    void SampleRow(int y, PixelF* scratch, PixelF* out, int x_begin = 0, int x_end = -1) const;

private:
    const ImageF& source_;
    float inv_scale_ = 1.0f;
    int target_width_ = 0;
    BorderMode border_ = BorderMode::kZero;
    std::vector<int> tap_index_;    // 4 per target column
    std::vector<float> tap_weight_; // 4 per target column
};

// Samples level k (k >= 1) at a level-0 pixel centre.
PixelF SampleLevel(const ImageF& level, int k, float x0, float y0, BorderMode border = BorderMode::kZero);

}  // namespace cosmic
