#pragma once

// The GPU renderer: the same stages as RenderCosmic, run as the kernels in
// CosmicKernels.cu on a GpuDevice. The plug-in's device is CUDA; the tests'
// device runs the kernels on the CPU.

#include <cstddef>

#include "core/CosmicPipeline.h"
#include "gpu/KernelParams.h"

namespace cosmic {

enum class Kernel {
    kRowStats = 0,
    kWarpGrid,
    kShape,
    kSmooth,
    kRelief,
    kBase,
    kReduceH,
    kReduceV,
    kFocus,
    kCollapse,
    kComposite,
    kCount
};

// The kernel's entry point name in the PTX.
const char* KernelName(Kernel kernel);

class GpuDevice {
public:
    virtual ~GpuDevice() = default;

    virtual bool Allocate(std::size_t bytes, DevicePtr* out) = 0;
    virtual void Free(DevicePtr ptr) = 0;
    virtual bool Upload(DevicePtr dst, const void* src, std::size_t bytes) = 0;
    virtual bool Download(void* dst, DevicePtr src, std::size_t bytes) = 0;
    // Runs `kernel` with one thread per cell of a width x height grid, passing
    // it the parameter block.
    virtual bool Launch(Kernel kernel, int width, int height, const void* params, std::size_t params_size) = 0;
    // Waits for everything launched so far.
    virtual bool Finish() = 0;
};

// An After Effects GPU frame: premultiplied float BGRA, pitch in pixels.
struct GpuFrame {
    DevicePtr data = 0;
    int width = 0;
    int height = 0;
    int pitch = 0;
};

struct GpuRender {
    GpuFrame source;  // data 0 when there is no layer
    int source_left = 0;
    int source_top = 0;

    GpuFrame dest;
    int dest_left = 0;
    int dest_top = 0;

    float to_full_x = 1.0f;
    float to_full_y = 1.0f;
    float blur_scale = 1.0f;

    // For WorkingSpace::kAuto: whether the project renders in 32 bpc.
    bool float_project = true;
};

CosmicResult RenderCosmicGpu(const CosmicSettings& settings, const GpuRender& render, GpuDevice& device);

}  // namespace cosmic
