#pragma once

// CUDA for the GPU render path. The driver is loaded from nvcuda.dll at run
// time, so the plug-in has no link-time dependency on CUDA and loads fine on
// machines without an NVIDIA GPU; the kernels are PTX, compiled by the driver
// for whatever GPU is installed.

#include <cstddef>

#include "SdkIncludes.h"
#include "gpu/GpuPipeline.h"

namespace cosmic {

struct CudaApi;

// The driver API, or nullptr when there is no usable NVIDIA driver.
const CudaApi* LoadCudaApi();

// Per-device state made at PF_Cmd_GPU_DEVICE_SETUP: the kernels loaded into
// After Effects' CUDA context. Handed back by the host as gpu_data.
class CudaKernels {
public:
    // nullptr if the driver is missing or rejects the PTX; the effect then
    // declines the device and renders on the CPU.
    static CudaKernels* Create(void* context);
    ~CudaKernels();

    CudaKernels(const CudaKernels&) = delete;
    CudaKernels& operator=(const CudaKernels&) = delete;

    const CudaApi& api() const { return *api_; }
    void* context() const { return context_; }
    void* function(Kernel kernel) const { return functions_[static_cast<int>(kernel)]; }

private:
    CudaKernels() = default;

    const CudaApi* api_ = nullptr;
    void* context_ = nullptr;
    void* module_ = nullptr;
    void* functions_[static_cast<int>(Kernel::kCount)] = {};
};

// A GpuDevice on After Effects' CUDA context and stream. Memory comes from the
// host's GPU device suite, as the SDK requires. The context is current for the
// device's lifetime.
class CudaDevice final : public GpuDevice {
public:
    CudaDevice(const CudaKernels& kernels, void* stream, PF_ProgPtr effect_ref, PF_GPUDeviceSuite1* suite,
               A_u_long device_index);
    ~CudaDevice() override;

    CudaDevice(const CudaDevice&) = delete;
    CudaDevice& operator=(const CudaDevice&) = delete;

    bool Valid() const { return pushed_; }

    bool Allocate(std::size_t bytes, DevicePtr* out) override;
    void Free(DevicePtr ptr) override;
    bool Upload(DevicePtr dst, const void* src, std::size_t bytes) override;
    bool Download(void* dst, DevicePtr src, std::size_t bytes) override;
    bool Launch(Kernel kernel, int width, int height, const void* params, std::size_t params_size) override;
    bool Finish() override;

private:
    const CudaKernels& kernels_;
    void* stream_ = nullptr;
    PF_ProgPtr effect_ref_ = nullptr;
    PF_GPUDeviceSuite1* suite_ = nullptr;
    A_u_long device_index_ = 0;
    bool pushed_ = false;
};

}  // namespace cosmic
