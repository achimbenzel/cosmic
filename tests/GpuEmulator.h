#pragma once

// Runs the CUDA kernels on the CPU, so the GPU pipeline can be tested on a
// machine without an NVIDIA GPU. "Device" memory is host memory, filled with
// NaN on allocation the way real device memory holds garbage: a kernel that
// reads something nobody wrote shows up as NaN in the output.

#include <cstddef>
#include <map>
#include <mutex>

#include "gpu/GpuPipeline.h"

namespace cosmic_test {

// Runs the kernel with this entry point name over a width x height grid.
// False for a name the PTX does not have.
bool EmulateKernel(const char* name, int width, int height, const void* params);

class EmulatedDevice final : public cosmic::GpuDevice {
public:
    bool Allocate(std::size_t bytes, cosmic::DevicePtr* out) override;
    void Free(cosmic::DevicePtr ptr) override;
    bool Upload(cosmic::DevicePtr dst, const void* src, std::size_t bytes) override;
    bool Download(void* dst, cosmic::DevicePtr src, std::size_t bytes) override;
    bool Launch(cosmic::Kernel kernel, int width, int height, const void* params, std::size_t params_size) override;
    bool Finish() override { return true; }

    // Fails allocations once this many bytes are live, to test running out.
    void SetBudget(std::size_t bytes) { budget_ = bytes; }

    int allocations() const { return allocations_; }
    int frees() const { return frees_; }
    int launches() const { return launches_; }
    std::size_t peak_bytes() const { return peak_; }

private:
    std::mutex mutex_;
    std::map<cosmic::DevicePtr, std::size_t> live_;
    std::size_t live_bytes_ = 0;
    std::size_t peak_ = 0;
    std::size_t budget_ = 0;
    int allocations_ = 0;
    int frees_ = 0;
    int launches_ = 0;
};

}  // namespace cosmic_test
