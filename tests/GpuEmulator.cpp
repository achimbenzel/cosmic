#include "GpuEmulator.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

std::mutex g_atomic_mutex;

}  // namespace

// The kernels' atomics, for the one kernel (CosmicBounds) that uses them.
inline void EmulatorAtomicMin(int* ptr, int value) {
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    if (value < *ptr) *ptr = value;
}

inline void EmulatorAtomicMax(int* ptr, int value) {
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    if (value > *ptr) *ptr = value;
}

#include "gpu/CosmicKernels.cu"

namespace cosmic_test {
namespace {

using Runner = void (*)(const void* params, int x, int y);

template <typename Params, void (*Body)(const Params&, int, int)>
void Run(const void* params, int x, int y) {
    Body(*static_cast<const Params*>(params), x, y);
}

const std::map<std::string, Runner>& Kernels() {
    static const std::map<std::string, Runner> table = {
        {"CosmicBounds", &Run<cosmic::BoundsParams, CosmicBounds>},
        {"CosmicWarpGrid", &Run<cosmic::WarpGridParams, CosmicWarpGrid>},
        {"CosmicBase", &Run<cosmic::BaseParams, CosmicBase>},
        {"CosmicReduceH", &Run<cosmic::ReduceParams, CosmicReduceH>},
        {"CosmicReduceV", &Run<cosmic::ReduceParams, CosmicReduceV>},
        {"CosmicFocus", &Run<cosmic::FocusParams, CosmicFocus>},
        {"CosmicCollapse", &Run<cosmic::CollapseParams, CosmicCollapse>},
        {"CosmicComposite", &Run<cosmic::CompositeParams, CosmicComposite>},
    };
    return table;
}

}  // namespace

bool EmulateKernel(const char* name, int width, int height, const void* params) {
    const auto& table = Kernels();
    auto it = table.find(name);
    if (it == table.end()) return false;
    const Runner run = it->second;
    // A real launch covers whole 16 x 16 blocks, so threads past the edge run
    // too and must return early: emulate that as well.
    const int grid_w = (width + 15) / 16 * 16;
    const int grid_h = (height + 15) / 16 * 16;
    const int threads = std::max(1, std::min(8, static_cast<int>(std::thread::hardware_concurrency())));
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t]() {
            for (int y = t; y < grid_h; y += threads) {
                for (int x = 0; x < grid_w; ++x) run(params, x, y);
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    return true;
}

bool EmulatedDevice::Allocate(std::size_t bytes, cosmic::DevicePtr* out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (budget_ != 0 && live_bytes_ + bytes > budget_) return false;
    void* memory = std::malloc(bytes);
    if (memory == nullptr) return false;
    std::memset(memory, 0xFF, bytes);  // NaN in every float
    *out = reinterpret_cast<cosmic::DevicePtr>(memory);
    live_[*out] = bytes;
    live_bytes_ += bytes;
    peak_ = std::max(peak_, live_bytes_);
    ++allocations_;
    return true;
}

void EmulatedDevice::Free(cosmic::DevicePtr ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = live_.find(ptr);
    if (it == live_.end()) return;
    live_bytes_ -= it->second;
    live_.erase(it);
    std::free(reinterpret_cast<void*>(ptr));
    ++frees_;
}

bool EmulatedDevice::Upload(cosmic::DevicePtr dst, const void* src, std::size_t bytes) {
    std::memcpy(reinterpret_cast<void*>(dst), src, bytes);
    return true;
}

bool EmulatedDevice::Download(void* dst, cosmic::DevicePtr src, std::size_t bytes) {
    std::memcpy(dst, reinterpret_cast<const void*>(src), bytes);
    return true;
}

bool EmulatedDevice::Launch(cosmic::Kernel kernel, int width, int height, const void* params, std::size_t) {
    ++launches_;
    return EmulateKernel(cosmic::KernelName(kernel), width, height, params);
}

}  // namespace cosmic_test
