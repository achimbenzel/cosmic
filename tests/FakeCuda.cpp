// A stand-in nvcuda.dll for testing the plug-in's CUDA path on a machine
// without an NVIDIA GPU. It implements the driver calls CudaDevice uses: device
// memory is host memory, and cuLaunchKernel runs the kernel through the CPU
// emulator. It also checks what a real driver would insist on - a context
// current for every call, PTX handed to the module loader, launches covering
// whole 16 x 16 blocks - and counts calls so the mock host can check them.

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "GpuEmulator.h"

#define FAKE_API extern "C" __declspec(dllexport)

namespace {

constexpr int kSuccess = 0;
constexpr int kInvalidValue = 1;
constexpr int kInvalidContext = 201;
constexpr int kInvalidImage = 200;
constexpr int kNotFound = 500;
constexpr int kLaunchFailed = 719;

struct FakeFunction {
    std::string name;
};

struct FakeModule {
    int unused = 0;
};

std::mutex g_mutex;
thread_local int t_context_depth = 0;
int g_modules_loaded = 0;
int g_modules_unloaded = 0;
int g_launches = 0;
int g_context_violations = 0;
int g_fail_launches = 0;

bool ContextCurrent() {
    if (t_context_depth > 0) return true;
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_context_violations;
    return false;
}

}  // namespace

FAKE_API int __stdcall cuInit(unsigned int) { return kSuccess; }

FAKE_API int __stdcall cuCtxPushCurrent_v2(void* context) {
    if (context == nullptr) return kInvalidContext;
    ++t_context_depth;
    return kSuccess;
}

FAKE_API int __stdcall cuCtxPopCurrent_v2(void** context) {
    if (t_context_depth <= 0) return kInvalidContext;
    --t_context_depth;
    if (context != nullptr) *context = reinterpret_cast<void*>(0x1);
    return kSuccess;
}

FAKE_API int __stdcall cuModuleLoadData(void** module, const void* image) {
    if (!ContextCurrent()) return kInvalidContext;
    const char* text = static_cast<const char*>(image);
    // What the plug-in hands over must be the PTX NVRTC produced.
    if (text == nullptr || std::strstr(text, ".version") == nullptr || std::strstr(text, ".target") == nullptr ||
        std::strstr(text, ".visible .entry CosmicComposite") == nullptr) {
        return kInvalidImage;
    }
    *module = new FakeModule();
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_modules_loaded;
    return kSuccess;
}

FAKE_API int __stdcall cuModuleUnload(void* module) {
    if (!ContextCurrent()) return kInvalidContext;
    delete static_cast<FakeModule*>(module);
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_modules_unloaded;
    return kSuccess;
}

FAKE_API int __stdcall cuModuleGetFunction(void** function, void* module, const char* name) {
    if (!ContextCurrent()) return kInvalidContext;
    if (module == nullptr || name == nullptr) return kInvalidValue;
    // Only names the emulator (and so the PTX) has.
    static const char* const kKnown[] = {"CosmicRowStats", "CosmicWarpGrid", "CosmicShape",   "CosmicSmooth",
                                         "CosmicRelief",   "CosmicBase",     "CosmicReduceH", "CosmicReduceV",
                                         "CosmicFocus",    "CosmicCollapse", "CosmicComposite"};
    for (const char* known : kKnown) {
        if (std::strcmp(known, name) == 0) {
            *function = new FakeFunction{name};  // lives as long as the process; a handful of bytes
            return kSuccess;
        }
    }
    return kNotFound;
}

FAKE_API int __stdcall cuLaunchKernel(void* function, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                                      unsigned int block_x, unsigned int block_y, unsigned int block_z,
                                      unsigned int shared_bytes, void* stream, void** params, void** extra) {
    (void)stream;
    if (!ContextCurrent()) return kInvalidContext;
    if (function == nullptr || params == nullptr || params[0] == nullptr || extra != nullptr || grid_z != 1 ||
        block_z != 1 || shared_bytes != 0 || block_x * block_y > 1024) {
        return kInvalidValue;
    }
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        ++g_launches;
        if (g_fail_launches > 0) {
            --g_fail_launches;
            return kLaunchFailed;
        }
    }
    const FakeFunction* f = static_cast<const FakeFunction*>(function);
    return cosmic_test::EmulateKernel(f->name.c_str(), static_cast<int>(grid_x * block_x),
                                      static_cast<int>(grid_y * block_y), params[0])
               ? kSuccess
               : kNotFound;
}

FAKE_API int __stdcall cuStreamSynchronize(void*) { return ContextCurrent() ? kSuccess : kInvalidContext; }

FAKE_API int __stdcall cuMemcpyHtoD_v2(unsigned long long dst, const void* src, size_t bytes) {
    if (!ContextCurrent()) return kInvalidContext;
    std::memcpy(reinterpret_cast<void*>(dst), src, bytes);
    return kSuccess;
}

FAKE_API int __stdcall cuMemcpyDtoH_v2(void* dst, unsigned long long src, size_t bytes) {
    if (!ContextCurrent()) return kInvalidContext;
    std::memcpy(dst, reinterpret_cast<const void*>(src), bytes);
    return kSuccess;
}

// --- test controls ---------------------------------------------------------

FAKE_API void FakeCudaStats(int* modules_loaded, int* modules_unloaded, int* launches, int* context_violations) {
    std::lock_guard<std::mutex> lock(g_mutex);
    *modules_loaded = g_modules_loaded;
    *modules_unloaded = g_modules_unloaded;
    *launches = g_launches;
    *context_violations = g_context_violations;
}

// Makes the next `count` launches fail, to exercise the CPU fallback.
FAKE_API void FakeCudaFailLaunches(int count) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_fail_launches = count;
}
