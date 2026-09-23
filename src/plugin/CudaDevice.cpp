#include "CudaDevice.h"

#include "gpu/CosmicKernelsPtx.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace cosmic {

// The few driver API entry points used, declared here so no CUDA headers are
// needed to build. Signatures as in cuda.h; the _v2 names are the ones the
// driver exports for the 64-bit API.
typedef int CUresult;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef unsigned long long CUdeviceptr;

#if defined(_WIN32)
#define COSMIC_CUDAAPI __stdcall
#else
#define COSMIC_CUDAAPI
#endif

struct CudaApi {
    CUresult(COSMIC_CUDAAPI* cuInit)(unsigned int flags) = nullptr;
    CUresult(COSMIC_CUDAAPI* cuCtxPushCurrent)(CUcontext context) = nullptr;
    CUresult(COSMIC_CUDAAPI* cuCtxPopCurrent)(CUcontext* context) = nullptr;
    CUresult(COSMIC_CUDAAPI* cuModuleLoadData)(CUmodule* module, const void* image) = nullptr;
    CUresult(COSMIC_CUDAAPI* cuModuleUnload)(CUmodule module) = nullptr;
    CUresult(COSMIC_CUDAAPI* cuModuleGetFunction)(CUfunction* function, CUmodule module, const char* name) = nullptr;
    CUresult(COSMIC_CUDAAPI* cuLaunchKernel)(CUfunction function, unsigned int grid_x, unsigned int grid_y,
                                             unsigned int grid_z, unsigned int block_x, unsigned int block_y,
                                             unsigned int block_z, unsigned int shared_bytes, CUstream stream,
                                             void** params, void** extra) = nullptr;
    CUresult(COSMIC_CUDAAPI* cuStreamSynchronize)(CUstream stream) = nullptr;
    CUresult(COSMIC_CUDAAPI* cuMemcpyHtoD)(CUdeviceptr dst, const void* src, size_t bytes) = nullptr;
    CUresult(COSMIC_CUDAAPI* cuMemcpyDtoH)(void* dst, CUdeviceptr src, size_t bytes) = nullptr;
};

namespace {

constexpr CUresult kCudaSuccess = 0;
constexpr unsigned int kBlock = 16;

template <typename Fn>
bool Resolve(void* module, const char* name, Fn* out) {
#if defined(_WIN32)
    *out = reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module), name)));
    return *out != nullptr;
#else
    (void)module;
    (void)name;
    *out = nullptr;
    return false;
#endif
}

CudaApi* LoadOnce() {
#if defined(_WIN32)
    // After Effects has normally loaded the driver already when it renders
    // with CUDA; otherwise take it from the usual search path.
    HMODULE module = GetModuleHandleA("nvcuda.dll");
    if (module == nullptr) module = LoadLibraryA("nvcuda.dll");
    if (module == nullptr) return nullptr;
    static CudaApi api;
    bool ok = Resolve(module, "cuInit", &api.cuInit);
    ok = ok && Resolve(module, "cuCtxPushCurrent_v2", &api.cuCtxPushCurrent);
    ok = ok && Resolve(module, "cuCtxPopCurrent_v2", &api.cuCtxPopCurrent);
    ok = ok && Resolve(module, "cuModuleLoadData", &api.cuModuleLoadData);
    ok = ok && Resolve(module, "cuModuleUnload", &api.cuModuleUnload);
    ok = ok && Resolve(module, "cuModuleGetFunction", &api.cuModuleGetFunction);
    ok = ok && Resolve(module, "cuLaunchKernel", &api.cuLaunchKernel);
    ok = ok && Resolve(module, "cuStreamSynchronize", &api.cuStreamSynchronize);
    ok = ok && Resolve(module, "cuMemcpyHtoD_v2", &api.cuMemcpyHtoD);
    ok = ok && Resolve(module, "cuMemcpyDtoH_v2", &api.cuMemcpyDtoH);
    if (!ok || api.cuInit(0) != kCudaSuccess) return nullptr;
    return &api;
#else
    return nullptr;
#endif
}

// Makes a context current for a scope, as the GPU device suite asks CUDA
// users to do.
class ScopedContext {
public:
    ScopedContext(const CudaApi& api, void* context) : api_(api) {
        pushed_ = context != nullptr && api_.cuCtxPushCurrent(static_cast<CUcontext>(context)) == kCudaSuccess;
    }
    ~ScopedContext() {
        if (!pushed_) return;
        CUcontext popped = nullptr;
        api_.cuCtxPopCurrent(&popped);
    }
    bool ok() const { return pushed_; }

private:
    const CudaApi& api_;
    bool pushed_ = false;
};

}  // namespace

const CudaApi* LoadCudaApi() {
    static const CudaApi* api = LoadOnce();
    return api;
}

CudaKernels* CudaKernels::Create(void* context) {
    const CudaApi* api = LoadCudaApi();
    if (api == nullptr || context == nullptr) return nullptr;
    ScopedContext scope(*api, context);
    if (!scope.ok()) return nullptr;

    // The driver compiles the PTX for this GPU here (and caches the result),
    // so an unsupported driver shows up now rather than mid-render.
    CUmodule module = nullptr;
    if (api->cuModuleLoadData(&module, kCosmicKernelsPtx) != kCudaSuccess || module == nullptr) return nullptr;

    CudaKernels* kernels = new CudaKernels();
    kernels->api_ = api;
    kernels->context_ = context;
    kernels->module_ = module;
    for (int k = 0; k < static_cast<int>(Kernel::kCount); ++k) {
        CUfunction function = nullptr;
        if (api->cuModuleGetFunction(&function, module, KernelName(static_cast<Kernel>(k))) != kCudaSuccess ||
            function == nullptr) {
            delete kernels;
            return nullptr;
        }
        kernels->functions_[k] = function;
    }
    return kernels;
}

CudaKernels::~CudaKernels() {
    if (api_ == nullptr || module_ == nullptr) return;
    ScopedContext scope(*api_, context_);
    if (scope.ok()) api_->cuModuleUnload(static_cast<CUmodule>(module_));
}

CudaDevice::CudaDevice(const CudaKernels& kernels, void* stream, PF_ProgPtr effect_ref, PF_GPUDeviceSuite1* suite,
                       A_u_long device_index)
    : kernels_(kernels), stream_(stream), effect_ref_(effect_ref), suite_(suite), device_index_(device_index) {
    pushed_ = kernels_.api().cuCtxPushCurrent(static_cast<CUcontext>(kernels_.context())) == kCudaSuccess;
}

CudaDevice::~CudaDevice() {
    if (!pushed_) return;
    CUcontext popped = nullptr;
    kernels_.api().cuCtxPopCurrent(&popped);
}

bool CudaDevice::Allocate(std::size_t bytes, DevicePtr* out) {
    if (!pushed_ || suite_ == nullptr) return false;
    void* memory = nullptr;
    if (suite_->AllocateDeviceMemory(effect_ref_, device_index_, bytes, &memory) != PF_Err_NONE || memory == nullptr) {
        return false;
    }
    *out = reinterpret_cast<DevicePtr>(memory);
    return true;
}

void CudaDevice::Free(DevicePtr ptr) {
    if (suite_ == nullptr || ptr == 0) return;
    suite_->FreeDeviceMemory(effect_ref_, device_index_, reinterpret_cast<void*>(ptr));
}

bool CudaDevice::Upload(DevicePtr dst, const void* src, std::size_t bytes) {
    // Synchronous, so the host buffer may go away as soon as this returns.
    return pushed_ && kernels_.api().cuMemcpyHtoD(dst, src, bytes) == kCudaSuccess;
}

bool CudaDevice::Download(void* dst, DevicePtr src, std::size_t bytes) {
    if (!pushed_ || !Finish()) return false;
    return kernels_.api().cuMemcpyDtoH(dst, src, bytes) == kCudaSuccess;
}

bool CudaDevice::Launch(Kernel kernel, int width, int height, const void* params, std::size_t params_size) {
    (void)params_size;  // the kernel's signature fixes the size
    if (!pushed_) return false;
    void* function = kernels_.function(kernel);
    if (function == nullptr) return false;
    const unsigned int grid_x = (static_cast<unsigned int>(width) + kBlock - 1) / kBlock;
    const unsigned int grid_y = (static_cast<unsigned int>(height) + kBlock - 1) / kBlock;
    void* args[] = {const_cast<void*>(params)};
    return kernels_.api().cuLaunchKernel(static_cast<CUfunction>(function), grid_x, grid_y, 1, kBlock, kBlock, 1, 0,
                                         static_cast<CUstream>(stream_), args, nullptr) == kCudaSuccess;
}

bool CudaDevice::Finish() {
    return pushed_ && kernels_.api().cuStreamSynchronize(static_cast<CUstream>(stream_)) == kCudaSuccess;
}

}  // namespace cosmic
