// Minimal After Effects stand-in: loads CosmicGradient.aex and drives the same
// command sequence the host does, so the plug-in layer can be exercised
// without launching After Effects.
//
// Usage: cosmic_mock_host <path to CosmicGradient.aex> [output directory]

#include <windows.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "TestSupport.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "AEConfig.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"
#include "AE_EffectPixelFormat.h"
#include "AE_EffectSuites.h"
#include "AE_Macros.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "core/Palette.h"
#include "core/Pixel.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wmultichar"
#endif

namespace {

using EffectMainFn = PF_Err (*)(PF_Cmd, PF_InData*, PF_OutData*, PF_ParamDef*[], PF_LayerDef*, void*);

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    if (!condition) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

// Indices must match ParamIndex in CosmicParams.h.
enum {
    kIndexPalette = 2,
    kIndexColor1 = 3,
    kIndexColor5 = 7,
    kIndexType = 12,
    kIndexFit = 13,
    kIndexDepthShape = 22,
    kIndexLoopWithAngle = 32,
    kIndexDefocus = 39,
    kIndexGlowIntensity = 42,
    kIndexGlowRadius = 43,
    kIndexGrain = 54,
    kIndexAnimateGrain = 56,
    kIndexMatte = 59,
    kIndexExpandBounds = 62,
    kIndexGpu = 68,
    kParamCountV10 = 67,  // the layout v1.0 shipped with; v1.1 only appends
    kParamCount = 70
};

constexpr A_long kExpectedOutFlags = PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_I_EXPAND_BUFFER | PF_OutFlag_NON_PARAM_VARY;
constexpr A_long kExpectedOutFlags2 = PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG | PF_OutFlag2_SUPPORTS_SMART_RENDER |
                                      PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
                                      PF_OutFlag2_SUPPORTS_QUERY_DYNAMIC_FLAGS | PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;

// ---------------------------------------------------------------------------
// Host state
// ---------------------------------------------------------------------------

struct MockHost {
    std::vector<PF_ParamDef> params;
    std::vector<std::string> popup_strings;  // copied, as the host does
    std::vector<void*> live_handles;
    int handles_created = 0;
    int handles_disposed = 0;

    PF_EffectWorld input_world = {};
    PF_EffectWorld output_world = {};
    PF_LRect layer_rect = {};
    int downsample = 1;
    bool input_checked_out = false;
    bool output_checked_out = false;
};

MockHost* g_host = nullptr;

PF_Err MockAddParam(PF_ProgPtr, PF_ParamIndex, PF_ParamDefPtr def) {
    g_host->params.push_back(*def);
    if (def->param_type == PF_Param_POPUP && def->u.pd.u.namesptr != nullptr) {
        g_host->popup_strings.push_back(def->u.pd.u.namesptr);
    }
    return PF_Err_NONE;
}

// Index 0 is the input layer, which the host owns; 1..n are the added params.
// Point values are scaled for the downsample factor, as After Effects does.
PF_Err MockCheckoutParam(PF_ProgPtr, PF_ParamIndex index, A_long, A_long, A_u_long, PF_ParamDef* param) {
    if (index <= 0 || static_cast<std::size_t>(index - 1) >= g_host->params.size()) {
        return PF_Err_BAD_CALLBACK_PARAM;
    }
    *param = g_host->params[static_cast<std::size_t>(index - 1)];
    if (param->param_type == PF_Param_POINT) {
        param->u.td.x_value /= g_host->downsample;
        param->u.td.y_value /= g_host->downsample;
    }
    return PF_Err_NONE;
}

PF_Err MockCheckinParam(PF_ProgPtr, PF_ParamDef*) { return PF_Err_NONE; }
PF_Err MockAbort(PF_ProgPtr) { return PF_Err_NONE; }
PF_Err MockProgress(PF_ProgPtr, A_long, A_long) { return PF_Err_NONE; }

int MockSprintf(char* buffer, const char* format, ...) {
    va_list args;
    va_start(args, format);
    const int written = vsprintf(buffer, format, args);
    va_end(args);
    return written;
}

PF_Handle MockNewHandle(A_HandleSize size) {
    void** block = static_cast<void**>(std::malloc(static_cast<std::size_t>(size) + sizeof(void*)));
    if (block == nullptr) return nullptr;
    block[0] = block + 1;
    ++g_host->handles_created;
    g_host->live_handles.push_back(block);
    return reinterpret_cast<PF_Handle>(block);
}

void* MockLockHandle(PF_Handle handle) {
    void** block = reinterpret_cast<void**>(handle);
    return block == nullptr ? nullptr : block[0];
}

void MockUnlockHandle(PF_Handle) {}

void MockDisposeHandle(PF_Handle handle) {
    if (handle == nullptr) return;
    void** block = reinterpret_cast<void**>(handle);
    for (std::size_t i = 0; i < g_host->live_handles.size(); ++i) {
        if (g_host->live_handles[i] == block) {
            g_host->live_handles.erase(g_host->live_handles.begin() + static_cast<std::ptrdiff_t>(i));
            ++g_host->handles_disposed;
            std::free(block);
            return;
        }
    }
    std::printf("FAIL: disposing a handle the host never created\n");
    ++g_failures;
}

A_HandleSize MockGetHandleSize(PF_Handle) { return 0; }
PF_Err MockResizeHandle(A_HandleSize, PF_Handle*) { return PF_Err_NONE; }

PF_HandleSuite1 g_handle_suite = {MockNewHandle,     MockLockHandle,    MockUnlockHandle,
                                  MockDisposeHandle, MockGetHandleSize, MockResizeHandle};

// Runs the iterations across real threads, the way the host's pool does.
PF_Err MockIterateGeneric(A_long iterations, void* refcon, PF_Err (*fn)(void*, A_long, A_long, A_long)) {
    const int hardware = static_cast<int>(std::thread::hardware_concurrency());
    const int workers = hardware > 0 ? hardware : 4;
    if (iterations == PF_Iterations_ONCE_PER_PROCESSOR) {
        for (int i = 0; i < workers; ++i) fn(refcon, i, i, workers);
        return PF_Err_NONE;
    }
    std::vector<std::thread> threads;
    const int used = workers < iterations ? workers : static_cast<int>(iterations);
    for (int t = 0; t < used; ++t) {
        threads.emplace_back([&, t]() {
            for (A_long i = t; i < iterations; i += used) fn(refcon, t, i, iterations);
        });
    }
    for (std::thread& thread : threads) thread.join();
    return PF_Err_NONE;
}

PF_Iterate8Suite1 g_iterate_suite = {};

// --- GPU: device memory is host memory; the fake nvcuda.dll runs kernels on it.

std::map<const PF_EffectWorld*, void*> g_gpu_world_data;  // GPU worlds' pixels
std::map<void*, std::size_t> g_device_live;
int g_device_allocations = 0;

PF_Err MockGetDeviceCount(PF_ProgPtr, A_u_long* count) {
    *count = 1;
    return PF_Err_NONE;
}

PF_Err MockGetDeviceInfo(PF_ProgPtr, A_u_long, PF_GPUDeviceInfo* info) {
    std::memset(info, 0, sizeof(*info));
    info->device_framework = PF_GPU_Framework_CUDA;
    info->compatibleB = TRUE;
    info->contextPV = reinterpret_cast<void*>(0xC0DE);  // any non-null CUcontext
    info->command_queuePV = nullptr;                    // the default stream
    return PF_Err_NONE;
}

PF_Err MockAllocateDeviceMemory(PF_ProgPtr, A_u_long, size_t bytes, void** memory) {
    void* block = std::malloc(bytes);
    if (block == nullptr) return PF_Err_OUT_OF_MEMORY;
    std::memset(block, 0xFF, bytes);  // garbage, as real device memory holds
    g_device_live[block] = bytes;
    ++g_device_allocations;
    *memory = block;
    return PF_Err_NONE;
}

PF_Err MockFreeDeviceMemory(PF_ProgPtr, A_u_long, void* memory) {
    if (g_device_live.erase(memory) == 0) {
        std::printf("FAIL: freeing device memory the host never allocated\n");
        ++g_failures;
        return PF_Err_BAD_CALLBACK_PARAM;
    }
    std::free(memory);
    return PF_Err_NONE;
}

PF_Err MockGetGPUWorldData(PF_ProgPtr, PF_EffectWorld* world, void** pixels) {
    auto it = g_gpu_world_data.find(world);
    if (it == g_gpu_world_data.end()) return PF_Err_BAD_CALLBACK_PARAM;
    *pixels = it->second;
    return PF_Err_NONE;
}

PF_GPUDeviceSuite1 g_gpu_suite = {};

PF_Err MockGetPixelFormat(const PF_EffectWorld* world, PF_PixelFormat* format) {
    *format = g_gpu_world_data.count(world) != 0 ? PF_PixelFormat_GPU_BGRA128 : PF_PixelFormat_ARGB128;
    return PF_Err_NONE;
}

PF_WorldSuite2 g_world_suite = {};

SPErr MockAcquireSuite(const char* name, int32_t version, const void** suite) {
    if (std::strcmp(name, kPFGPUDeviceSuite) == 0 && version == kPFGPUDeviceSuiteVersion1) {
        *suite = &g_gpu_suite;
        return kSPNoError;
    }
    if (std::strcmp(name, kPFWorldSuite) == 0 && version == kPFWorldSuiteVersion2) {
        *suite = &g_world_suite;
        return kSPNoError;
    }
    if (std::strcmp(name, kPFHandleSuite) == 0 && version == kPFHandleSuiteVersion1) {
        *suite = &g_handle_suite;
        return kSPNoError;
    }
    if (std::strcmp(name, kPFIterate8Suite) == 0 && version == kPFIterate8SuiteVersion1) {
        *suite = &g_iterate_suite;
        return kSPNoError;
    }
    *suite = nullptr;
    return kSPSuiteNotFoundError;
}

SPErr MockReleaseSuite(const char*, int32_t) { return kSPNoError; }

SPBasicSuite g_basic_suite = {};

PF_Err MockCheckoutLayer(PF_ProgPtr, PF_ParamIndex, A_long, const PF_RenderRequest* request, A_long, A_long,
                         A_u_long, PF_CheckoutResult* result) {
    std::memset(result, 0, sizeof(*result));
    result->max_result_rect = g_host->layer_rect;
    result->ref_width = g_host->layer_rect.right - g_host->layer_rect.left;
    result->ref_height = g_host->layer_rect.bottom - g_host->layer_rect.top;
    result->par.num = 1;
    result->par.den = 1;
    PF_LRect clipped = g_host->layer_rect;
    if (request != nullptr) {
        clipped.left = request->rect.left > clipped.left ? request->rect.left : clipped.left;
        clipped.top = request->rect.top > clipped.top ? request->rect.top : clipped.top;
        clipped.right = request->rect.right < clipped.right ? request->rect.right : clipped.right;
        clipped.bottom = request->rect.bottom < clipped.bottom ? request->rect.bottom : clipped.bottom;
    }
    if (clipped.right < clipped.left) clipped.right = clipped.left;
    if (clipped.bottom < clipped.top) clipped.bottom = clipped.top;
    result->result_rect = clipped;
    return PF_Err_NONE;
}

PF_Err MockGuidMixInPtr(PF_ProgPtr, A_u_long, const void*) { return PF_Err_NONE; }

PF_Err MockCheckoutLayerPixels(PF_ProgPtr, A_long, PF_EffectWorld** pixels) {
    g_host->input_checked_out = true;
    *pixels = &g_host->input_world;
    return PF_Err_NONE;
}

PF_Err MockCheckinLayerPixels(PF_ProgPtr, A_long) { return PF_Err_NONE; }

PF_Err MockCheckoutOutput(PF_ProgPtr, PF_EffectWorld** output) {
    g_host->output_checked_out = true;
    *output = &g_host->output_world;
    return PF_Err_NONE;
}

// ---------------------------------------------------------------------------

struct WorldStorage {
    std::vector<unsigned char> bytes;
};

void MakeWorld(PF_EffectWorld* world, WorldStorage* storage, int width, int height, short bitdepth,
               const PF_LRect& rect) {
    const int bytes_per_pixel = bitdepth == 8 ? 4 : (bitdepth == 16 ? 8 : 16);
    std::memset(world, 0, sizeof(*world));
    world->width = width;
    world->height = height;
    world->rowbytes = width * bytes_per_pixel;
    storage->bytes.assign(static_cast<std::size_t>(world->rowbytes) * static_cast<std::size_t>(height), 0);
    world->data = reinterpret_cast<PF_PixelPtr>(storage->bytes.data());
    world->world_flags = bitdepth == 16 ? PF_WorldFlag_DEEP : static_cast<PF_WorldFlags>(0);
    world->origin_x = rect.left;
    world->origin_y = rect.top;
    world->extent_hint.right = width;
    world->extent_hint.bottom = height;
    world->pix_aspect_ratio.num = 1;
    world->pix_aspect_ratio.den = 1;
}

// White shapes on transparency: a bar and a disc, standing in for text and a
// shape layer.
void FillTestScene(PF_EffectWorld* world, short bitdepth) {
    const float cx = world->width * 0.68f;
    const float cy = world->height * 0.5f;
    const float r = world->height * 0.28f;
    for (int y = 0; y < world->height; ++y) {
        unsigned char* row = reinterpret_cast<unsigned char*>(world->data) + static_cast<std::ptrdiff_t>(y) * world->rowbytes;
        for (int x = 0; x < world->width; ++x) {
            const bool bar = x > world->width / 8 && x < world->width / 2 && y > world->height / 3 &&
                             y < 2 * world->height / 3;
            const float dx = x + 0.5f - cx;
            const float dy = y + 0.5f - cy;
            const bool disc = dx * dx + dy * dy < r * r;
            const float a = (bar || disc) ? 1.0f : 0.0f;
            if (bitdepth == 8) {
                cosmic::Pixel8& p = reinterpret_cast<cosmic::Pixel8*>(row)[x];
                p.a = p.r = p.g = p.b = static_cast<unsigned char>(a * 255.0f);
            } else if (bitdepth == 16) {
                cosmic::Pixel16& p = reinterpret_cast<cosmic::Pixel16*>(row)[x];
                p.a = p.r = p.g = p.b = static_cast<unsigned short>(a * cosmic::kMaxChannel16);
            } else {
                cosmic::PixelF& p = reinterpret_cast<cosmic::PixelF*>(row)[x];
                p.a = p.r = p.g = p.b = a;
            }
        }
    }
}

void SetupInData(PF_InData* in_data, PF_OutData* out_data, PF_UtilCallbacks* utils, MockHost* host) {
    std::memset(in_data, 0, sizeof(*in_data));
    std::memset(out_data, 0, sizeof(*out_data));
    std::memset(utils, 0, sizeof(*utils));
    utils->ansi.sprintf = MockSprintf;
    in_data->inter.add_param = MockAddParam;
    in_data->inter.checkout_param = MockCheckoutParam;
    in_data->inter.checkin_param = MockCheckinParam;
    in_data->inter.abort = MockAbort;
    in_data->inter.progress = MockProgress;
    in_data->utils = reinterpret_cast<struct _PF_UtilCallbacks*>(utils);
    in_data->effect_ref = reinterpret_cast<PF_ProgPtr>(host);
    in_data->pica_basicP = &g_basic_suite;
    in_data->quality = PF_Quality_HI;
    in_data->version.major = PF_AE_PLUG_IN_VERSION;
    in_data->version.minor = PF_AE_PLUG_IN_SUBVERS;
    in_data->time_scale = 30;
    in_data->time_step = 1;
    in_data->downsample_x.num = 1;
    in_data->downsample_x.den = 1;
    in_data->downsample_y.num = 1;
    in_data->downsample_y.den = 1;
    in_data->pixel_aspect_ratio.num = 1;
    in_data->pixel_aspect_ratio.den = 1;
}

MockHost* NewInstance(EffectMainFn effect_main, PF_InData* in_data, PF_OutData* out_data, PF_UtilCallbacks* utils,
                      int width, int height) {
    MockHost* host = new MockHost();
    g_host = host;
    SetupInData(in_data, out_data, utils, host);
    in_data->width = width;
    in_data->height = height;

    PF_Err err = effect_main(PF_Cmd_GLOBAL_SETUP, in_data, out_data, nullptr, nullptr, nullptr);
    Check(err == PF_Err_NONE, "global setup succeeds");
    Check(out_data->out_flags == kExpectedOutFlags, "global out flags match the PiPL");
    Check(out_data->out_flags2 == kExpectedOutFlags2, "global out flags 2 match the PiPL");

    err = effect_main(PF_Cmd_PARAMS_SETUP, in_data, out_data, nullptr, nullptr, nullptr);
    Check(err == PF_Err_NONE, "params setup succeeds");
    Check(out_data->num_params == static_cast<A_long>(host->params.size()) + 1,
          "reported parameter count matches the parameters added");
    Check(out_data->num_params == kParamCount, "the v1.1 parameter layout");
    // Saved projects find values by these ids: every id equals its index, the
    // v1.0 ones unchanged and the new ones appended after them.
    bool ids_ok = host->params.size() + 1 == static_cast<std::size_t>(kParamCount);
    for (std::size_t i = 0; i < host->params.size(); ++i) {
        ids_ok = ids_ok && host->params[i].uu.id == static_cast<A_long>(i + 1);
    }
    Check(ids_ok, "parameter ids equal their indices, so v1.0 projects map onto v1.1");

    // After Effects turns a point's percentage default into layer pixels
    // when the effect is applied.
    for (PF_ParamDef& def : host->params) {
        if (def.param_type == PF_Param_POINT) {
            def.u.td.x_value = static_cast<PF_Fixed>(static_cast<double>(def.u.td.x_value) / 100.0 * width);
            def.u.td.y_value = static_cast<PF_Fixed>(static_cast<double>(def.u.td.y_value) / 100.0 * height);
        }
    }
    return host;
}

PF_ParamDef& Param(MockHost* host, int index) { return host->params[static_cast<std::size_t>(index - 1)]; }

struct RenderOptions {
    std::string label;
    short bitdepth = 8;
    int width = 480;
    int height = 270;
    int downsample = 1;
    bool expand_bounds = true;
    int matte = 1;
    int type = 1;
    float defocus = 0.0f;
    float glow_radius = 80.0f;
    bool animate_grain = false;
    float turbulence = -1.0f;  // percent; negative keeps the default
    int depth_shape = 0;       // 0 keeps the default
};

void ApplyOptions(MockHost* host, const RenderOptions& options);
void FillTestScene(PF_EffectWorld* world, short bitdepth);

bool RunRender(EffectMainFn effect_main, const RenderOptions& options, const std::string& out_dir,
               cosmic_test::TestImage* out_image = nullptr, PF_LRect* out_rect = nullptr) {
    PF_InData in_data;
    PF_OutData out_data;
    PF_UtilCallbacks utils;
    MockHost* host = NewInstance(effect_main, &in_data, &out_data, &utils, options.width, options.height);
    host->downsample = options.downsample;
    in_data.downsample_x.den = options.downsample;
    in_data.downsample_y.den = options.downsample;
    in_data.current_time = 12;

    ApplyOptions(host, options);

    const int w = options.width / options.downsample;
    const int h = options.height / options.downsample;
    host->layer_rect.left = 0;
    host->layer_rect.top = 0;
    host->layer_rect.right = w;
    host->layer_rect.bottom = h;

    PF_PreRenderInput pre_input = {};
    pre_input.bitdepth = options.bitdepth;
    pre_input.output_request.rect = host->layer_rect;
    pre_input.output_request.field = PF_Field_FRAME;
    pre_input.output_request.channel_mask = PF_ChannelMask_ARGB;
    PF_PreRenderOutput pre_output = {};
    PF_PreRenderCallbacks pre_callbacks = {MockCheckoutLayer, MockGuidMixInPtr};
    PF_PreRenderExtra pre_extra = {&pre_input, &pre_output, &pre_callbacks};

    PF_Err err = effect_main(PF_Cmd_SMART_PRE_RENDER, &in_data, &out_data, nullptr, nullptr, &pre_extra);
    Check(err == PF_Err_NONE, options.label + ": smart pre-render succeeds");
    Check(pre_output.pre_render_data != nullptr, options.label + ": pre-render data was allocated");

    const A_long result_width = pre_output.result_rect.right - pre_output.result_rect.left;
    const A_long result_height = pre_output.result_rect.bottom - pre_output.result_rect.top;
    if (options.expand_bounds && options.matte == 1) {
        Check(result_width > w && result_height > h, options.label + ": expanded bounds grow the result rect");
    } else {
        Check(result_width == w && result_height == h, options.label + ": result rect matches the layer");
    }

    WorldStorage input_storage;
    WorldStorage output_storage;
    MakeWorld(&host->input_world, &input_storage, w, h, options.bitdepth, host->layer_rect);
    MakeWorld(&host->output_world, &output_storage, static_cast<int>(result_width), static_cast<int>(result_height),
              options.bitdepth, pre_output.result_rect);
    FillTestScene(&host->input_world, options.bitdepth);

    PF_SmartRenderInput render_input = {};
    render_input.output_request = pre_input.output_request;
    render_input.bitdepth = options.bitdepth;
    render_input.pre_render_data = pre_output.pre_render_data;
    PF_SmartRenderCallbacks render_callbacks = {MockCheckoutLayerPixels, MockCheckinLayerPixels, MockCheckoutOutput};
    PF_SmartRenderExtra render_extra = {&render_input, &render_callbacks};

    err = effect_main(PF_Cmd_SMART_RENDER, &in_data, &out_data, nullptr, nullptr, &render_extra);
    Check(err == PF_Err_NONE, options.label + ": smart render succeeds");
    Check(host->input_checked_out && host->output_checked_out, options.label + ": render checked out its buffers");
    Check(host->handles_created == host->handles_disposed, options.label + ": every scratch handle was released");
    Check(host->live_handles.empty(), options.label + ": no handles leaked");

    if (pre_output.delete_pre_render_data_func != nullptr) {
        pre_output.delete_pre_render_data_func(pre_output.pre_render_data);
    }

    const cosmic::PixelDepth depth = options.bitdepth == 8    ? cosmic::PixelDepth::kBits8
                                     : options.bitdepth == 16 ? cosmic::PixelDepth::kBits16
                                                              : cosmic::PixelDepth::kFloat32;
    cosmic_test::TestImage image(static_cast<int>(result_width), static_cast<int>(result_height), depth);
    std::memcpy(image.View().data, output_storage.bytes.data(), output_storage.bytes.size());

    // Inside the bar the gradient is opaque and not black.
    const int ex = static_cast<int>(-pre_output.result_rect.left);
    const int ey = static_cast<int>(-pre_output.result_rect.top);
    const cosmic::PixelF inside = image.GetPixel(ex + w / 4, ey + h / 2);
    Check(inside.a > 0.99f, options.label + ": the shape is opaque");
    Check(inside.r + inside.g + inside.b > 0.02f, options.label + ": the shape is filled");
    if (options.matte == 1 && options.defocus == 0.0f) {
        const cosmic::PixelF corner = image.GetPixel(0, 0);
        Check(corner.a < 0.02f, options.label + ": far from the shapes stays transparent");
    }

    if (!out_dir.empty()) cosmic_test::WritePng(out_dir + "/mockhost_" + options.label + ".png", image);
    if (out_image != nullptr) {
        out_image->Resize(image.View().width, image.View().height, depth);
        std::memcpy(out_image->View().data, image.View().data,
                    static_cast<std::size_t>(image.View().rowbytes) * image.View().height);
    }
    if (out_rect != nullptr) *out_rect = pre_output.result_rect;

    g_host = nullptr;
    delete host;
    return true;
}

void ApplyOptions(MockHost* host, const RenderOptions& options) {
    Param(host, kIndexExpandBounds).u.bd.value = options.expand_bounds ? TRUE : FALSE;
    Param(host, kIndexMatte).u.pd.value = options.matte;
    Param(host, kIndexType).u.pd.value = options.type;
    Param(host, kIndexDefocus).u.fs_d.value = options.defocus;
    Param(host, kIndexGlowRadius).u.fs_d.value = options.glow_radius;
    Param(host, kIndexAnimateGrain).u.bd.value = options.animate_grain ? TRUE : FALSE;
    if (options.turbulence >= 0.0f) Param(host, 28).u.fs_d.value = options.turbulence;
    if (options.depth_shape > 0) Param(host, kIndexDepthShape).u.pd.value = options.depth_shape;
}

// ---------------------------------------------------------------------------
// GPU
// ---------------------------------------------------------------------------

using FakeStatsFn = void (*)(int*, int*, int*, int*);
using FakeFailFn = void (*)(int);
FakeStatsFn g_fake_stats = nullptr;
FakeFailFn g_fake_fail = nullptr;

// Renders one frame through GPU_DEVICE_SETUP, SMART_PRE_RENDER and
// SMART_RENDER_GPU on 32-bit float GPU frames. Returns the frame as ARGB.
bool RunGpuRender(EffectMainFn effect_main, const RenderOptions& options, bool gpu_checkbox,
                  cosmic_test::TestImage* out_image, PF_LRect* out_rect) {
    PF_InData in_data;
    PF_OutData out_data;
    PF_UtilCallbacks utils;
    MockHost* host = NewInstance(effect_main, &in_data, &out_data, &utils, options.width, options.height);
    host->downsample = options.downsample;
    in_data.downsample_x.den = options.downsample;
    in_data.downsample_y.den = options.downsample;
    in_data.current_time = 12;
    ApplyOptions(host, options);
    Param(host, kIndexGpu).u.bd.value = gpu_checkbox ? TRUE : FALSE;
    const std::string label = "gpu " + options.label;

    // Device setup: the kernels load into the (fake) CUDA context.
    PF_GPUDeviceSetupInput setup_in = {PF_GPU_Framework_CUDA, 0};
    PF_GPUDeviceSetupOutput setup_out = {nullptr};
    PF_GPUDeviceSetupExtra setup = {&setup_in, &setup_out};
    out_data.out_flags2 = 0;
    PF_Err err = effect_main(PF_Cmd_GPU_DEVICE_SETUP, &in_data, &out_data, nullptr, nullptr, &setup);
    Check(err == PF_Err_NONE, label + ": GPU device setup succeeds");
    Check(setup_out.gpu_data != nullptr, label + ": the CUDA device is accepted");
    Check((out_data.out_flags2 & PF_OutFlag2_SUPPORTS_GPU_RENDER_F32) != 0, label + ": and GPU rendering is claimed");

    const int w = options.width / options.downsample;
    const int h = options.height / options.downsample;
    host->layer_rect = PF_LRect{0, 0, w, h};

    PF_PreRenderInput pre_input = {};
    pre_input.bitdepth = 32;
    pre_input.output_request.rect = host->layer_rect;
    pre_input.output_request.field = PF_Field_FRAME;
    pre_input.output_request.channel_mask = PF_ChannelMask_ARGB;
    pre_input.gpu_data = setup_out.gpu_data;
    pre_input.what_gpu = PF_GPU_Framework_CUDA;
    pre_input.device_index = 0;
    PF_PreRenderOutput pre_output = {};
    PF_PreRenderCallbacks pre_callbacks = {MockCheckoutLayer, MockGuidMixInPtr};
    PF_PreRenderExtra pre_extra = {&pre_input, &pre_output, &pre_callbacks};
    err = effect_main(PF_Cmd_SMART_PRE_RENDER, &in_data, &out_data, nullptr, nullptr, &pre_extra);
    Check(err == PF_Err_NONE, label + ": smart pre-render succeeds");
    const bool possible = (pre_output.flags & PF_RenderOutputFlag_GPU_RENDER_POSSIBLE) != 0;
    Check(possible == gpu_checkbox, label + ": GPU rendering offered exactly when GPU Acceleration is on");

    bool rendered = false;
    if (possible) {
        const int ow = pre_output.result_rect.right - pre_output.result_rect.left;
        const int oh = pre_output.result_rect.bottom - pre_output.result_rect.top;

        // The scene, as a float world, then as a BGRA GPU frame with padding.
        WorldStorage scene_storage;
        PF_EffectWorld scene = {};
        MakeWorld(&scene, &scene_storage, w, h, 32, host->layer_rect);
        FillTestScene(&scene, 32);
        const int in_pitch = w + 3;
        const int out_pitch = ow + 5;
        void* in_mem = nullptr;
        void* out_mem = nullptr;
        MockAllocateDeviceMemory(nullptr, 0, static_cast<size_t>(in_pitch) * h * 16, &in_mem);
        MockAllocateDeviceMemory(nullptr, 0, static_cast<size_t>(out_pitch) * oh * 16, &out_mem);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const cosmic::PixelF p = reinterpret_cast<const cosmic::PixelF*>(scene.data)[y * w + x];
                float* q = static_cast<float*>(in_mem) + (static_cast<std::size_t>(y) * in_pitch + x) * 4;
                q[0] = p.b;
                q[1] = p.g;
                q[2] = p.r;
                q[3] = p.a;
            }
        }
        std::memset(&host->input_world, 0, sizeof(host->input_world));
        host->input_world.width = w;
        host->input_world.height = h;
        host->input_world.rowbytes = in_pitch * 16;
        host->input_world.origin_x = 0;
        host->input_world.origin_y = 0;
        std::memset(&host->output_world, 0, sizeof(host->output_world));
        host->output_world.width = ow;
        host->output_world.height = oh;
        host->output_world.rowbytes = out_pitch * 16;
        host->output_world.origin_x = pre_output.result_rect.left;
        host->output_world.origin_y = pre_output.result_rect.top;
        g_gpu_world_data[&host->input_world] = in_mem;
        g_gpu_world_data[&host->output_world] = out_mem;

        PF_SmartRenderInput render_input = {};
        render_input.output_request = pre_input.output_request;
        render_input.bitdepth = 32;
        render_input.pre_render_data = pre_output.pre_render_data;
        render_input.gpu_data = setup_out.gpu_data;
        render_input.what_gpu = PF_GPU_Framework_CUDA;
        render_input.device_index = 0;
        PF_SmartRenderCallbacks render_callbacks = {MockCheckoutLayerPixels, MockCheckinLayerPixels,
                                                    MockCheckoutOutput};
        PF_SmartRenderExtra render_extra = {&render_input, &render_callbacks};
        err = effect_main(PF_Cmd_SMART_RENDER_GPU, &in_data, &out_data, nullptr, nullptr, &render_extra);
        Check(err == PF_Err_NONE, label + ": smart render GPU succeeds");
        rendered = err == PF_Err_NONE;

        if (out_image != nullptr) {
            out_image->Resize(ow, oh, cosmic::PixelDepth::kFloat32);
            for (int y = 0; y < oh; ++y) {
                for (int x = 0; x < ow; ++x) {
                    const float* q = static_cast<const float*>(out_mem) + (static_cast<std::size_t>(y) * out_pitch + x) * 4;
                    out_image->SetPixel(x, y, cosmic::PixelF{q[3], q[2], q[1], q[0]});
                }
            }
        }
        if (out_rect != nullptr) *out_rect = pre_output.result_rect;
        g_gpu_world_data.clear();
        MockFreeDeviceMemory(nullptr, 0, in_mem);
        MockFreeDeviceMemory(nullptr, 0, out_mem);
    }
    if (pre_output.delete_pre_render_data_func != nullptr) {
        pre_output.delete_pre_render_data_func(pre_output.pre_render_data);
    }

    PF_GPUDeviceSetdownInput setdown_in = {setup_out.gpu_data, PF_GPU_Framework_CUDA, 0};
    PF_GPUDeviceSetdownExtra setdown = {&setdown_in};
    err = effect_main(PF_Cmd_GPU_DEVICE_SETDOWN, &in_data, &out_data, nullptr, nullptr, &setdown);
    Check(err == PF_Err_NONE, label + ": GPU device setdown succeeds");
    Check(g_device_live.empty(), label + ": all device memory was freed");
    Check(host->live_handles.empty(), label + ": no host handles leaked");

    g_host = nullptr;
    delete host;
    return rendered;
}

double MaxDifference(const cosmic_test::TestImage& a, const cosmic_test::TestImage& b) {
    if (a.View().width != b.View().width || a.View().height != b.View().height) return 1.0e9;
    double worst = 0.0;
    for (int y = 0; y < a.View().height; ++y) {
        for (int x = 0; x < a.View().width; ++x) {
            const cosmic::PixelF p = a.GetPixel(x, y);
            const cosmic::PixelF q = b.GetPixel(x, y);
            if (!std::isfinite(q.a) || !std::isfinite(q.r) || !std::isfinite(q.g) || !std::isfinite(q.b)) return 1.0e9;
            worst = std::max({worst, (double)std::fabs(p.a - q.a), (double)std::fabs(p.r - q.r),
                              (double)std::fabs(p.g - q.g), (double)std::fabs(p.b - q.b)});
        }
    }
    return worst;
}

void TestGpu(EffectMainFn effect_main) {
    std::printf("-- GPU (CUDA through the fake driver)\n");
    std::vector<RenderOptions> cases;
    cases.push_back({"defaults", 32});
    {
        RenderOptions o{"defocus_turbulence", 32};
        o.defocus = 14.0f;
        o.turbulence = 12.0f;
        o.depth_shape = 5;  // Bulge
        cases.push_back(o);
    }
    {
        RenderOptions o{"full_frame_half", 32};
        o.matte = 3;
        o.type = 2;
        o.downsample = 2;
        cases.push_back(o);
    }
    {
        RenderOptions o{"odd_no_expand", 32, 97, 61};
        o.expand_bounds = false;
        o.animate_grain = true;
        cases.push_back(o);
    }
    for (const RenderOptions& options : cases) {
        cosmic_test::TestImage cpu;
        cosmic_test::TestImage gpu;
        PF_LRect cpu_rect = {};
        PF_LRect gpu_rect = {};
        RunRender(effect_main, options, "", &cpu, &cpu_rect);
        if (!RunGpuRender(effect_main, options, true, &gpu, &gpu_rect)) continue;
        Check(cpu_rect.left == gpu_rect.left && cpu_rect.right == gpu_rect.right && cpu_rect.top == gpu_rect.top &&
                  cpu_rect.bottom == gpu_rect.bottom,
              options.label + ": GPU and CPU declare the same bounds");
        const double diff = MaxDifference(cpu, gpu);
        char line[128];
        std::snprintf(line, sizeof(line), "%s: GPU frame matches the CPU frame (max difference %.2e)",
                      options.label.c_str(), diff);
        std::printf("   %s\n", line);
        Check(diff < 2.0e-3, line);
    }

    // GPU Acceleration off: After Effects is told to render on the CPU.
    RunGpuRender(effect_main, cases[0], false, nullptr, nullptr);

    // A device that fails mid-render: the frame still comes out, via the CPU.
    if (g_fake_fail != nullptr) {
        cosmic_test::TestImage cpu;
        cosmic_test::TestImage gpu;
        RunRender(effect_main, cases[1], "", &cpu, nullptr);
        g_fake_fail(1000000);
        const bool rendered = RunGpuRender(effect_main, cases[1], true, &gpu, nullptr);
        g_fake_fail(0);
        Check(rendered, "a failing GPU falls back to the CPU");
        Check(MaxDifference(cpu, gpu) < 2.0e-3, "and the fallback frame matches the CPU frame");
    }

    // Anything but CUDA is declined, so After Effects renders on the CPU.
    {
        PF_InData in_data;
        PF_OutData out_data;
        PF_UtilCallbacks utils;
        MockHost* host = NewInstance(effect_main, &in_data, &out_data, &utils, 64, 64);
        PF_GPUDeviceSetupInput setup_in = {PF_GPU_Framework_OPENCL, 0};
        PF_GPUDeviceSetupOutput setup_out = {nullptr};
        PF_GPUDeviceSetupExtra setup = {&setup_in, &setup_out};
        out_data.out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
        const PF_Err err = effect_main(PF_Cmd_GPU_DEVICE_SETUP, &in_data, &out_data, nullptr, nullptr, &setup);
        Check(err == PF_Err_NONE && setup_out.gpu_data == nullptr &&
                  (out_data.out_flags2 & PF_OutFlag2_SUPPORTS_GPU_RENDER_F32) == 0,
              "an OpenCL device is declined");
        g_host = nullptr;
        delete host;
    }

    if (g_fake_stats != nullptr) {
        int loaded = 0, unloaded = 0, launches = 0, violations = 0;
        g_fake_stats(&loaded, &unloaded, &launches, &violations);
        std::printf("   fake driver: %d module loads, %d unloads, %d kernel launches\n", loaded, unloaded, launches);
        Check(loaded > 0 && loaded == unloaded, "every module loaded is unloaded");
        Check(launches > 0, "kernels ran on the device");
        Check(violations == 0, "every driver call had the context current");
    }
}

void TestV10Project(EffectMainFn effect_main) {
    std::printf("-- a project saved with v1.0\n");
    // After Effects restores saved values by parameter id; the ids v1.0 had
    // are unchanged, so its values land on the same controls. Values only
    // v1.0 could have produced must still render.
    PF_InData in_data;
    PF_OutData out_data;
    PF_UtilCallbacks utils;
    MockHost* host = NewInstance(effect_main, &in_data, &out_data, &utils, 160, 90);
    std::string depth_names;
    for (const std::string& names : host->popup_strings) {
        if (names.rfind("Dome|", 0) == 0) depth_names = names;
    }
    Check(depth_names.rfind("Dome|Sphere|Ridge|Wave|", 0) == 0,
          "Depth Shape keeps v1.0's four entries in place (" + depth_names + ")");
    Check(Param(host, kIndexDepthShape).u.pd.num_choices == 5, "and appends Bulge");
    Check(Param(host, kIndexLoopWithAngle).u.bd.value == FALSE, "new instances do not loop the noise with the angle");
    Check(Param(host, kIndexGpu).u.bd.value == TRUE, "GPU Acceleration is on for new and old projects alike");
    g_host = nullptr;
    delete host;

    RenderOptions o{"v1.0 values", 8, 160, 90};
    o.depth_shape = 4;  // Wave, a v1.0 value
    o.turbulence = 10.0f;
    RunRender(effect_main, o, "");
}

void TestPaletteSupervision(EffectMainFn effect_main) {
    std::printf("-- palette popup and colour controls\n");
    PF_InData in_data;
    PF_OutData out_data;
    PF_UtilCallbacks utils;
    MockHost* host = NewInstance(effect_main, &in_data, &out_data, &utils, 320, 180);

    const std::string names = host->popup_strings.empty() ? "" : host->popup_strings[0];
    Check(names.find("Deep Space|") == 0, "the palette popup lists the presets");
    Check(Param(host, kIndexPalette).u.pd.num_choices == cosmic::PresetCount() + 1,
          "the palette popup has every preset plus Custom");
    Check((Param(host, kIndexPalette).flags & PF_ParamFlag_SUPERVISE) != 0, "the palette popup is supervised");

    std::vector<PF_ParamDef*> params(kParamCount, nullptr);
    PF_ParamDef input = {};
    params[0] = &input;
    for (int i = 1; i < kParamCount; ++i) params[static_cast<std::size_t>(i)] = &Param(host, i);

    // Pick Sunset: the colours follow.
    const int sunset = 4;
    Param(host, kIndexPalette).u.pd.value = sunset;
    PF_UserChangedParamExtra extra = {kIndexPalette};
    PF_Err err = effect_main(PF_Cmd_USER_CHANGED_PARAM, &in_data, &out_data, params.data(), nullptr, &extra);
    Check(err == PF_Err_NONE, "user changed palette succeeds");
    const cosmic::PalettePreset& preset = cosmic::Preset(sunset - 1);
    bool colours_ok = true;
    for (int k = 0; k < cosmic::kStopCount; ++k) {
        const PF_ParamDef& c = Param(host, kIndexColor1 + k);
        colours_ok = colours_ok && c.u.cd.value.red == preset.srgb[k][0] && c.u.cd.value.green == preset.srgb[k][1] &&
                     c.u.cd.value.blue == preset.srgb[k][2] && (c.uu.change_flags & PF_ChangeFlag_CHANGED_VALUE);
    }
    Check(colours_ok, "picking a palette fills the colour controls");

    // Edit a colour: the popup switches to Custom.
    Param(host, kIndexColor1 + 2).u.cd.value.red = 1;
    extra.param_index = kIndexColor1 + 2;
    err = effect_main(PF_Cmd_USER_CHANGED_PARAM, &in_data, &out_data, params.data(), nullptr, &extra);
    Check(err == PF_Err_NONE, "user changed colour succeeds");
    Check(Param(host, kIndexPalette).u.pd.value == cosmic::PresetCount() + 1, "editing a colour selects Custom");

    // Dynamic flags: time-varying only when the grain animates.
    out_data.out_flags = kExpectedOutFlags;
    err = effect_main(PF_Cmd_QUERY_DYNAMIC_FLAGS, &in_data, &out_data, nullptr, nullptr, nullptr);
    Check(err == PF_Err_NONE, "query dynamic flags succeeds");
    Check((out_data.out_flags & PF_OutFlag_NON_PARAM_VARY) == 0, "still grain lets frames be cached");
    Check((out_data.out_flags & PF_OutFlag_DEEP_COLOR_AWARE) != 0, "other flags are left alone");
    Param(host, kIndexAnimateGrain).u.bd.value = TRUE;
    err = effect_main(PF_Cmd_QUERY_DYNAMIC_FLAGS, &in_data, &out_data, nullptr, nullptr, nullptr);
    Check((out_data.out_flags & PF_OutFlag_NON_PARAM_VARY) != 0, "animated grain varies with time");
    Param(host, kIndexGrain).u.fs_d.value = 0.0;
    err = effect_main(PF_Cmd_QUERY_DYNAMIC_FLAGS, &in_data, &out_data, nullptr, nullptr, nullptr);
    Check((out_data.out_flags & PF_OutFlag_NON_PARAM_VARY) == 0, "no grain, nothing to animate");

    err = effect_main(PF_Cmd_ABOUT, &in_data, &out_data, nullptr, nullptr, nullptr);
    Check(err == PF_Err_NONE && std::strstr(out_data.return_msg, "Cosmic Gradient") != nullptr,
          "about names the effect");

    g_host = nullptr;
    delete host;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: cosmic_mock_host <path to CosmicGradient.aex> [output directory]\n");
        return 2;
    }
    const std::string plugin_path = argv[1];
    const std::string out_dir = argc > 2 ? argv[2] : "";
    const std::string fake_cuda = argc > 3 ? argv[3] : "";

    g_basic_suite.AcquireSuite = MockAcquireSuite;
    g_gpu_suite.GetDeviceCount = MockGetDeviceCount;
    g_gpu_suite.GetDeviceInfo = MockGetDeviceInfo;
    g_gpu_suite.AllocateDeviceMemory = MockAllocateDeviceMemory;
    g_gpu_suite.FreeDeviceMemory = MockFreeDeviceMemory;
    g_gpu_suite.GetGPUWorldData = MockGetGPUWorldData;
    g_world_suite.PF_GetPixelFormat = MockGetPixelFormat;

    // The fake driver is loaded first, so the plug-in finds it as nvcuda.dll.
    if (!fake_cuda.empty()) {
        HMODULE fake = LoadLibraryA(fake_cuda.c_str());
        Check(fake != nullptr, "the fake CUDA driver loads");
        if (fake != nullptr) {
            g_fake_stats = reinterpret_cast<FakeStatsFn>(reinterpret_cast<void*>(GetProcAddress(fake, "FakeCudaStats")));
            g_fake_fail = reinterpret_cast<FakeFailFn>(reinterpret_cast<void*>(GetProcAddress(fake, "FakeCudaFailLaunches")));
        }
    }
    g_basic_suite.ReleaseSuite = MockReleaseSuite;
    g_iterate_suite.iterate_generic = MockIterateGeneric;

    HMODULE module = LoadLibraryA(plugin_path.c_str());
    if (module == nullptr) {
        std::printf("FAIL: could not load %s (error %lu)\n", plugin_path.c_str(), GetLastError());
        return 1;
    }
    EffectMainFn effect_main = reinterpret_cast<EffectMainFn>(GetProcAddress(module, "EffectMain"));
    Check(effect_main != nullptr, "EffectMain is exported");
    Check(GetProcAddress(module, "PluginDataEntryFunction2") != nullptr, "PluginDataEntryFunction2 is exported");
    if (effect_main == nullptr) return 1;

    TestPaletteSupervision(effect_main);
    TestV10Project(effect_main);

    std::vector<RenderOptions> cases;
    cases.push_back({"8bpc", 8});
    cases.push_back({"16bpc", 16});
    cases.push_back({"32bpc", 32});
    {
        RenderOptions o{"no_expand", 8};
        o.expand_bounds = false;
        cases.push_back(o);
    }
    {
        RenderOptions o{"full_frame", 16};
        o.matte = 3;
        o.type = 2;
        cases.push_back(o);
    }
    {
        RenderOptions o{"half_res", 32};
        o.downsample = 2;
        cases.push_back(o);
    }
    {
        RenderOptions o{"defocus", 8};
        o.defocus = 12.0f;
        o.type = 3;
        cases.push_back(o);
    }
    {
        RenderOptions o{"odd_size", 8, 97, 61};
        o.glow_radius = 400.0f;
        o.animate_grain = true;
        cases.push_back(o);
    }
    for (const RenderOptions& options : cases) {
        std::printf("-- %s (%d bpc, %dx%d, 1/%d)\n", options.label.c_str(), options.bitdepth, options.width,
                    options.height, options.downsample);
        RunRender(effect_main, options, out_dir);
    }

    // Applying and removing the effect repeatedly must not leak or crash.
    for (int i = 0; i < 20; ++i) {
        RenderOptions options = cases[0];
        options.label = "repeat";
        RunRender(effect_main, options, "");
    }

    if (!fake_cuda.empty()) {
        TestGpu(effect_main);
    } else {
        std::printf("-- GPU: skipped (no fake CUDA driver given)\n");
    }

    FreeLibrary(module);
    std::printf("%s\n", g_failures == 0 ? "mock host: all checks passed" : "mock host: FAILURES");
    return g_failures == 0 ? 0 : 1;
}
