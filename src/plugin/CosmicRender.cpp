#include "CosmicRender.h"

#include <algorithm>
#include <cmath>
#include <new>
#include <vector>

#include "AeAdapters.h"
#include "CosmicParams.h"
#include "CudaDevice.h"
#include "core/CosmicPipeline.h"
#include "gpu/GpuPipeline.h"

namespace cosmic {
namespace {

constexpr A_long kProbeCheckoutId = 1;
constexpr A_long kInputCheckoutId = 2;

// Guard rail so an extreme glow or defocus cannot ask After Effects for an
// absurd buffer.
constexpr A_long kMaxBoundsExpansion = 3000;

struct PreRenderData {
    EffectParams params;
    PF_LRect output_rect = {};
    PF_LRect input_rect = {};
    bool has_input = false;
};

void DeletePreRenderData(void* data) {
    delete static_cast<PreRenderData*>(data);
}

bool IsEmpty(const PF_LRect& rect) {
    return rect.right <= rect.left || rect.bottom <= rect.top;
}

PF_LRect Inflate(const PF_LRect& rect, A_long amount) {
    PF_LRect out = rect;
    out.left -= amount;
    out.top -= amount;
    out.right += amount;
    out.bottom += amount;
    return out;
}

float RationalToFloat(const PF_RationalScale& value) {
    if (value.den == 0) return 1.0f;
    return static_cast<float>(value.num) / static_cast<float>(value.den);
}

A_long BoundsExpansion(const EffectParams& params, float blur_scale) {
    if (!params.expand_bounds || !CanExpand(params.settings)) return 0;
    const float reach = EffectReach(params.settings, blur_scale);
    if (!(reach > 0.0f)) return 0;
    return std::min<A_long>(kMaxBoundsExpansion, static_cast<A_long>(std::ceil(reach)));
}

// Where the world's top-left sits in layer coordinates. When the host hands
// back exactly the rectangle we asked for, that rectangle is authoritative.
void ResolveWorldOrigin(const PF_EffectWorld* world, const PF_LRect& declared, A_long* left, A_long* top) {
    const A_long width = declared.right - declared.left;
    const A_long height = declared.bottom - declared.top;
    if (world->width == width && world->height == height) {
        *left = declared.left;
        *top = declared.top;
    } else {
        *left = world->origin_x;
        *top = world->origin_y;
    }
}

void RenderMapping(const PF_InData* in_data, float* to_full_x, float* to_full_y) {
    const float dsx = RationalToFloat(in_data->downsample_x);
    const float dsy = RationalToFloat(in_data->downsample_y);
    const float par = RationalToFloat(in_data->pixel_aspect_ratio);
    *to_full_x = (par > 0.0f ? par : 1.0f) / (dsx > 0.0f ? dsx : 1.0f);
    *to_full_y = 1.0f / (dsy > 0.0f ? dsy : 1.0f);
}

// A suite acquired for the length of a scope.
template <typename Suite>
class ScopedSuite {
public:
    ScopedSuite(SPBasicSuite* basic, const char* name, int version)
        : basic_(basic), name_(name), version_(version), suite_(AcquireSuite<Suite>(basic, name, version)) {}
    ~ScopedSuite() {
        if (suite_ != nullptr) basic_->ReleaseSuite(name_, version_);
    }
    ScopedSuite(const ScopedSuite&) = delete;
    ScopedSuite& operator=(const ScopedSuite&) = delete;

    Suite* get() const { return suite_; }
    Suite* operator->() const { return suite_; }

private:
    SPBasicSuite* basic_;
    const char* name_;
    int version_;
    Suite* suite_;
};

PF_Err ToPfErr(CosmicResult result) {
    switch (result) {
        case CosmicResult::kOk: return PF_Err_NONE;
        case CosmicResult::kOutOfMemory: return PF_Err_OUT_OF_MEMORY;
        case CosmicResult::kInvalidArguments: return PF_Err_BAD_CALLBACK_PARAM;
        case CosmicResult::kDeviceError:
        default: return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
}

// The CPU renderer on host images, with host memory and the host's threads.
PF_Err RunPipeline(PF_InData* in_data, const CosmicSettings& settings, const HostImage& source, int source_left,
                   int source_top, const HostImage& dest, int dest_left, int dest_top) {
    SPBasicSuite* basic = in_data->pica_basicP;
    if (basic == nullptr) return PF_Err_BAD_CALLBACK_PARAM;
    ScopedSuite<PF_HandleSuite1> handle_suite(basic, kPFHandleSuite, kPFHandleSuiteVersion1);
    if (handle_suite.get() == nullptr) return PF_Err_OUT_OF_MEMORY;
    ScopedSuite<PF_Iterate8Suite1> iterate_suite(basic, kPFIterate8Suite, kPFIterate8SuiteVersion1);

    AeAllocator allocator(in_data, handle_suite.get());
    AeTaskRunner runner(in_data, iterate_suite.get());

    CosmicRender render;
    render.source = source;
    render.source_left = source_left;
    render.source_top = source_top;
    render.dest = dest;
    render.dest_left = dest_left;
    render.dest_top = dest_top;
    RenderMapping(in_data, &render.to_full_x, &render.to_full_y);
    render.blur_scale = BlurScale(in_data);
    return ToPfErr(RenderCosmic(settings, render, allocator, runner));
}

// GPU frames always hold 32-bit float, whatever the project's depth, so the
// Auto working space is decided from the depth the host reports instead.
CosmicSettings ResolveWorkingSpace(CosmicSettings settings, short bitdepth) {
    if (settings.working_space == WorkingSpace::kAuto) {
        settings.working_space = bitdepth == 32 ? WorkingSpace::kLinear : WorkingSpace::kSrgb;
    }
    return settings;
}

// Last resort for a GPU render the device could not do (out of video memory,
// a driver error): bring the frames to the host, render them on the CPU and
// send the result back, so the frame still comes out right.
PF_Err RenderGpuFramesOnCpu(PF_InData* in_data, const CosmicSettings& settings, CudaDevice& device,
                            PF_EffectWorld* input_world, void* input_mem, int input_left, int input_top,
                            PF_EffectWorld* output_world, void* output_mem, int output_left, int output_top) {
    if (!device.Valid() || output_mem == nullptr) return PF_Err_INTERNAL_STRUCT_DAMAGED;

    auto download = [&](PF_EffectWorld* world, void* mem, std::vector<PixelF>* out) {
        const std::size_t bytes = static_cast<std::size_t>(world->rowbytes) * static_cast<std::size_t>(world->height);
        std::vector<unsigned char> raw(bytes);
        if (!device.Download(raw.data(), reinterpret_cast<DevicePtr>(mem), bytes)) return false;
        out->resize(static_cast<std::size_t>(world->width) * static_cast<std::size_t>(world->height));
        for (A_long y = 0; y < world->height; ++y) {
            const FrameBGRA* row = reinterpret_cast<const FrameBGRA*>(raw.data() + y * world->rowbytes);
            for (A_long x = 0; x < world->width; ++x) {
                (*out)[static_cast<std::size_t>(y) * world->width + x] = PixelF{row[x].a, row[x].r, row[x].g, row[x].b};
            }
        }
        return true;
    };

    std::vector<PixelF> source_pixels;
    HostImage source;
    if (input_world != nullptr && input_mem != nullptr) {
        if (!download(input_world, input_mem, &source_pixels)) return PF_Err_INTERNAL_STRUCT_DAMAGED;
        source.data = source_pixels.data();
        source.rowbytes = input_world->width * static_cast<int>(sizeof(PixelF));
        source.width = input_world->width;
        source.height = input_world->height;
        source.depth = PixelDepth::kFloat32;
    }

    std::vector<PixelF> dest_pixels(static_cast<std::size_t>(output_world->width) * output_world->height);
    HostImage dest;
    dest.data = dest_pixels.data();
    dest.rowbytes = output_world->width * static_cast<int>(sizeof(PixelF));
    dest.width = output_world->width;
    dest.height = output_world->height;
    dest.depth = PixelDepth::kFloat32;

    PF_Err err = RunPipeline(in_data, settings, source, input_left, input_top, dest, output_left, output_top);
    if (err) return err;

    const std::size_t bytes =
        static_cast<std::size_t>(output_world->rowbytes) * static_cast<std::size_t>(output_world->height);
    std::vector<unsigned char> raw(bytes, 0);
    for (A_long y = 0; y < output_world->height; ++y) {
        FrameBGRA* row = reinterpret_cast<FrameBGRA*>(raw.data() + y * output_world->rowbytes);
        for (A_long x = 0; x < output_world->width; ++x) {
            const PixelF& p = dest_pixels[static_cast<std::size_t>(y) * output_world->width + x];
            row[x] = FrameBGRA{p.b, p.g, p.r, p.a};
        }
    }
    return device.Upload(reinterpret_cast<DevicePtr>(output_mem), raw.data(), bytes) ? PF_Err_NONE
                                                                                     : PF_Err_INTERNAL_STRUCT_DAMAGED;
}

struct CheckedOutFrames {
    PF_EffectWorld* input = nullptr;
    PF_EffectWorld* output = nullptr;
    A_long input_left = 0;
    A_long input_top = 0;
    A_long output_left = 0;
    A_long output_top = 0;
};

PF_Err CheckOutFrames(PF_InData* in_data, PF_SmartRenderExtra* extra, const PreRenderData& data, bool gpu,
                      CheckedOutFrames* frames) {
    PF_Err err = PF_Err_NONE;
    if (data.has_input) {
        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, kInputCheckoutId, &frames->input));
    }
    ERR(extra->cb->checkout_output(in_data->effect_ref, &frames->output));
    if (err) return err;
    if (frames->output == nullptr) return PF_Err_NONE;

    frames->output_left = data.output_rect.left;
    frames->output_top = data.output_rect.top;
    ResolveWorldOrigin(frames->output, data.output_rect, &frames->output_left, &frames->output_top);

    // GPU frames have no host data pointer; their pixels are on the device.
    const bool has_input = frames->input != nullptr && (gpu || frames->input->data != nullptr);
    if (has_input) {
        frames->input_left = data.input_rect.left;
        frames->input_top = data.input_rect.top;
        ResolveWorldOrigin(frames->input, data.input_rect, &frames->input_left, &frames->input_top);
    } else {
        frames->input = nullptr;
    }
    return PF_Err_NONE;
}

}  // namespace

PF_Err SmartPreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra) {
    (void)out_data;
    PF_Err err = PF_Err_NONE;

    EffectParams params;
    ERR(ReadParams(in_data, &params));
    if (err) return err;

    PF_RenderRequest request = extra->input->output_request;
    request.channel_mask = PF_ChannelMask_ARGB;

    // An empty request is the cheap way to learn the layer's real extent.
    PF_RenderRequest probe = request;
    probe.rect.left = 0;
    probe.rect.top = 0;
    probe.rect.right = 0;
    probe.rect.bottom = 0;

    PF_CheckoutResult probe_result = {};
    ERR(extra->cb->checkout_layer(in_data->effect_ref, kParamInput, kProbeCheckoutId, &probe,
                                  in_data->current_time, in_data->time_step, in_data->time_scale, &probe_result));
    if (err) return err;

    const PF_LRect layer_rect = probe_result.max_result_rect;
    if (IsEmpty(layer_rect)) {
        extra->output->result_rect = layer_rect;
        extra->output->max_result_rect = layer_rect;
        extra->output->solid = FALSE;
        return PF_Err_NONE;
    }

    const A_long expansion = BoundsExpansion(params, BlurScale(in_data));
    const PF_LRect output_rect = Inflate(layer_rect, expansion);

    // Blurs, the glow and the content bounds all need the whole layer, so ask
    // for all of it and always render the full result; After Effects caches
    // the frame, so partial requests would only redo the same work.
    PF_RenderRequest full = request;
    full.rect = layer_rect;

    PF_CheckoutResult input_result = {};
    ERR(extra->cb->checkout_layer(in_data->effect_ref, kParamInput, kInputCheckoutId, &full, in_data->current_time,
                                  in_data->time_step, in_data->time_scale, &input_result));
    if (err) return err;

    PreRenderData* data = new (std::nothrow) PreRenderData();
    if (data == nullptr) return PF_Err_OUT_OF_MEMORY;
    data->params = params;
    data->output_rect = output_rect;
    data->input_rect = input_result.result_rect;
    data->has_input = !IsEmpty(input_result.result_rect);

    extra->output->result_rect = output_rect;
    extra->output->max_result_rect = output_rect;
    extra->output->solid = FALSE;
    extra->output->flags |= PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS;
    // The GPU renders when the project uses CUDA, the device accepted the
    // kernels at GPU_DEVICE_SETUP, and the user has not switched it off.
    if (params.gpu && extra->input->what_gpu == PF_GPU_Framework_CUDA && extra->input->gpu_data != nullptr) {
        extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
    }
    extra->output->pre_render_data = data;
    extra->output->delete_pre_render_data_func = DeletePreRenderData;
    return PF_Err_NONE;
}

PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    (void)out_data;
    PreRenderData* data = static_cast<PreRenderData*>(extra->input->pre_render_data);
    if (data == nullptr) return PF_Err_INTERNAL_STRUCT_DAMAGED;

    CheckedOutFrames frames;
    PF_Err err = CheckOutFrames(in_data, extra, *data, false, &frames);
    if (err || frames.output == nullptr) return err;

    const PixelDepth depth = DepthFromBitsPerChannel(extra->input->bitdepth);
    return RunPipeline(in_data, data->params.settings, MakeHostImage(frames.input, depth),
                       static_cast<int>(frames.input_left), static_cast<int>(frames.input_top),
                       MakeHostImage(frames.output, depth), static_cast<int>(frames.output_left),
                       static_cast<int>(frames.output_top));
}

PF_Err GpuDeviceSetup(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetupExtra* extra) {
    // Only CUDA is implemented. Declining a device is not an error: After
    // Effects then renders the effect on the CPU for it.
    extra->output->gpu_data = nullptr;
    out_data->out_flags2 &= ~static_cast<PF_OutFlags2>(PF_OutFlag2_SUPPORTS_GPU_RENDER_F32);
    if (extra->input->what_gpu != PF_GPU_Framework_CUDA) return PF_Err_NONE;

    ScopedSuite<PF_GPUDeviceSuite1> gpu_suite(in_data->pica_basicP, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1);
    if (gpu_suite.get() == nullptr) return PF_Err_NONE;
    PF_GPUDeviceInfo info = {};
    if (gpu_suite->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &info) != PF_Err_NONE) {
        return PF_Err_NONE;
    }
    CudaKernels* kernels = CudaKernels::Create(info.contextPV);
    if (kernels == nullptr) return PF_Err_NONE;
    extra->output->gpu_data = kernels;
    out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    return PF_Err_NONE;
}

PF_Err GpuDeviceSetdown(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetdownExtra* extra) {
    (void)in_data;
    (void)out_data;
    delete static_cast<CudaKernels*>(extra->input->gpu_data);
    extra->input->gpu_data = nullptr;
    return PF_Err_NONE;
}

PF_Err SmartRenderGpu(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    (void)out_data;
    PreRenderData* data = static_cast<PreRenderData*>(extra->input->pre_render_data);
    const CudaKernels* kernels = static_cast<const CudaKernels*>(extra->input->gpu_data);
    if (data == nullptr || kernels == nullptr) return PF_Err_INTERNAL_STRUCT_DAMAGED;

    CheckedOutFrames frames;
    PF_Err err = CheckOutFrames(in_data, extra, *data, true, &frames);
    if (err || frames.output == nullptr) return err;

    SPBasicSuite* basic = in_data->pica_basicP;
    ScopedSuite<PF_GPUDeviceSuite1> gpu_suite(basic, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1);
    if (gpu_suite.get() == nullptr) return PF_Err_BAD_CALLBACK_PARAM;
    {
        ScopedSuite<PF_WorldSuite2> world_suite(basic, kPFWorldSuite, kPFWorldSuiteVersion2);
        PF_PixelFormat format = PF_PixelFormat_GPU_BGRA128;
        if (world_suite.get() != nullptr) world_suite->PF_GetPixelFormat(frames.output, &format);
        if (format != PF_PixelFormat_GPU_BGRA128) return PF_Err_UNRECOGNIZED_PARAM_TYPE;
    }

    void* input_mem = nullptr;
    void* output_mem = nullptr;
    if (frames.input != nullptr) ERR(gpu_suite->GetGPUWorldData(in_data->effect_ref, frames.input, &input_mem));
    ERR(gpu_suite->GetGPUWorldData(in_data->effect_ref, frames.output, &output_mem));
    PF_GPUDeviceInfo info = {};
    ERR(gpu_suite->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &info));
    if (err) return err;
    if (output_mem == nullptr) return PF_Err_INTERNAL_STRUCT_DAMAGED;

    const CosmicSettings settings = ResolveWorkingSpace(data->params.settings, extra->input->bitdepth);
    constexpr int kFrameBytes = static_cast<int>(sizeof(FrameBGRA));

    GpuRender render;
    if (frames.input != nullptr && input_mem != nullptr) {
        render.source = GpuFrame{reinterpret_cast<DevicePtr>(input_mem), static_cast<int>(frames.input->width),
                                 static_cast<int>(frames.input->height),
                                 static_cast<int>(frames.input->rowbytes) / kFrameBytes};
        render.source_left = static_cast<int>(frames.input_left);
        render.source_top = static_cast<int>(frames.input_top);
    }
    render.dest = GpuFrame{reinterpret_cast<DevicePtr>(output_mem), static_cast<int>(frames.output->width),
                           static_cast<int>(frames.output->height),
                           static_cast<int>(frames.output->rowbytes) / kFrameBytes};
    render.dest_left = static_cast<int>(frames.output_left);
    render.dest_top = static_cast<int>(frames.output_top);
    RenderMapping(in_data, &render.to_full_x, &render.to_full_y);
    render.blur_scale = BlurScale(in_data);
    render.float_project = settings.working_space == WorkingSpace::kLinear;

    CudaDevice device(*kernels, info.command_queuePV, in_data->effect_ref, gpu_suite.get(),
                      extra->input->device_index);
    CosmicResult result = CosmicResult::kDeviceError;
    if (device.Valid()) result = RenderCosmicGpu(settings, render, device);
    if (result == CosmicResult::kOk) return PF_Err_NONE;
    if (result == CosmicResult::kInvalidArguments) return PF_Err_BAD_CALLBACK_PARAM;
    return RenderGpuFramesOnCpu(in_data, settings, device, frames.input, input_mem,
                                static_cast<int>(frames.input_left), static_cast<int>(frames.input_top),
                                frames.output, output_mem, static_cast<int>(frames.output_left),
                                static_cast<int>(frames.output_top));
}

PF_Err LegacyFrameSetup(PF_InData* in_data, PF_OutData* out_data) {
    PF_Err err = PF_Err_NONE;

    EffectParams params;
    ERR(ReadParams(in_data, &params));
    if (err) return err;

    const A_long expansion = BoundsExpansion(params, BlurScale(in_data));
    out_data->width = in_data->width + 2 * expansion;
    out_data->height = in_data->height + 2 * expansion;
    out_data->origin.h = static_cast<A_short>(expansion);
    out_data->origin.v = static_cast<A_short>(expansion);
    return PF_Err_NONE;
}

PF_Err LegacyRender(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output) {
    (void)out_data;
    PF_Err err = PF_Err_NONE;

    EffectParams effect_params;
    ERR(ReadParams(in_data, &effect_params));
    if (err) return err;

    PF_EffectWorld* input_world = &params[kParamInput]->u.ld;
    const PixelDepth depth = PF_WORLD_IS_DEEP(output) ? PixelDepth::kBits16 : PixelDepth::kBits8;

    // output_origin says where the input sits inside a buffer we expanded.
    return RunPipeline(in_data, effect_params.settings, MakeHostImage(input_world, depth), 0, 0,
                       MakeHostImage(output, depth), -static_cast<int>(in_data->output_origin_x),
                       -static_cast<int>(in_data->output_origin_y));
}

}  // namespace cosmic
