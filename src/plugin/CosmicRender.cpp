#include "CosmicRender.h"

#include <algorithm>
#include <cmath>
#include <new>

#include "AeAdapters.h"
#include "CosmicParams.h"
#include "core/CosmicPipeline.h"

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

PF_Err RunPipeline(PF_InData* in_data, const CosmicSettings& settings, PF_EffectWorld* input_world,
                   int source_left, int source_top, PF_EffectWorld* output_world, int dest_left, int dest_top,
                   PixelDepth depth) {
    SPBasicSuite* basic = in_data->pica_basicP;
    if (basic == nullptr) return PF_Err_BAD_CALLBACK_PARAM;

    PF_HandleSuite1* handle_suite = AcquireSuite<PF_HandleSuite1>(basic, kPFHandleSuite, kPFHandleSuiteVersion1);
    if (handle_suite == nullptr) return PF_Err_OUT_OF_MEMORY;
    PF_Iterate8Suite1* iterate_suite =
        AcquireSuite<PF_Iterate8Suite1>(basic, kPFIterate8Suite, kPFIterate8SuiteVersion1);

    PF_Err err = PF_Err_NONE;
    {
        AeAllocator allocator(in_data, handle_suite);
        AeTaskRunner runner(in_data, iterate_suite);

        const float dsx = RationalToFloat(in_data->downsample_x);
        const float dsy = RationalToFloat(in_data->downsample_y);
        const float par = RationalToFloat(in_data->pixel_aspect_ratio);

        CosmicRender render;
        render.source = MakeHostImage(input_world, depth);
        render.source_left = source_left;
        render.source_top = source_top;
        render.dest = MakeHostImage(output_world, depth);
        render.dest_left = dest_left;
        render.dest_top = dest_top;
        render.to_full_x = (par > 0.0f ? par : 1.0f) / (dsx > 0.0f ? dsx : 1.0f);
        render.to_full_y = 1.0f / (dsy > 0.0f ? dsy : 1.0f);
        render.blur_scale = BlurScale(in_data);

        switch (RenderCosmic(settings, render, allocator, runner)) {
            case CosmicResult::kOk: break;
            case CosmicResult::kOutOfMemory: err = PF_Err_OUT_OF_MEMORY; break;
            case CosmicResult::kInvalidArguments: err = PF_Err_BAD_CALLBACK_PARAM; break;
        }
    }

    if (iterate_suite != nullptr) basic->ReleaseSuite(kPFIterate8Suite, kPFIterate8SuiteVersion1);
    basic->ReleaseSuite(kPFHandleSuite, kPFHandleSuiteVersion1);
    return err;
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
    extra->output->pre_render_data = data;
    extra->output->delete_pre_render_data_func = DeletePreRenderData;
    return PF_Err_NONE;
}

PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    (void)out_data;
    PF_Err err = PF_Err_NONE;

    PreRenderData* data = static_cast<PreRenderData*>(extra->input->pre_render_data);
    if (data == nullptr) return PF_Err_INTERNAL_STRUCT_DAMAGED;

    PF_EffectWorld* input_world = nullptr;
    PF_EffectWorld* output_world = nullptr;

    if (data->has_input) {
        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, kInputCheckoutId, &input_world));
    }
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output_world));
    if (err) return err;
    if (output_world == nullptr) return PF_Err_NONE;

    A_long output_left = data->output_rect.left;
    A_long output_top = data->output_rect.top;
    ResolveWorldOrigin(output_world, data->output_rect, &output_left, &output_top);

    A_long input_left = 0;
    A_long input_top = 0;
    if (input_world != nullptr && input_world->data != nullptr) {
        input_left = data->input_rect.left;
        input_top = data->input_rect.top;
        ResolveWorldOrigin(input_world, data->input_rect, &input_left, &input_top);
    } else {
        input_world = nullptr;
    }

    const PixelDepth depth = DepthFromBitsPerChannel(extra->input->bitdepth);
    return RunPipeline(in_data, data->params.settings, input_world, static_cast<int>(input_left),
                       static_cast<int>(input_top), output_world, static_cast<int>(output_left),
                       static_cast<int>(output_top), depth);
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
    return RunPipeline(in_data, effect_params.settings, input_world, 0, 0, output,
                       -static_cast<int>(in_data->output_origin_x), -static_cast<int>(in_data->output_origin_y),
                       depth);
}

}  // namespace cosmic
