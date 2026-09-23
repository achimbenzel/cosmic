// Minimal After Effects stand-in: loads CosmicGradient.aex and drives the same
// command sequence the host does, so the plug-in layer can be exercised
// without launching After Effects.
//
// Usage: cosmic_mock_host <path to CosmicGradient.aex> [output directory]

#include <windows.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    kIndexDefocus = 39,
    kIndexGlowIntensity = 42,
    kIndexGlowRadius = 43,
    kIndexGrain = 54,
    kIndexAnimateGrain = 56,
    kIndexMatte = 59,
    kIndexExpandBounds = 62,
    kParamCountAsShipped = 67
};

constexpr A_long kExpectedOutFlags = PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_I_EXPAND_BUFFER | PF_OutFlag_NON_PARAM_VARY;
constexpr A_long kExpectedOutFlags2 = PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG | PF_OutFlag2_SUPPORTS_SMART_RENDER |
                                      PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
                                      PF_OutFlag2_SUPPORTS_QUERY_DYNAMIC_FLAGS;

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

SPErr MockAcquireSuite(const char* name, int32_t version, const void** suite) {
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
    Check(out_data->num_params == kParamCountAsShipped, "the shipped parameter layout is unchanged");

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
};

bool RunRender(EffectMainFn effect_main, const RenderOptions& options, const std::string& out_dir) {
    PF_InData in_data;
    PF_OutData out_data;
    PF_UtilCallbacks utils;
    MockHost* host = NewInstance(effect_main, &in_data, &out_data, &utils, options.width, options.height);
    host->downsample = options.downsample;
    in_data.downsample_x.den = options.downsample;
    in_data.downsample_y.den = options.downsample;
    in_data.current_time = 12;

    Param(host, kIndexExpandBounds).u.bd.value = options.expand_bounds ? TRUE : FALSE;
    Param(host, kIndexMatte).u.pd.value = options.matte;
    Param(host, kIndexType).u.pd.value = options.type;
    Param(host, kIndexDefocus).u.fs_d.value = options.defocus;
    Param(host, kIndexGlowRadius).u.fs_d.value = options.glow_radius;
    Param(host, kIndexAnimateGrain).u.bd.value = options.animate_grain ? TRUE : FALSE;

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

    g_host = nullptr;
    delete host;
    return true;
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

    std::vector<PF_ParamDef*> params(kParamCountAsShipped, nullptr);
    PF_ParamDef input = {};
    params[0] = &input;
    for (int i = 1; i < kParamCountAsShipped; ++i) params[static_cast<std::size_t>(i)] = &Param(host, i);

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

    g_basic_suite.AcquireSuite = MockAcquireSuite;
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

    FreeLibrary(module);
    std::printf("%s\n", g_failures == 0 ? "mock host: all checks passed" : "mock host: FAILURES");
    return g_failures == 0 ? 0 : 1;
}
