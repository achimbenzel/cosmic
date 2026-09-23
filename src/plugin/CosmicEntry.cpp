#include "Cosmic.h"
#include "CosmicParams.h"
#include "CosmicPiPLFlags.h"
#include "CosmicRender.h"
#include "SdkIncludes.h"

// The PiPL resource and PF_Cmd_GLOBAL_SETUP must advertise the same behaviour,
// so both read the flags from the generated header.
static_assert(COSMIC_OUT_FLAGS ==
                  (PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_I_EXPAND_BUFFER | PF_OutFlag_NON_PARAM_VARY),
              "PiPL out flags are out of sync with GlobalSetup");
static_assert(COSMIC_OUT_FLAGS2 == (PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG | PF_OutFlag2_SUPPORTS_SMART_RENDER |
                                    PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
                                    PF_OutFlag2_SUPPORTS_QUERY_DYNAMIC_FLAGS),
              "PiPL out flags 2 are out of sync with GlobalSetup");

namespace {

PF_Err About(PF_InData* in_data, PF_OutData* out_data) {
    PF_SPRINTF(out_data->return_msg, "%s v%s (%s)\r%s", COSMIC_NAME, COSMIC_VERSION_STRING, COSMIC_BUILD_ID,
               COSMIC_DESCRIPTION);
    (void)in_data;
    return PF_Err_NONE;
}

PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data) {
    (void)in_data;
    out_data->my_version = PF_VERSION(COSMIC_VERSION_MAJOR, COSMIC_VERSION_MINOR, COSMIC_VERSION_BUG,
                                      PF_Stage_RELEASE, COSMIC_VERSION_BUILD);
    // NON_PARAM_VARY is dynamic: PF_Cmd_QUERY_DYNAMIC_FLAGS clears it unless
    // the grain animates, so still frames are cached normally.
    out_data->out_flags = COSMIC_OUT_FLAGS;
    out_data->out_flags2 = COSMIC_OUT_FLAGS2;
    return PF_Err_NONE;
}

}  // namespace

extern "C" DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[],
                                       PF_LayerDef* output, void* extra) {
    PF_Err err = PF_Err_NONE;
    // After Effects is a C host: no exception may cross this boundary.
    try {
        switch (cmd) {
            case PF_Cmd_ABOUT:
                err = About(in_data, out_data);
                break;
            case PF_Cmd_GLOBAL_SETUP:
                err = GlobalSetup(in_data, out_data);
                break;
            case PF_Cmd_PARAMS_SETUP:
                err = cosmic::SetupParams(in_data, out_data);
                break;
            case PF_Cmd_USER_CHANGED_PARAM:
                err = cosmic::UserChangedParam(in_data, out_data, params,
                                               static_cast<const PF_UserChangedParamExtra*>(extra));
                break;
            case PF_Cmd_QUERY_DYNAMIC_FLAGS:
                err = cosmic::QueryDynamicFlags(in_data, out_data);
                break;
            case PF_Cmd_FRAME_SETUP:
                err = cosmic::LegacyFrameSetup(in_data, out_data);
                break;
            case PF_Cmd_RENDER:
                err = cosmic::LegacyRender(in_data, out_data, params, output);
                break;
            case PF_Cmd_SMART_PRE_RENDER:
                err = cosmic::SmartPreRender(in_data, out_data, static_cast<PF_PreRenderExtra*>(extra));
                break;
            case PF_Cmd_SMART_RENDER:
                err = cosmic::SmartRender(in_data, out_data, static_cast<PF_SmartRenderExtra*>(extra));
                break;
            default:
                break;
        }
    } catch (const std::bad_alloc&) {
        err = PF_Err_OUT_OF_MEMORY;
    } catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}

// PF_REGISTER_EFFECT_EXT2 expands to the SDK's 'eFKT' four character constant.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmultichar"
#endif

extern "C" DllExport PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr, PF_PluginDataCB2 inPluginDataCallBackPtr,
                                                     SPBasicSuite* inSPBasicSuitePtr, const char* inHostName,
                                                     const char* inHostVersion) {
    (void)inSPBasicSuitePtr;
    (void)inHostName;
    (void)inHostVersion;

    PF_Err result = PF_Err_INVALID_CALLBACK;
    PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, COSMIC_NAME, COSMIC_MATCH_NAME, COSMIC_CATEGORY,
                            AE_RESERVED_INFO, "EffectMain", "");
    return result;
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
