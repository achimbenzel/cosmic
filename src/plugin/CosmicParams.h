#pragma once

#include "SdkIncludes.h"
#include "core/CosmicPipeline.h"
#include "core/UiModel.h"

namespace cosmic {

// Positions in the parameter list, which is what After Effects indexes the
// params[] array by. Saved projects do not depend on them: After Effects finds
// a parameter's saved value by its disk id (ParamId in CosmicParams.cpp), so
// parameters can be placed anywhere as long as their ids never change. The
// numbers are spelled out so the layout reads at a glance.
enum ParamIndex {
    kParamInput = 0,

    kParamPaletteGroupStart = 1,
    kParamPalette = 2,
    kParamColor1 = 3,
    kParamColor2 = 4,
    kParamColor3 = 5,
    kParamColor4 = 6,
    kParamColor5 = 7,
    kParamColorBlend = 8,
    kParamReverse = 9,
    kParamPaletteGroupEnd = 10,

    kParamGradientGroupStart = 11,
    kParamType = 12,
    kParamFit = 13,
    kParamCenter = 14,
    kParamAngle = 15,
    kParamSize = 16,
    kParamCycles = 17,
    kParamOffset = 18,
    kParamRepeat = 19,
    kParamGradientGroupEnd = 20,

    kParamDepthGroupStart = 21,
    kParamDepthShape = 22,
    kParamDepth = 23,
    kParamDepthCenter = 24,
    kParamDepthRadius = 25,
    kParamBulge = 26,  // v1.2
    kParamRounding = 27,
    kParamSoftness = 28,
    kParamLightAngle = 29,
    kParamContrast = 30,
    kParamDepthGroupEnd = 31,

    kParamTurbulenceGroupStart = 32,
    kParamTurbulence = 33,
    kParamTurbulenceSize = 34,
    kParamComplexity = 35,
    kParamEvolution = 36,
    kParamLoopWithAngle = 37,
    kParamSeed = 38,
    kParamTurbulenceGroupEnd = 39,

    kParamFocusGroupStart = 40,
    kParamFocusPoint = 41,
    kParamFocusRadius = 42,
    kParamFocusFalloff = 43,
    kParamDefocus = 44,
    kParamFocusGroupEnd = 45,

    kParamGlowGroupStart = 46,
    kParamGlowIntensity = 47,
    kParamGlowRadius = 48,
    kParamGlowFalloff = 49,
    kParamGlowThreshold = 50,
    kParamGlowSoftness = 51,
    kParamHighlightProtection = 52,
    kParamGlowGroupEnd = 53,

    kParamDiffusionGroupStart = 54,
    kParamDiffusion = 55,
    kParamDiffusionRadius = 56,
    kParamDiffusionGroupEnd = 57,

    kParamGrainGroupStart = 58,
    kParamGrain = 59,
    kParamGrainSize = 60,
    kParamAnimateGrain = 61,
    kParamGrainGroupEnd = 62,

    kParamCompositeGroupStart = 63,
    kParamMatte = 64,
    kParamBlend = 65,
    kParamOpacity = 66,
    kParamExpandBounds = 67,
    kParamWorkingSpace = 68,
    kParamCompositeGroupEnd = 69,

    kParamPerformanceGroupStart = 70,
    kParamGpu = 71,
    kParamPerformanceGroupEnd = 72,

    kParamAboutGroupStart = 73,
    kParamAboutGroupEnd = 74,

    kParamCount = 75
};

PF_Err SetupParams(PF_InData* in_data, PF_OutData* out_data);

struct EffectParams {
    CosmicSettings settings;
    bool gpu = true;
    bool expand_bounds = true;
    bool animate_grain = false;
    float grain = 0.0f;
};

// Reads every parameter at `in_data->current_time` and converts it into render
// settings for the layer this call is about.
PF_Err ReadParams(PF_InData* in_data, EffectParams* out_params);

// Palette popup <-> colour controls: picking a palette fills the colours,
// editing a colour switches the popup to Custom.
PF_Err UserChangedParam(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[],
                        const PF_UserChangedParamExtra* extra);

// Only animated grain makes a frame depend on time alone.
PF_Err QueryDynamicFlags(PF_InData* in_data, PF_OutData* out_data);

// Full-resolution pixel -> render pixel, for blur sizes. Isotropic: the mean of
// the two axes when downsampling or the pixel aspect makes them differ.
float BlurScale(const PF_InData* in_data);

}  // namespace cosmic
