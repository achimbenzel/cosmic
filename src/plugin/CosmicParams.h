#pragma once

#include "SdkIncludes.h"
#include "core/CosmicPipeline.h"
#include "core/UiModel.h"

namespace cosmic {

// Parameter order is part of the saved project format: once shipped, append,
// never insert or reorder, or projects saved by an earlier build open with
// their values against the wrong parameters. The numbers are spelled out so a
// stray insertion is a compile error rather than a corrupted project.
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
    kParamDepthGroupEnd = 26,

    kParamTurbulenceGroupStart = 27,
    kParamTurbulence = 28,
    kParamTurbulenceSize = 29,
    kParamComplexity = 30,
    kParamEvolution = 31,
    kParamLoopWithAngle = 32,
    kParamSeed = 33,
    kParamTurbulenceGroupEnd = 34,

    kParamFocusGroupStart = 35,
    kParamFocusPoint = 36,
    kParamFocusRadius = 37,
    kParamFocusFalloff = 38,
    kParamDefocus = 39,
    kParamFocusGroupEnd = 40,

    kParamGlowGroupStart = 41,
    kParamGlowIntensity = 42,
    kParamGlowRadius = 43,
    kParamGlowFalloff = 44,
    kParamGlowThreshold = 45,
    kParamGlowSoftness = 46,
    kParamHighlightProtection = 47,
    kParamGlowGroupEnd = 48,

    kParamDiffusionGroupStart = 49,
    kParamDiffusion = 50,
    kParamDiffusionRadius = 51,
    kParamDiffusionGroupEnd = 52,

    kParamGrainGroupStart = 53,
    kParamGrain = 54,
    kParamGrainSize = 55,
    kParamAnimateGrain = 56,
    kParamGrainGroupEnd = 57,

    kParamCompositeGroupStart = 58,
    kParamMatte = 59,
    kParamBlend = 60,
    kParamOpacity = 61,
    kParamExpandBounds = 62,
    kParamWorkingSpace = 63,
    kParamCompositeGroupEnd = 64,

    kParamAboutGroupStart = 65,
    kParamAboutGroupEnd = 66,

    // Appended in v1.1. Everything above is the v1.0 layout and must stay put:
    // saved projects find their values by these positions and ids.
    kParamPerformanceGroupStart = 67,
    kParamGpu = 68,
    kParamPerformanceGroupEnd = 69,

    kParamCount = 70
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
