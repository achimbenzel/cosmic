#include "CosmicParams.h"

#include <algorithm>
#include <cmath>

#include "AeAdapters.h"
#include "Cosmic.h"
#include "core/Palette.h"

namespace cosmic {
namespace {

// Disk ids: how After Effects finds a parameter's saved value when a project
// is reopened, whatever position the parameter has now. An id must never
// change or be reused once shipped; anything new takes the next free number.
// v1.0's ids equal its positions; later versions insert parameters where they
// belong in the panel, with new ids.
enum ParamId {
    kIdPaletteGroup = 1,
    kIdPalette = 2,
    kIdColor1 = 3,
    kIdColor2 = 4,
    kIdColor3 = 5,
    kIdColor4 = 6,
    kIdColor5 = 7,
    kIdColorBlend = 8,
    kIdReverse = 9,
    kIdPaletteGroupEnd = 10,
    kIdGradientGroup = 11,
    kIdType = 12,
    kIdFit = 13,
    kIdCenter = 14,
    kIdAngle = 15,
    kIdSize = 16,
    kIdCycles = 17,
    kIdOffset = 18,
    kIdRepeat = 19,
    kIdGradientGroupEnd = 20,
    kIdDepthGroup = 21,
    kIdDepthShape = 22,
    kIdDepth = 23,
    kIdDepthCenter = 24,
    kIdDepthRadius = 25,
    kIdDepthGroupEnd = 26,
    kIdTurbulenceGroup = 27,
    kIdTurbulence = 28,
    kIdTurbulenceSize = 29,
    kIdComplexity = 30,
    kIdEvolution = 31,
    kIdLoopWithAngle = 32,
    kIdSeed = 33,
    kIdTurbulenceGroupEnd = 34,
    kIdFocusGroup = 35,
    kIdFocusPoint = 36,
    kIdFocusRadius = 37,
    kIdFocusFalloff = 38,
    kIdDefocus = 39,
    kIdFocusGroupEnd = 40,
    kIdGlowGroup = 41,
    kIdGlowIntensity = 42,
    kIdGlowRadius = 43,
    kIdGlowFalloff = 44,
    kIdGlowThreshold = 45,
    kIdGlowSoftness = 46,
    kIdHighlightProtection = 47,
    kIdGlowGroupEnd = 48,
    kIdDiffusionGroup = 49,
    kIdDiffusion = 50,
    kIdDiffusionRadius = 51,
    kIdDiffusionGroupEnd = 52,
    kIdGrainGroup = 53,
    kIdGrain = 54,
    kIdGrainSize = 55,
    kIdAnimateGrain = 56,
    kIdGrainGroupEnd = 57,
    kIdCompositeGroup = 58,
    kIdMatte = 59,
    kIdBlend = 60,
    kIdOpacity = 61,
    kIdExpandBounds = 62,
    kIdWorkingSpace = 63,
    kIdCompositeGroupEnd = 64,
    kIdAboutGroup = 65,
    kIdAboutGroupEnd = 66,
    // v1.1
    kIdPerformanceGroup = 67,
    kIdGpu = 68,
    kIdPerformanceGroupEnd = 69,
    // v1.2
    kIdBulge = 70,
    kIdRounding = 71,
    kIdSoftness = 72,
    kIdLightAngle = 73,
    kIdContrast = 74
};

// The id at each position, in the order SetupParams adds them.
constexpr int kIdAt[kParamCount] = {
    0,  // the input layer, which After Effects adds
    kIdPaletteGroup, kIdPalette, kIdColor1, kIdColor2, kIdColor3, kIdColor4, kIdColor5, kIdColorBlend,
    kIdReverse, kIdPaletteGroupEnd,
    kIdGradientGroup, kIdType, kIdFit, kIdCenter, kIdAngle, kIdSize, kIdCycles, kIdOffset, kIdRepeat,
    kIdGradientGroupEnd,
    kIdDepthGroup, kIdDepthShape, kIdDepth, kIdDepthCenter, kIdDepthRadius, kIdBulge, kIdRounding, kIdSoftness,
    kIdLightAngle, kIdContrast, kIdDepthGroupEnd,
    kIdTurbulenceGroup, kIdTurbulence, kIdTurbulenceSize, kIdComplexity, kIdEvolution, kIdLoopWithAngle, kIdSeed,
    kIdTurbulenceGroupEnd,
    kIdFocusGroup, kIdFocusPoint, kIdFocusRadius, kIdFocusFalloff, kIdDefocus, kIdFocusGroupEnd,
    kIdGlowGroup, kIdGlowIntensity, kIdGlowRadius, kIdGlowFalloff, kIdGlowThreshold, kIdGlowSoftness,
    kIdHighlightProtection, kIdGlowGroupEnd,
    kIdDiffusionGroup, kIdDiffusion, kIdDiffusionRadius, kIdDiffusionGroupEnd,
    kIdGrainGroup, kIdGrain, kIdGrainSize, kIdAnimateGrain, kIdGrainGroupEnd,
    kIdCompositeGroup, kIdMatte, kIdBlend, kIdOpacity, kIdExpandBounds, kIdWorkingSpace, kIdCompositeGroupEnd,
    kIdPerformanceGroup, kIdGpu, kIdPerformanceGroupEnd,
    kIdAboutGroup, kIdAboutGroupEnd,
};

// Every id exactly once, and each position holding the id its name says.
constexpr bool IdsAreUnique() {
    for (int i = 1; i < kParamCount; ++i) {
        if (kIdAt[i] < 1 || kIdAt[i] > 9999) return false;
        for (int j = i + 1; j < kParamCount; ++j) {
            if (kIdAt[i] == kIdAt[j]) return false;
        }
    }
    return true;
}
static_assert(IdsAreUnique(), "disk ids must be unique and in 1..9999");
static_assert(kIdAt[kParamPalette] == kIdPalette && kIdAt[kParamColor5] == kIdColor5 &&
                  kIdAt[kParamDepthShape] == kIdDepthShape && kIdAt[kParamDepthRadius] == kIdDepthRadius &&
                  kIdAt[kParamBulge] == kIdBulge && kIdAt[kParamContrast] == kIdContrast &&
                  kIdAt[kParamDepthGroupEnd] == kIdDepthGroupEnd && kIdAt[kParamTurbulence] == kIdTurbulence &&
                  kIdAt[kParamLoopWithAngle] == kIdLoopWithAngle && kIdAt[kParamDefocus] == kIdDefocus &&
                  kIdAt[kParamGlowIntensity] == kIdGlowIntensity && kIdAt[kParamAnimateGrain] == kIdAnimateGrain &&
                  kIdAt[kParamMatte] == kIdMatte && kIdAt[kParamWorkingSpace] == kIdWorkingSpace &&
                  kIdAt[kParamGpu] == kIdGpu && kIdAt[kParamAboutGroupEnd] == kIdAboutGroupEnd,
              "ParamIndex and kIdAt disagree");
static_assert(kParamColor5 - kParamColor1 == kStopCount - 1, "one colour control per palette stop");

// Popup strings. Their order is the enums' order in Shared.h, CosmicPipeline.h
// and Palette.h, and a popup saves its position, so it is part of the project
// format too: append only (renaming an entry is fine).
constexpr char kColorBlendChoices[] = "Oklab Smooth|Oklab|Linear Light|sRGB";
constexpr char kTypeChoices[] = "Linear|Radial|Conic|Diamond|Reflected";
constexpr char kFitChoices[] = "Content Bounds|Layer";
constexpr char kRepeatChoices[] = "None|Repeat|Mirror";
// The fifth, appended in v1.1 as "Bulge", is the lens: Bulge is now the
// relief control.
constexpr char kDepthShapeChoices[] = "Dome|Sphere|Ridge|Wave|Lens";
constexpr char kMatteChoices[] = "Layer Alpha|Inverted Alpha|Full Frame";
constexpr char kBlendChoices[] = "Normal|Multiply|Screen|Overlay|Color";
constexpr char kWorkingSpaceChoices[] = "Auto|Linear|sRGB";

constexpr float kFixedToFloat = 1.0f / 65536.0f;

int CustomPaletteIndex() { return PresetCount() + 1; }

float RationalToFloat(const PF_RationalScale& value) {
    if (value.den == 0) return 1.0f;
    return static_cast<float>(value.num) / static_cast<float>(value.den);
}

// Checks a parameter out at the current time, hands it to `read`, and checks
// it back in whatever happened.
template <typename Read>
PF_Err Checkout(PF_InData* in_data, int index, const Read& read) {
    PF_ParamDef param;
    AEFX_CLR_STRUCT(param);
    PF_Err err = PF_CHECKOUT_PARAM(in_data, index, in_data->current_time, in_data->time_step, in_data->time_scale,
                                   &param);
    if (!err) read(param);
    PF_Err checkin = PF_CHECKIN_PARAM(in_data, &param);
    return err ? err : checkin;
}

PF_Err ReadFloat(PF_InData* in_data, int index, float* out) {
    return Checkout(in_data, index, [&](const PF_ParamDef& p) { *out = static_cast<float>(p.u.fs_d.value); });
}

PF_Err ReadPopup(PF_InData* in_data, int index, int* out) {
    return Checkout(in_data, index, [&](const PF_ParamDef& p) { *out = static_cast<int>(p.u.pd.value); });
}

PF_Err ReadCheckbox(PF_InData* in_data, int index, bool* out) {
    return Checkout(in_data, index, [&](const PF_ParamDef& p) { *out = p.u.bd.value != 0; });
}

PF_Err ReadSlider(PF_InData* in_data, int index, int* out) {
    return Checkout(in_data, index, [&](const PF_ParamDef& p) { *out = static_cast<int>(p.u.sd.value); });
}

PF_Err ReadAngle(PF_InData* in_data, int index, float* out_degrees) {
    return Checkout(in_data, index,
                    [&](const PF_ParamDef& p) { *out_degrees = static_cast<float>(p.u.ad.value) * kFixedToFloat; });
}

// Point controls arrive in the input buffer's coordinates, already scaled for
// the downsample factor; the render wants full-resolution square pixels.
PF_Err ReadPoint(PF_InData* in_data, int index, float* out_x, float* out_y) {
    const float dsx = RationalToFloat(in_data->downsample_x);
    const float dsy = RationalToFloat(in_data->downsample_y);
    const float par = RationalToFloat(in_data->pixel_aspect_ratio);
    return Checkout(in_data, index, [&](const PF_ParamDef& p) {
        *out_x = static_cast<float>(p.u.td.x_value) * kFixedToFloat / (dsx > 0.0f ? dsx : 1.0f) * par;
        *out_y = static_cast<float>(p.u.td.y_value) * kFixedToFloat / (dsy > 0.0f ? dsy : 1.0f);
    });
}

// Colour controls hold display-referred sRGB; the 8-bit value is what the
// swatch shows, which is what a palette is designed in.
PF_Err ReadColor(PF_InData* in_data, int index, std::uint8_t* out_rgb) {
    return Checkout(in_data, index, [&](const PF_ParamDef& p) {
        out_rgb[0] = p.u.cd.value.red;
        out_rgb[1] = p.u.cd.value.green;
        out_rgb[2] = p.u.cd.value.blue;
    });
}

}  // namespace

float BlurScale(const PF_InData* in_data) {
    const float dsx = RationalToFloat(in_data->downsample_x);
    const float dsy = RationalToFloat(in_data->downsample_y);
    const float par = RationalToFloat(in_data->pixel_aspect_ratio);
    const float scale = dsy * dsx / (par > 0.0f ? par : 1.0f);
    return scale > 0.0f ? std::sqrt(scale) : 1.0f;
}

PF_Err SetupParams(PF_InData* in_data, PF_OutData* out_data) {
    const UiValues d;  // defaults
    PF_ParamDef def;

    // --- Palette -------------------------------------------------------------
    PF_ADD_TOPICX("Palette", 0, kIdPaletteGroup);
    PF_ADD_POPUPX("Palette", static_cast<short>(PresetCount() + 1), d.palette, PresetPopupString(),
                  PF_ParamFlag_SUPERVISE, kIdPalette);
    const char* color_names[kStopCount] = {"Color 1", "Color 2", "Color 3", "Color 4", "Color 5"};
    const int color_ids[kStopCount] = {kIdColor1, kIdColor2, kIdColor3, kIdColor4, kIdColor5};
    for (int k = 0; k < kStopCount; ++k) {
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_SUPERVISE;
        PF_ADD_COLOR(color_names[k], d.colors[k][0], d.colors[k][1], d.colors[k][2], color_ids[k]);
    }
    PF_ADD_POPUPX("Color Blend", 4, d.color_blend, kColorBlendChoices, 0, kIdColorBlend);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Reverse Palette", d.reverse ? TRUE : FALSE, 0, kIdReverse);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(kIdPaletteGroupEnd);

    // --- Gradient ------------------------------------------------------------
    PF_ADD_TOPICX("Gradient", 0, kIdGradientGroup);
    PF_ADD_POPUPX("Type", 5, d.gradient_type, kTypeChoices, 0, kIdType);
    PF_ADD_POPUPX("Fit", 2, d.fit, kFitChoices, 0, kIdFit);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT("Center", kDefaultCenterPct[0], kDefaultCenterPct[1], FALSE, kIdCenter);
    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Angle", d.angle_deg, kIdAngle);
    PF_ADD_FLOAT_SLIDERX("Size", 1.0f, 2000.0f, 10.0f, 300.0f, d.size_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdSize);
    PF_ADD_FLOAT_SLIDERX("Cycles", 0.0f, 50.0f, 0.0f, 8.0f, d.cycles, PF_Precision_HUNDREDTHS, 0, 0, kIdCycles);
    PF_ADD_FLOAT_SLIDERX("Offset", -100000.0f, 100000.0f, -100.0f, 100.0f, d.offset_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdOffset);
    PF_ADD_POPUPX("Repeat", 3, d.repeat, kRepeatChoices, 0, kIdRepeat);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(kIdGradientGroupEnd);

    // --- Depth ---------------------------------------------------------------
    PF_ADD_TOPICX("Depth", 0, kIdDepthGroup);
    PF_ADD_POPUPX("Depth Shape", 5, d.depth_shape, kDepthShapeChoices, 0, kIdDepthShape);
    PF_ADD_FLOAT_SLIDERX("Depth", -1000.0f, 1000.0f, -100.0f, 100.0f, d.depth_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdDepth);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT("Depth Center", kDefaultDepthCenterPct[0], kDefaultDepthCenterPct[1], FALSE, kIdDepthCenter);
    PF_ADD_FLOAT_SLIDERX("Depth Radius", 1.0f, 2000.0f, 5.0f, 200.0f, d.depth_radius_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdDepthRadius);
    // v1.2: the layer's own shape raised into a glass relief.
    PF_ADD_FLOAT_SLIDERX("Bulge", 0.0f, 400.0f, 0.0f, 200.0f, d.bulge_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdBulge);
    PF_ADD_FLOAT_SLIDERX("Rounding", 0.0f, 100.0f, 0.0f, 100.0f, d.rounding_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdRounding);
    PF_ADD_FLOAT_SLIDERX("Softness", 1.0f, 1000.0f, 10.0f, 400.0f, d.softness_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdSoftness);
    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Light Angle", d.light_angle_deg, kIdLightAngle);
    PF_ADD_FLOAT_SLIDERX("Contrast", 0.0f, 400.0f, 0.0f, 200.0f, d.contrast_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdContrast);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(kIdDepthGroupEnd);

    // --- Turbulence ----------------------------------------------------------
    PF_ADD_TOPICX("Turbulence", PF_ParamFlag_START_COLLAPSED, kIdTurbulenceGroup);
    PF_ADD_FLOAT_SLIDERX("Turbulence", 0.0f, 500.0f, 0.0f, 30.0f, d.turbulence_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdTurbulence);
    PF_ADD_FLOAT_SLIDERX("Turbulence Size", 1.0f, 2000.0f, 5.0f, 200.0f, d.turbulence_size_pct,
                         PF_Precision_TENTHS, PF_ValueDisplayFlag_PERCENT, 0, kIdTurbulenceSize);
    PF_ADD_FLOAT_SLIDERX("Complexity", 1.0f, 8.0f, 1.0f, 8.0f, d.complexity, PF_Precision_TENTHS, 0, 0,
                         kIdComplexity);
    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Evolution", d.evolution_deg, kIdEvolution);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Loop With Angle", d.evolve_with_angle ? TRUE : FALSE, 0, kIdLoopWithAngle);
    AEFX_CLR_STRUCT(def);
    PF_ADD_SLIDER("Random Seed", 0, 100000, 0, 1000, d.seed, kIdSeed);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(kIdTurbulenceGroupEnd);

    // --- Focus ---------------------------------------------------------------
    PF_ADD_TOPICX("Focus", PF_ParamFlag_START_COLLAPSED, kIdFocusGroup);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT("Focus Point", kDefaultFocusPct[0], kDefaultFocusPct[1], FALSE, kIdFocusPoint);
    PF_ADD_FLOAT_SLIDERX("Focus Radius", 0.0f, 1000.0f, 0.0f, 100.0f, d.focus_radius_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdFocusRadius);
    PF_ADD_FLOAT_SLIDERX("Focus Falloff", 0.0f, 1000.0f, 0.0f, 200.0f, d.focus_falloff_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdFocusFalloff);
    PF_ADD_FLOAT_SLIDERX("Defocus", 0.0f, 2000.0f, 0.0f, 100.0f, d.defocus_px, PF_Precision_TENTHS, 0, 0,
                         kIdDefocus);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(kIdFocusGroupEnd);

    // --- Glow ----------------------------------------------------------------
    PF_ADD_TOPICX("Glow", 0, kIdGlowGroup);
    PF_ADD_FLOAT_SLIDERX("Glow Intensity", 0.0f, 5000.0f, 0.0f, 300.0f, d.glow_intensity_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdGlowIntensity);
    PF_ADD_FLOAT_SLIDERX("Glow Radius", 0.0f, 4000.0f, 0.0f, 500.0f, d.glow_radius_px, PF_Precision_TENTHS, 0, 0,
                         kIdGlowRadius);
    PF_ADD_FLOAT_SLIDERX("Glow Falloff", 1.0f, 3.0f, 1.0f, 3.0f, d.glow_falloff, PF_Precision_HUNDREDTHS, 0, 0,
                         kIdGlowFalloff);
    PF_ADD_FLOAT_SLIDERX("Glow Threshold", 0.0f, 4.0f, 0.0f, 1.5f, d.glow_threshold, PF_Precision_THOUSANDTHS, 0,
                         0, kIdGlowThreshold);
    PF_ADD_FLOAT_SLIDERX("Threshold Softness", 0.0f, 100.0f, 0.0f, 100.0f, d.glow_softness_pct,
                         PF_Precision_TENTHS, PF_ValueDisplayFlag_PERCENT, 0, kIdGlowSoftness);
    PF_ADD_FLOAT_SLIDERX("Highlight Protection", 0.0f, 100.0f, 0.0f, 100.0f, d.protection_pct,
                         PF_Precision_TENTHS, PF_ValueDisplayFlag_PERCENT, 0, kIdHighlightProtection);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(kIdGlowGroupEnd);

    // --- Optical diffusion ---------------------------------------------------
    PF_ADD_TOPICX("Optical Diffusion", PF_ParamFlag_START_COLLAPSED, kIdDiffusionGroup);
    PF_ADD_FLOAT_SLIDERX("Diffusion", 0.0f, 100.0f, 0.0f, 100.0f, d.diffusion_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdDiffusion);
    PF_ADD_FLOAT_SLIDERX("Diffusion Radius", 0.0f, 2000.0f, 0.0f, 300.0f, d.diffusion_radius_px,
                         PF_Precision_TENTHS, 0, 0, kIdDiffusionRadius);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(kIdDiffusionGroupEnd);

    // --- Grain ---------------------------------------------------------------
    PF_ADD_TOPICX("Grain", PF_ParamFlag_START_COLLAPSED, kIdGrainGroup);
    PF_ADD_FLOAT_SLIDERX("Grain Amount", 0.0f, 100.0f, 0.0f, 40.0f, d.grain_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdGrain);
    PF_ADD_FLOAT_SLIDERX("Grain Size", 0.1f, 20.0f, 0.5f, 5.0f, d.grain_size_px, PF_Precision_HUNDREDTHS, 0, 0,
                         kIdGrainSize);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Animate Grain", d.animate_grain ? TRUE : FALSE, 0, kIdAnimateGrain);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(kIdGrainGroupEnd);

    // --- Composite -----------------------------------------------------------
    PF_ADD_TOPICX("Composite", PF_ParamFlag_START_COLLAPSED, kIdCompositeGroup);
    PF_ADD_POPUPX("Matte", 3, d.matte, kMatteChoices, 0, kIdMatte);
    PF_ADD_POPUPX("Blend Mode", 5, d.blend, kBlendChoices, 0, kIdBlend);
    PF_ADD_FLOAT_SLIDERX("Opacity", 0.0f, 100.0f, 0.0f, 100.0f, d.opacity_pct, PF_Precision_TENTHS,
                         PF_ValueDisplayFlag_PERCENT, 0, kIdOpacity);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Expand Bounds", d.expand_bounds ? TRUE : FALSE, 0, kIdExpandBounds);
    PF_ADD_POPUPX("Working Space", 3, d.working_space, kWorkingSpaceChoices, 0, kIdWorkingSpace);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(kIdCompositeGroupEnd);

    // --- Performance ------------------------------------------------------
    PF_ADD_TOPICX("Performance", PF_ParamFlag_START_COLLAPSED, kIdPerformanceGroup);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("GPU Acceleration", d.gpu ? TRUE : FALSE, 0, kIdGpu);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(kIdPerformanceGroupEnd);

    // An empty group whose header is the build, so the loaded build can be
    // identified without leaving the panel.
    PF_ADD_TOPICX("v" COSMIC_VERSION_STRING " (" COSMIC_BUILD_ID ")", PF_ParamFlag_START_COLLAPSED,
                  kIdAboutGroup);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(kIdAboutGroupEnd);

    out_data->num_params = kParamCount;
    return PF_Err_NONE;
}

PF_Err ReadParams(PF_InData* in_data, EffectParams* out_params) {
    PF_Err err = PF_Err_NONE;
    UiValues ui;

    if (!err) err = ReadPopup(in_data, kParamPalette, &ui.palette);
    for (int k = 0; k < kStopCount && !err; ++k) err = ReadColor(in_data, kParamColor1 + k, ui.colors[k]);
    if (!err) err = ReadPopup(in_data, kParamColorBlend, &ui.color_blend);
    if (!err) err = ReadCheckbox(in_data, kParamReverse, &ui.reverse);

    if (!err) err = ReadPopup(in_data, kParamType, &ui.gradient_type);
    if (!err) err = ReadPopup(in_data, kParamFit, &ui.fit);
    if (!err) err = ReadPoint(in_data, kParamCenter, &ui.center_x, &ui.center_y);
    if (!err) err = ReadAngle(in_data, kParamAngle, &ui.angle_deg);
    if (!err) err = ReadFloat(in_data, kParamSize, &ui.size_pct);
    if (!err) err = ReadFloat(in_data, kParamCycles, &ui.cycles);
    if (!err) err = ReadFloat(in_data, kParamOffset, &ui.offset_pct);
    if (!err) err = ReadPopup(in_data, kParamRepeat, &ui.repeat);

    if (!err) err = ReadPopup(in_data, kParamDepthShape, &ui.depth_shape);
    if (!err) err = ReadFloat(in_data, kParamDepth, &ui.depth_pct);
    if (!err) err = ReadPoint(in_data, kParamDepthCenter, &ui.depth_x, &ui.depth_y);
    if (!err) err = ReadFloat(in_data, kParamDepthRadius, &ui.depth_radius_pct);
    if (!err) err = ReadFloat(in_data, kParamBulge, &ui.bulge_pct);
    if (!err) err = ReadFloat(in_data, kParamRounding, &ui.rounding_pct);
    if (!err) err = ReadFloat(in_data, kParamSoftness, &ui.softness_pct);
    if (!err) err = ReadAngle(in_data, kParamLightAngle, &ui.light_angle_deg);
    if (!err) err = ReadFloat(in_data, kParamContrast, &ui.contrast_pct);

    if (!err) err = ReadFloat(in_data, kParamTurbulence, &ui.turbulence_pct);
    if (!err) err = ReadFloat(in_data, kParamTurbulenceSize, &ui.turbulence_size_pct);
    if (!err) err = ReadFloat(in_data, kParamComplexity, &ui.complexity);
    if (!err) err = ReadAngle(in_data, kParamEvolution, &ui.evolution_deg);
    if (!err) err = ReadCheckbox(in_data, kParamLoopWithAngle, &ui.evolve_with_angle);
    if (!err) err = ReadSlider(in_data, kParamSeed, &ui.seed);

    if (!err) err = ReadPoint(in_data, kParamFocusPoint, &ui.focus_x, &ui.focus_y);
    if (!err) err = ReadFloat(in_data, kParamFocusRadius, &ui.focus_radius_pct);
    if (!err) err = ReadFloat(in_data, kParamFocusFalloff, &ui.focus_falloff_pct);
    if (!err) err = ReadFloat(in_data, kParamDefocus, &ui.defocus_px);

    if (!err) err = ReadFloat(in_data, kParamGlowIntensity, &ui.glow_intensity_pct);
    if (!err) err = ReadFloat(in_data, kParamGlowRadius, &ui.glow_radius_px);
    if (!err) err = ReadFloat(in_data, kParamGlowFalloff, &ui.glow_falloff);
    if (!err) err = ReadFloat(in_data, kParamGlowThreshold, &ui.glow_threshold);
    if (!err) err = ReadFloat(in_data, kParamGlowSoftness, &ui.glow_softness_pct);
    if (!err) err = ReadFloat(in_data, kParamHighlightProtection, &ui.protection_pct);

    if (!err) err = ReadFloat(in_data, kParamDiffusion, &ui.diffusion_pct);
    if (!err) err = ReadFloat(in_data, kParamDiffusionRadius, &ui.diffusion_radius_px);

    if (!err) err = ReadFloat(in_data, kParamGrain, &ui.grain_pct);
    if (!err) err = ReadFloat(in_data, kParamGrainSize, &ui.grain_size_px);
    if (!err) err = ReadCheckbox(in_data, kParamAnimateGrain, &ui.animate_grain);

    if (!err) err = ReadPopup(in_data, kParamMatte, &ui.matte);
    if (!err) err = ReadPopup(in_data, kParamBlend, &ui.blend);
    if (!err) err = ReadFloat(in_data, kParamOpacity, &ui.opacity_pct);
    if (!err) err = ReadCheckbox(in_data, kParamExpandBounds, &ui.expand_bounds);
    if (!err) err = ReadPopup(in_data, kParamWorkingSpace, &ui.working_space);
    if (!err) err = ReadCheckbox(in_data, kParamGpu, &ui.gpu);
    if (err) return err;

    // The layer's own size, before any effect or downsampling: the frame the
    // point controls are placed in.
    const float par = RationalToFloat(in_data->pixel_aspect_ratio);
    const float layer_width = static_cast<float>(in_data->width) * (par > 0.0f ? par : 1.0f);
    const float layer_height = static_cast<float>(in_data->height);
    const int frame = in_data->time_step != 0 ? static_cast<int>(in_data->current_time / in_data->time_step) : 0;

    EffectParams params;
    params.settings = SettingsFromUi(ui, layer_width, layer_height, frame);
    params.gpu = ui.gpu;
    params.expand_bounds = ui.expand_bounds;
    params.animate_grain = ui.animate_grain;
    params.grain = ui.grain_pct;
    *out_params = params;
    return PF_Err_NONE;
}

PF_Err UserChangedParam(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[],
                        const PF_UserChangedParamExtra* extra) {
    (void)in_data;
    (void)out_data;
    if (extra == nullptr || params == nullptr) return PF_Err_NONE;
    const PF_ParamIndex index = extra->param_index;

    if (index == kParamPalette) {
        const int palette = static_cast<int>(params[kParamPalette]->u.pd.value);
        if (palette < 1 || palette > PresetCount()) return PF_Err_NONE;  // Custom keeps the colours
        const PalettePreset& preset = Preset(palette - 1);
        for (int k = 0; k < kStopCount; ++k) {
            PF_ParamDef* color = params[kParamColor1 + k];
            color->u.cd.value.red = preset.srgb[k][0];
            color->u.cd.value.green = preset.srgb[k][1];
            color->u.cd.value.blue = preset.srgb[k][2];
            color->u.cd.value.alpha = 255;
            color->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
        }
    } else if (index >= kParamColor1 && index <= kParamColor5) {
        // The colours no longer match any preset, so the popup should not
        // claim they do.
        PF_ParamDef* popup = params[kParamPalette];
        if (popup->u.pd.value != CustomPaletteIndex()) {
            popup->u.pd.value = CustomPaletteIndex();
            popup->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
        }
    }
    return PF_Err_NONE;
}

PF_Err QueryDynamicFlags(PF_InData* in_data, PF_OutData* out_data) {
    bool animate = false;
    float grain = 0.0f;
    PF_Err err = ReadCheckbox(in_data, kParamAnimateGrain, &animate);
    if (!err) err = ReadFloat(in_data, kParamGrain, &grain);
    if (err) return err;
    if (animate && grain > 0.0f) {
        out_data->out_flags |= PF_OutFlag_NON_PARAM_VARY;
    } else {
        out_data->out_flags &= ~static_cast<PF_OutFlags>(PF_OutFlag_NON_PARAM_VARY);
    }
    return PF_Err_NONE;
}

}  // namespace cosmic
