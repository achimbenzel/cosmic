#pragma once

#include "SdkIncludes.h"

namespace cosmic {

PF_Err SmartPreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra);
PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra);

// CUDA rendering: device setup loads the kernels into After Effects' context,
// or declines the device so it renders on the CPU instead.
PF_Err GpuDeviceSetup(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetupExtra* extra);
PF_Err GpuDeviceSetdown(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetdownExtra* extra);
PF_Err SmartRenderGpu(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra);

// Fallback path for hosts that do not drive SmartFX.
PF_Err LegacyFrameSetup(PF_InData* in_data, PF_OutData* out_data);
PF_Err LegacyRender(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output);

}  // namespace cosmic
