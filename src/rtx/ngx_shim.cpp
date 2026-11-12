// Thin C ABI over the NGX (RTX Video SDK) calls MaxCap needs for Video Super
// Resolution. Compiled only with `--features rtx-video`, against the SDK headers on
// the build machine; no NVIDIA source is copied here. NGX itself is not thread-safe:
// the Rust side only calls this from the render thread.

#include <d3d12.h>
#include <nvsdk_ngx_helpers_vsr.h>

static NVSDK_NGX_Parameter* g_params = nullptr;

extern "C" {

// Returns an NVSDK_NGX_Result (1 = success).
int mc_ngx_init(ID3D12Device* device, const wchar_t* data_path) {
    NVSDK_NGX_Result r = NVSDK_NGX_D3D12_Init(0, data_path, device);
    if (NVSDK_NGX_FAILED(r)) return (int)r;
    r = NVSDK_NGX_D3D12_GetCapabilityParameters(&g_params);
    return (int)r;
}

// available / driver-update hint / minimum driver (major.minor) when it's too old.
int mc_ngx_vsr_caps(int* available, int* needs_driver, unsigned* min_major, unsigned* min_minor) {
    if (!g_params) return (int)NVSDK_NGX_Result_Fail;
    *available = 0;
    *needs_driver = 0;
    *min_major = 0;
    *min_minor = 0;
    g_params->Get(NVSDK_NGX_Parameter_VSR_Available, available);
    g_params->Get(NVSDK_NGX_Parameter_VSR_NeedsUpdatedDriver, needs_driver);
    g_params->Get(NVSDK_NGX_Parameter_VSR_MinDriverVersionMajor, min_major);
    g_params->Get(NVSDK_NGX_Parameter_VSR_MinDriverVersionMinor, min_minor);
    return (int)NVSDK_NGX_Result_Success;
}

// Records feature creation into `list` (execute it before the first evaluate).
int mc_ngx_vsr_create(ID3D12GraphicsCommandList* list, NVSDK_NGX_Handle** out) {
    if (!g_params) return (int)NVSDK_NGX_Result_Fail;
    NVSDK_NGX_Feature_Create_Params create = {};
    return (int)NGX_D3D12_CREATE_VSR_EXT(list, 1, 1, out, g_params, &create);
}

// Records one upscale of in[in_rect] into out[out_rect]. Both resources must be in
// D3D12_RESOURCE_STATE_COMMON; `out` needs ALLOW_UNORDERED_ACCESS.
int mc_ngx_vsr_eval(ID3D12GraphicsCommandList* list, NVSDK_NGX_Handle* handle,
                    ID3D12Resource* in, const unsigned in_rect[4],
                    ID3D12Resource* out, const unsigned out_rect[4], int quality) {
    if (!g_params || !handle) return (int)NVSDK_NGX_Result_Fail;
    NVSDK_NGX_D3D12_VSR_Eval_Params p = {};
    p.pInput = in;
    p.pOutput = out;
    p.InputSubrectBase.X = in_rect[0];
    p.InputSubrectBase.Y = in_rect[1];
    p.InputSubrectSize.Width = in_rect[2];
    p.InputSubrectSize.Height = in_rect[3];
    p.OutputSubrectBase.X = out_rect[0];
    p.OutputSubrectBase.Y = out_rect[1];
    p.OutputSubrectSize.Width = out_rect[2];
    p.OutputSubrectSize.Height = out_rect[3];
    p.QualityLevel = (NVSDK_NGX_VSR_QualityLevel)quality;
    return (int)NGX_D3D12_EVALUATE_VSR_EXT(list, handle, g_params, &p);
}

void mc_ngx_vsr_release(NVSDK_NGX_Handle* handle) {
    if (handle) NVSDK_NGX_D3D12_ReleaseFeature(handle);
}

void mc_ngx_shutdown(ID3D12Device* device) {
    NVSDK_NGX_D3D12_Shutdown1(device);
    if (g_params) NVSDK_NGX_D3D12_DestroyParameters(g_params);
    g_params = nullptr;
}

}  // extern "C"
