fn main() {
    #[cfg(feature = "rtx-video")]
    rtx_video();
}

/// Compile the NGX shim against the RTX Video SDK and put the VSR runtime next to the
/// exe. The SDK is NVIDIA-licensed and never lives in this repo: point NV_RTX_VIDEO_SDK
/// at it, or unpack it to ./maxcap-deps/rtx-video-sdk (gitignored).
#[cfg(feature = "rtx-video")]
fn rtx_video() {
    use std::path::PathBuf;
    println!("cargo:rerun-if-changed=src/rtx/ngx_shim.cpp");
    println!("cargo:rerun-if-env-changed=NV_RTX_VIDEO_SDK");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return; // VSR is Windows-only; the feature is a no-op elsewhere.
    }
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let sdk = std::env::var_os("NV_RTX_VIDEO_SDK")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("maxcap-deps").join("rtx-video-sdk"));
    if !sdk.join("include").join("nvsdk_ngx_helpers_vsr.h").exists() {
        panic!(
            "rtx-video: RTX Video SDK not found at {} (set NV_RTX_VIDEO_SDK)",
            sdk.display()
        );
    }
    let arch = match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64",
        _ => "x64",
    };

    cc::Build::new()
        .cpp(true)
        .file("src/rtx/ngx_shim.cpp")
        .include(sdk.join("include"))
        .static_crt(true) // matches .cargo/config.toml (+crt-static) and nvsdk_ngx_s.lib
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/EHsc")
        .compile("maxcap_ngx_shim");

    let lib = sdk.join("lib").join("Windows").join(arch);
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=nvsdk_ngx_s");
    for l in ["d3d12", "advapi32", "shell32", "ole32", "version", "user32"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }

    // NGX loads nvngx_vsr.dll from the exe's folder. OUT_DIR is
    // target/<profile>/build/maxcap-<hash>/out, so the exe dir is three levels up.
    let dll = sdk
        .join("bin")
        .join("Windows")
        .join(arch)
        .join("rel")
        .join("nvngx_vsr.dll");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    if let Some(exe_dir) = out.ancestors().nth(3) {
        let dst = exe_dir.join("nvngx_vsr.dll");
        let stale = match (std::fs::metadata(&dll), std::fs::metadata(&dst)) {
            (Ok(a), Ok(b)) => a.len() != b.len(),
            _ => true,
        };
        if stale && let Err(e) = std::fs::copy(&dll, &dst) {
            println!("cargo:warning=rtx-video: couldn't copy nvngx_vsr.dll: {e}");
        }
    }
}
