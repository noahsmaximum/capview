# Build the Windows release zip with RTX Video Super Resolution included.
# CI can't do this: the RTX Video SDK is under NVIDIA's license and not in the repo.
# Needs the SDK in capview-deps/rtx-video-sdk (or NV_RTX_VIDEO_SDK). Run from the repo root:
#   powershell -File packaging/windows/package.ps1
$ErrorActionPreference = "Stop"
$v = (Select-String -Path Cargo.toml -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value

cargo build --release --features rtx-video
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

$pkg = "dist/pkg-windows"
Remove-Item -Recurse -Force $pkg -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$pkg/licenses" | Out-Null
Copy-Item target/release/capview.exe, target/release/nvngx_vsr.dll, README.md, LICENSE $pkg/
Copy-Item assets/fonts/OFL-*.txt, licenses/NVIDIA-*.txt "$pkg/licenses/"
$zip = "dist/capview-v$v-windows-x64.zip"
Compress-Archive -Path "$pkg/*" -DestinationPath $zip -Force
Write-Host "wrote $zip"
