#!/usr/bin/env bash
# Build capview.app (universal: Apple Silicon + Intel) and zip it.
# Run on a Mac from the repo root:  packaging/macos/bundle.sh
# Output: dist/capview-v<version>-macos-universal.zip
set -euo pipefail

cd "$(dirname "$0")/../.."
VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
TARGETS=(aarch64-apple-darwin x86_64-apple-darwin)

for t in "${TARGETS[@]}"; do
    rustup target add "$t" >/dev/null
    MACOSX_DEPLOYMENT_TARGET=11.0 cargo build --release --target "$t"
done

APP=dist/capview.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
lipo -create -output "$APP/Contents/MacOS/capview" \
    target/aarch64-apple-darwin/release/capview \
    target/x86_64-apple-darwin/release/capview
sed "s/@VERSION@/$VERSION/g" packaging/macos/Info.plist > "$APP/Contents/Info.plist"
cp LICENSE README.md "$APP/Contents/Resources/"
mkdir -p "$APP/Contents/Resources/licenses"
cp assets/fonts/OFL-*.txt "$APP/Contents/Resources/licenses/"

# Ad-hoc signature: required to run on Apple Silicon, and gives the camera /
# microphone permission a stable identity. (Not notarized — see README.)
codesign --force --deep --sign - "$APP"

ZIP="dist/capview-v$VERSION-macos-universal.zip"
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"
echo "built $ZIP"
