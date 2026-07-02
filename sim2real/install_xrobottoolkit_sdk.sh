#!/usr/bin/env bash
set -euo pipefail

SIM2REAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TELEOP_DIR="$SIM2REAL_ROOT/teleop"
DEPS_DIR="$TELEOP_DIR/deps"
PYBIND_DIR="$DEPS_DIR/XRoboToolkit-PC-Service-Pybind"
SDK_DIR="$DEPS_DIR/XRoboToolkit-PC-Service"
PYTHON_BIN="$SIM2REAL_ROOT/.venv/bin/python"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found in PATH."
    exit 1
fi

echo "[install_xrobottoolkit_sdk] syncing uv environment"
(cd "$SIM2REAL_ROOT" && uv sync)

mkdir -p "$DEPS_DIR"

if [[ ! -d "$PYBIND_DIR/.git" ]]; then
    echo "[install_xrobottoolkit_sdk] cloning XRoboToolkit-PC-Service-Pybind"
    git clone https://github.com/Axellwppr/XRoboToolkit-PC-Service-Pybind "$PYBIND_DIR"
else
    echo "[install_xrobottoolkit_sdk] using existing $PYBIND_DIR"
fi

if [[ ! -d "$SDK_DIR/.git" ]]; then
    echo "[install_xrobottoolkit_sdk] cloning XRoboToolkit-PC-Service"
    git clone --branch main --single-branch https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git "$SDK_DIR"
else
    echo "[install_xrobottoolkit_sdk] using existing $SDK_DIR"
fi

ensure_python_build_tools() {
    if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import setuptools
import wheel
PY
    then
        return 0
    fi

    echo "[install_xrobottoolkit_sdk] installing Python build tools with uv"
    if uv pip install --python "$PYTHON_BIN" setuptools wheel; then
        if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import setuptools
import wheel
PY
        then
            return 0
        fi
    fi

    echo "[install_xrobottoolkit_sdk] uv install failed; falling back to Python ensurepip"
    if "$PYTHON_BIN" -m ensurepip --upgrade; then
        if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import setuptools
import wheel
PY
        then
            return 0
        fi
    fi

    echo "[install_xrobottoolkit_sdk] ensurepip did not provide build tools; retrying uv pip"
    uv pip install --python "$PYTHON_BIN" setuptools wheel

    if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import setuptools
import wheel
PY
    then
        return 0
    fi

    echo "[install_xrobottoolkit_sdk] failed to install setuptools/wheel into $PYTHON_BIN" >&2
    exit 1
}

check_native_build_deps() {
    local missing=()
    for tool in cmake g++ make pkg-config protoc grpc_cpp_plugin; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing+=("$tool")
        fi
    done
    if ! pkg-config --exists protobuf; then
        missing+=("protobuf.pc")
    fi
    if ! pkg-config --exists grpc++; then
        missing+=("grpc++.pc")
    fi

    if (( ${#missing[@]} > 0 )); then
        cat >&2 <<EOF
[install_xrobottoolkit_sdk] missing native build dependencies:
  ${missing[*]}

Install them on Ubuntu with:
  sudo apt-get update
  sudo apt-get install -y build-essential cmake pkg-config \\
      libprotobuf-dev protobuf-compiler \\
      libgrpc++-dev libgrpc-dev protobuf-compiler-grpc

If this robot needs to use the deploy machine proxy, first open the SSH reverse
proxy from rp-deploy, then run apt with:
  sudo apt-get -o Acquire::http::Proxy=http://127.0.0.1:17890 \\
      -o Acquire::https::Proxy=http://127.0.0.1:17890 update
  sudo apt-get -o Acquire::http::Proxy=http://127.0.0.1:17890 \\
      -o Acquire::https::Proxy=http://127.0.0.1:17890 install -y \\
      build-essential cmake pkg-config \\
      libprotobuf-dev protobuf-compiler \\
      libgrpc++-dev libgrpc-dev protobuf-compiler-grpc
EOF
        exit 1
    fi
}

build_pxrea_sdk_from_source() {
    local machine
    machine=$(uname -m)
    local proto_subdir
    local compile_def
    case "$machine" in
        aarch64|arm64)
            proto_subdir="linux_aarch64"
            compile_def="LINUX_aarch64"
            ;;
        x86_64|amd64)
            proto_subdir="linux_x86"
            compile_def="LINUX_x86"
            ;;
        *)
            echo "[install_xrobottoolkit_sdk] unsupported architecture: $machine" >&2
            exit 1
            ;;
    esac

    local robotics_dir="$SDK_DIR/RoboticsService"
    local sdk_src_dir="$robotics_dir/PXREARobotSDK"
    local proto_dir="$robotics_dir/PXREAService/$proto_subdir"
    local sdk_build_dir="$PYBIND_DIR/build/pxrea_sdk"
    local sdk_build_src_dir="$sdk_build_dir/src"
    local sdk_output_dir="$sdk_build_dir/out"

    if [[ ! -f "$proto_dir/PXREAService.proto" ]]; then
        echo "[install_xrobottoolkit_sdk] missing proto file: $proto_dir/PXREAService.proto" >&2
        exit 1
    fi
    if [[ ! -f "$sdk_src_dir/PXREARobotSDK.cpp" ]]; then
        echo "[install_xrobottoolkit_sdk] missing SDK source: $sdk_src_dir/PXREARobotSDK.cpp" >&2
        exit 1
    fi

    echo "[install_xrobottoolkit_sdk] regenerating protobuf sources with system protoc"
    (
        cd "$proto_dir"
        protoc \
            -I "$proto_dir" \
            -I /usr/include \
            --plugin=protoc-gen-grpc="$(command -v grpc_cpp_plugin)" \
            --grpc_out="$proto_dir" \
            --cpp_out="$proto_dir" \
            PXREAService.proto
    )

    rm -rf "$sdk_build_dir"
    mkdir -p "$sdk_build_src_dir" "$sdk_output_dir"
    cat >"$sdk_build_src_dir/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.14)
project(PXREARobotSDKSourceBuild LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PROTOBUF REQUIRED protobuf)
pkg_check_modules(GRPCPP REQUIRED grpc++)

add_library(PXREARobotSDK SHARED
    "${PXREA_SDK_SRC_DIR}/PXREARobotSDK.cpp"
    "${PXREA_PROTO_DIR}/PXREAService.pb.cc"
    "${PXREA_PROTO_DIR}/PXREAService.grpc.pb.cc"
)

target_compile_definitions(PXREARobotSDK PRIVATE
    PXREACLIENTSDK_LIBRARY
    PXREAROBOTSDK_LIBRARY
    "${PXREA_LINUX_DEFINE}"
)

target_include_directories(PXREARobotSDK PRIVATE
    "${PXREA_SDK_SRC_DIR}"
    "${PXREA_PROTO_DIR}"
    ${PROTOBUF_INCLUDE_DIRS}
    ${GRPCPP_INCLUDE_DIRS}
)

target_compile_options(PXREARobotSDK PRIVATE
    ${PROTOBUF_CFLAGS_OTHER}
    ${GRPCPP_CFLAGS_OTHER}
    -include unistd.h
)

target_link_directories(PXREARobotSDK PRIVATE
    ${PROTOBUF_LIBRARY_DIRS}
    ${GRPCPP_LIBRARY_DIRS}
)

target_link_libraries(PXREARobotSDK PRIVATE
    ${GRPCPP_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
    Threads::Threads
    dl
)

set_target_properties(PXREARobotSDK PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PXREA_OUTPUT_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
EOF

    echo "[install_xrobottoolkit_sdk] building PXREARobotSDK from source"
    cmake \
        -S "$sdk_build_src_dir" \
        -B "$sdk_build_dir/cmake" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPXREA_SDK_SRC_DIR="$sdk_src_dir" \
        -DPXREA_PROTO_DIR="$proto_dir" \
        -DPXREA_LINUX_DEFINE="$compile_def" \
        -DPXREA_OUTPUT_DIR="$sdk_output_dir"
    cmake --build "$sdk_build_dir/cmake" --target PXREARobotSDK -j"$(nproc)"

    SDK_LIB="$sdk_output_dir/libPXREARobotSDK.so"
    if [[ ! -f "$SDK_LIB" ]]; then
        echo "[install_xrobottoolkit_sdk] PXREA SDK build did not produce $SDK_LIB" >&2
        exit 1
    fi
}

check_native_build_deps
ensure_python_build_tools
build_pxrea_sdk_from_source

echo "[install_xrobottoolkit_sdk] installing xrobotoolkit_sdk with $PYTHON_BIN"
(
    cd "$PYBIND_DIR"
    find build -mindepth 1 -maxdepth 1 ! -name pxrea_sdk -exec rm -rf {} + 2>/dev/null || true
    rm -rf ./*.egg-info
    if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
        PXREA_SDK_LIBRARY="$SDK_LIB" "$PYTHON_BIN" -m pip install --no-build-isolation --force-reinstall "$PYBIND_DIR"
    elif command -v uv >/dev/null 2>&1; then
        PXREA_SDK_LIBRARY="$SDK_LIB" uv pip install --python "$PYTHON_BIN" --no-build-isolation --force-reinstall "$PYBIND_DIR"
    else
        echo "Neither '$PYTHON_BIN -m pip' nor 'uv pip' is available." >&2
        exit 1
    fi
)

echo "[install_xrobottoolkit_sdk] verifying install"
(cd "$SIM2REAL_ROOT" && uv run python - <<'PY'
import xrobotoolkit_sdk as xrt

required = (
    "init",
    "register_frame_callback",
    "clear_frame_callback",
    "has_frame_callback",
)
missing = [name for name in required if not hasattr(xrt, name)]
if missing:
    raise RuntimeError(f"xrobotoolkit_sdk missing APIs: {missing}")
print("xrobotoolkit_sdk:", xrt.__file__)
print("callback_api: OK")
PY
)

echo "[install_xrobottoolkit_sdk] done"
