# CMakeLists.txt for ArmorDetector

target_include_directories(xr PUBLIC ${CMAKE_CURRENT_LIST_DIR})

get_filename_component(_MODULE_DIRNAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(_DEPS_TARGET "${_MODULE_DIRNAME}_deps")
add_library(${_DEPS_TARGET} INTERFACE)

find_package(OpenCV REQUIRED)

set(ARMOR_DETECTOR_BACKEND "AUTO" CACHE STRING
  "ArmorDetector inference backend: AUTO, ORT_CUDA, or OPENVINO")
set_property(CACHE ARMOR_DETECTOR_BACKEND PROPERTY STRINGS
  AUTO ORT_CUDA OPENVINO)
set(ARMOR_DETECTOR_TARGET_PROCESSOR "" CACHE STRING
  "Override target processor for backend selection; empty uses CMAKE_SYSTEM_PROCESSOR")

remove_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_definitions(xr PUBLIC _GLIBCXX_USE_CXX11_ABI=1)

if(ARMOR_DETECTOR_TARGET_PROCESSOR)
  set(_ARMOR_DETECTOR_PROCESSOR "${ARMOR_DETECTOR_TARGET_PROCESSOR}")
else()
  set(_ARMOR_DETECTOR_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
endif()
string(TOLOWER "${_ARMOR_DETECTOR_PROCESSOR}" _ARMOR_DETECTOR_PROCESSOR_LOWER)
set(ARMOR_DETECTOR_OPENVINO_ALLOWED TRUE)
if(_ARMOR_DETECTOR_PROCESSOR_LOWER MATCHES "^(aarch64|arm64)$")
  set(ARMOR_DETECTOR_OPENVINO_ALLOWED FALSE)
endif()

set(_ORT_HINTS
  "$ENV{ONNXRUNTIME_ROOT}"
  "$ENV{ONNXRUNTIME_DIR}"
  "$ENV{ORT_ROOT}"
  /usr/local
  /usr
  /opt/onnxruntime
)

find_path(ARMOR_DETECTOR_ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h onnxruntime/core/session/onnxruntime_cxx_api.h
  HINTS ${_ORT_HINTS}
  PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
)
find_library(ARMOR_DETECTOR_ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS ${_ORT_HINTS}
  PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu lib/x86_64-linux-gnu
)
find_library(ARMOR_DETECTOR_ONNXRUNTIME_CUDA_PROVIDER_LIBRARY
  NAMES onnxruntime_providers_cuda
  HINTS ${_ORT_HINTS}
  PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu lib/x86_64-linux-gnu
)
find_library(ARMOR_DETECTOR_ONNXRUNTIME_PROVIDERS_SHARED_LIBRARY
  NAMES onnxruntime_providers_shared
  HINTS ${_ORT_HINTS}
  PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu lib/x86_64-linux-gnu
)

set(ARMOR_DETECTOR_ORT_CUDA_FOUND FALSE)
if(ARMOR_DETECTOR_ONNXRUNTIME_INCLUDE_DIR
    AND ARMOR_DETECTOR_ONNXRUNTIME_LIBRARY
    AND ARMOR_DETECTOR_ONNXRUNTIME_CUDA_PROVIDER_LIBRARY)
  set(ARMOR_DETECTOR_ORT_CUDA_FOUND TRUE)
endif()

set(ARMOR_DETECTOR_SELECTED_BACKEND "")
if(ARMOR_DETECTOR_BACKEND STREQUAL "AUTO")
  if(ARMOR_DETECTOR_ORT_CUDA_FOUND)
    set(ARMOR_DETECTOR_SELECTED_BACKEND "ORT_CUDA")
  elseif(NOT ARMOR_DETECTOR_OPENVINO_ALLOWED)
    message(FATAL_ERROR
      "ArmorDetector AUTO cannot fall back to OpenVINO on ARM64: the current "
      "model contains opset11 Interpolate layers that are not supported by the "
      "OpenVINO ARM CPU plugin. Install ONNX Runtime CUDA and set "
      "ONNXRUNTIME_ROOT, or build on a supported OpenVINO target.")
  else()
    find_package(OpenVINO QUIET COMPONENTS Runtime)
    if(TARGET openvino::runtime)
      set(ARMOR_DETECTOR_SELECTED_BACKEND "OPENVINO")
    endif()
  endif()
elseif(ARMOR_DETECTOR_BACKEND STREQUAL "ORT_CUDA")
  if(NOT ARMOR_DETECTOR_ORT_CUDA_FOUND)
    message(FATAL_ERROR
      "ARMOR_DETECTOR_BACKEND=ORT_CUDA requires ONNX Runtime headers, "
      "libonnxruntime, and libonnxruntime_providers_cuda. Set ONNXRUNTIME_ROOT "
      "or use ARMOR_DETECTOR_BACKEND=AUTO.")
  endif()
  set(ARMOR_DETECTOR_SELECTED_BACKEND "ORT_CUDA")
elseif(ARMOR_DETECTOR_BACKEND STREQUAL "OPENVINO")
  if(NOT ARMOR_DETECTOR_OPENVINO_ALLOWED)
    message(FATAL_ERROR
      "ARMOR_DETECTOR_BACKEND=OPENVINO is disabled on ARM64 for this model: "
      "opset11 Interpolate layers are not supported by the OpenVINO ARM CPU "
      "plugin. Use ARMOR_DETECTOR_BACKEND=ORT_CUDA with ONNX Runtime CUDA.")
  endif()
  find_package(OpenVINO REQUIRED COMPONENTS Runtime)
  set(ARMOR_DETECTOR_SELECTED_BACKEND "OPENVINO")
else()
  message(FATAL_ERROR
    "Unsupported ARMOR_DETECTOR_BACKEND=${ARMOR_DETECTOR_BACKEND}; "
    "expected AUTO, ORT_CUDA, or OPENVINO.")
endif()

if(ARMOR_DETECTOR_SELECTED_BACKEND STREQUAL "")
  message(FATAL_ERROR
    "ArmorDetector could not find a usable backend. Install ONNX Runtime CUDA "
    "or OpenVINO, or set ARMOR_DETECTOR_BACKEND explicitly.")
endif()

message(STATUS "ArmorDetector selected backend: ${ARMOR_DETECTOR_SELECTED_BACKEND}")

if(ARMOR_DETECTOR_SELECTED_BACKEND STREQUAL "ORT_CUDA")
  set(ARMOR_DETECTOR_MODEL_FILE
    "${CMAKE_CURRENT_LIST_DIR}/model/armor_keypoint_640x512_bgr.onnx"
  )
  if(NOT EXISTS "${ARMOR_DETECTOR_MODEL_FILE}")
    message(FATAL_ERROR "ArmorDetector model file not found: ${ARMOR_DETECTOR_MODEL_FILE}")
  endif()

  target_include_directories(${_DEPS_TARGET} INTERFACE
    ${ARMOR_DETECTOR_ONNXRUNTIME_INCLUDE_DIR}
  )
  target_link_libraries(${_DEPS_TARGET} INTERFACE
    ${OpenCV_LIBS}
    ${ARMOR_DETECTOR_ONNXRUNTIME_LIBRARY}
    ${ARMOR_DETECTOR_ONNXRUNTIME_CUDA_PROVIDER_LIBRARY}
  )
  if(ARMOR_DETECTOR_ONNXRUNTIME_PROVIDERS_SHARED_LIBRARY)
    target_link_libraries(${_DEPS_TARGET} INTERFACE
      ${ARMOR_DETECTOR_ONNXRUNTIME_PROVIDERS_SHARED_LIBRARY}
    )
  endif()
  target_compile_definitions(${_DEPS_TARGET} INTERFACE
    ARMOR_DETECTOR_USE_ORT_CUDA=1
  )
else()
  find_package(OpenVINO REQUIRED COMPONENTS Runtime)

  get_target_property(OPENVINO_RUNTIME_INCLUDE_DIRS openvino::runtime INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(OPENVINO_RUNTIME_LIBRARY openvino::runtime IMPORTED_LOCATION_RELEASE)
  if(NOT OPENVINO_RUNTIME_LIBRARY)
    get_target_property(OPENVINO_RUNTIME_LIBRARY openvino::runtime IMPORTED_LOCATION)
  endif()
  if(NOT OPENVINO_RUNTIME_LIBRARY)
    message(FATAL_ERROR "Failed to resolve openvino::runtime library path")
  endif()

  target_compile_definitions(xr PUBLIC OV_THREAD=OV_THREAD_TBB)
  target_include_directories(${_DEPS_TARGET} INTERFACE
    ${OPENVINO_RUNTIME_INCLUDE_DIRS}
  )
  target_link_libraries(${_DEPS_TARGET} INTERFACE
    ${OpenCV_LIBS}
    ${OPENVINO_RUNTIME_LIBRARY}
  )

  set(ARMOR_DETECTOR_MODEL_FILE
    "${CMAKE_CURRENT_LIST_DIR}/model/armor_keypoint_640x512_bgr.xml"
  )
  if(NOT EXISTS "${ARMOR_DETECTOR_MODEL_FILE}")
    message(FATAL_ERROR "ArmorDetector model file not found: ${ARMOR_DETECTOR_MODEL_FILE}")
  endif()
endif()

target_compile_definitions(${_DEPS_TARGET} INTERFACE
  ARMOR_DETECTOR_BACKEND_NAME="${ARMOR_DETECTOR_SELECTED_BACKEND}"
  ARMOR_DETECTOR_MODEL_PATH="${ARMOR_DETECTOR_MODEL_FILE}"
)

set_property(GLOBAL APPEND PROPERTY XR_MODULE_DEPS ${_DEPS_TARGET})
