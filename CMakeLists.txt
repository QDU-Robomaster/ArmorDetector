# ArmorDetector 模块 CMake 配置。推理后端按找到的 SDK 编入：HailoRT（.hef）、OpenVINO（.onnx）。
# Inference backends are built in for the SDKs found: HailoRT (.hef), OpenVINO (.onnx).

target_include_directories(xr PUBLIC ${CMAKE_CURRENT_LIST_DIR})

get_filename_component(_MODULE_DIRNAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(_DEPS_TARGET "${_MODULE_DIRNAME}_deps")
add_library(${_DEPS_TARGET} INTERFACE)

find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)
find_package(HailoRT QUIET CONFIG)
find_package(OpenVINO QUIET COMPONENTS Runtime)

target_link_libraries(${_DEPS_TARGET} INTERFACE ${OpenCV_LIBS})
target_include_directories(${_DEPS_TARGET} INTERFACE ${OpenCV_INCLUDE_DIRS})

if(HailoRT_FOUND)
  target_link_libraries(${_DEPS_TARGET} INTERFACE HailoRT::libhailort)
  target_compile_definitions(${_DEPS_TARGET} INTERFACE ARMOR_DETECTOR_HAVE_HAILORT=1)
endif()
if(OpenVINO_FOUND)
  target_link_libraries(${_DEPS_TARGET} INTERFACE openvino::runtime)
  target_compile_definitions(${_DEPS_TARGET} INTERFACE ARMOR_DETECTOR_HAVE_OPENVINO=1)
endif()
message(STATUS "ArmorDetector backends: HailoRT=${HailoRT_FOUND} OpenVINO=${OpenVINO_FOUND}")

set_property(GLOBAL APPEND PROPERTY XR_MODULE_DEPS ${_DEPS_TARGET})

if(BUILD_TESTING)
  include(CTest)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests
                   ${CMAKE_CURRENT_BINARY_DIR}/ArmorDetector_tests)
endif()
