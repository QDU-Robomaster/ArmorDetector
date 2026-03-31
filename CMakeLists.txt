# CMakeLists.txt for ArmorDetector

# Add module to include path
target_include_directories(xr PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Auto-include source files
file(GLOB MODULE_ArmorDetector_SRC CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/*.cxx"
  "${CMAKE_CURRENT_LIST_DIR}/*.cc"
  "${CMAKE_CURRENT_LIST_DIR}/*.c"
)

target_sources(xr PRIVATE ${MODULE_ArmorDetector_SRC})

# INTERFACE deps shell
get_filename_component(_MODULE_DIRNAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(_DEPS_TARGET "${_MODULE_DIRNAME}_deps")
add_library(${_DEPS_TARGET} INTERFACE)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

target_link_libraries(${_DEPS_TARGET} INTERFACE
  ${OpenCV_LIBS}
  openvino::runtime
)

target_compile_definitions(${_DEPS_TARGET} INTERFACE
  DETECTOR_MODEL_PATH="${CMAKE_CURRENT_LIST_DIR}/model/mlp.onnx"
  YOLOV5_MODEL_PATH="${CMAKE_CURRENT_LIST_DIR}/model/yolov5.xml"
)
# target_compile_options(${_DEPS_TARGET} INTERFACE
#   # -Wall
# )

# Register to global
set_property(GLOBAL APPEND PROPERTY XR_MODULE_DEPS ${_DEPS_TARGET})
