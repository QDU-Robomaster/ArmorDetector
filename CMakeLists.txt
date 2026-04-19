# CMakeLists.txt for ArmorDetector

target_include_directories(xr PUBLIC ${CMAKE_CURRENT_LIST_DIR})

file(GLOB MODULE_ArmorDetector_SRC CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/*.cxx"
  "${CMAKE_CURRENT_LIST_DIR}/*.cc"
  "${CMAKE_CURRENT_LIST_DIR}/*.c"
)

target_sources(xr PRIVATE ${MODULE_ArmorDetector_SRC})

get_filename_component(_MODULE_DIRNAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(_DEPS_TARGET "${_MODULE_DIRNAME}_deps")
add_library(${_DEPS_TARGET} INTERFACE)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

remove_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_definitions(xr PUBLIC _GLIBCXX_USE_CXX11_ABI=1 OV_THREAD=OV_THREAD_TBB)

get_target_property(OPENVINO_RUNTIME_INCLUDE_DIRS openvino::runtime INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(OPENVINO_RUNTIME_LIBRARY openvino::runtime IMPORTED_LOCATION_RELEASE)
if(NOT OPENVINO_RUNTIME_LIBRARY)
  get_target_property(OPENVINO_RUNTIME_LIBRARY openvino::runtime IMPORTED_LOCATION)
endif()
if(NOT OPENVINO_RUNTIME_LIBRARY)
  message(FATAL_ERROR "Failed to resolve openvino::runtime library path")
endif()

target_include_directories(${_DEPS_TARGET} INTERFACE
  ${OPENVINO_RUNTIME_INCLUDE_DIRS}
)

target_link_libraries(${_DEPS_TARGET} INTERFACE
  ${OpenCV_LIBS}
  ${OPENVINO_RUNTIME_LIBRARY}
)

target_compile_definitions(${_DEPS_TARGET} INTERFACE
  OV_THREAD=OV_THREAD_TBB
  DETECTOR_MODEL_PATH="${CMAKE_CURRENT_LIST_DIR}/model/mlp.onnx"
  YOLOV5_MODEL_PATH="${CMAKE_CURRENT_LIST_DIR}/model/yolov5.xml"
)

set_property(GLOBAL APPEND PROPERTY XR_MODULE_DEPS ${_DEPS_TARGET})
