add_executable(armor_detector_publish_geometry_test publish_geometry_test.cpp)

target_link_libraries(armor_detector_publish_geometry_test
                      PRIVATE xr ${_DEPS_TARGET})
target_compile_features(armor_detector_publish_geometry_test PRIVATE cxx_std_20)

add_test(NAME armor_detector_publish_geometry_test
         COMMAND armor_detector_publish_geometry_test)

add_executable(armor_detector_pipeline_test pipeline_test.cpp)

find_package(Threads REQUIRED)
target_include_directories(armor_detector_pipeline_test
                           PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(armor_detector_pipeline_test PRIVATE Threads::Threads)
target_compile_features(armor_detector_pipeline_test PRIVATE cxx_std_20)

add_test(NAME armor_detector_pipeline_test COMMAND armor_detector_pipeline_test)

add_executable(armor_detector_input_view_test input_view_test.cpp)

target_include_directories(armor_detector_input_view_test
                           PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..
                                   ${OpenCV_INCLUDE_DIRS})
target_link_libraries(armor_detector_input_view_test PRIVATE ${OpenCV_LIBS})
target_compile_features(armor_detector_input_view_test PRIVATE cxx_std_20)

add_test(NAME armor_detector_input_view_test COMMAND armor_detector_input_view_test)

add_executable(armor_detector_backend_test backend_test.cpp)
target_link_libraries(armor_detector_backend_test PRIVATE xr ${_DEPS_TARGET} Threads::Threads)
target_compile_features(armor_detector_backend_test PRIVATE cxx_std_20)
add_test(NAME armor_detector_backend_test COMMAND armor_detector_backend_test)
set_tests_properties(armor_detector_backend_test PROPERTIES
                     ENVIRONMENT "XR_ARMOR_OPENVINO_DEVICE=CPU" TIMEOUT 180)
if(OpenVINO_FOUND)
  add_test(NAME armor_detector_unavailable_device_test
           COMMAND armor_detector_backend_test --unavailable-device)
  set_tests_properties(armor_detector_unavailable_device_test PROPERTIES
                       ENVIRONMENT "XR_ARMOR_OPENVINO_DEVICE=NOT_A_DEVICE" TIMEOUT 60)
endif()
