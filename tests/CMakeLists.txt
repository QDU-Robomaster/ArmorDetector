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
