add_executable(armor_detector_publish_geometry_test publish_geometry_test.cpp)

target_link_libraries(armor_detector_publish_geometry_test
                      PRIVATE xr ${_DEPS_TARGET})
target_compile_features(armor_detector_publish_geometry_test PRIVATE cxx_std_20)

add_test(NAME armor_detector_publish_geometry_test
         COMMAND armor_detector_publish_geometry_test)
