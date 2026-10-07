find_package(Threads REQUIRED)

add_executable(armor_detector_decoder_test decoder_test.cpp)
set(_ARMOR_TESTS armor_detector_decoder_test)
if(OpenVINO_FOUND)
  # 依赖模型文件，未设置环境变量时跳过 / Needs the model files; skipped without them.
  add_executable(armor_detector_model_test model_test.cpp)
  list(APPEND _ARMOR_TESTS armor_detector_model_test)
endif()

foreach(test ${_ARMOR_TESTS})
  target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)
  target_link_libraries(${test} PRIVATE xr ArmorDetector_deps Threads::Threads)
  target_compile_features(${test} PRIVATE cxx_std_20)
  add_test(NAME ${test} COMMAND ${test})
endforeach()
if(OpenVINO_FOUND)
  set_tests_properties(armor_detector_model_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()
