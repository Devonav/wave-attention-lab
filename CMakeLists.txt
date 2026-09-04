cmake_minimum_required(VERSION 3.25)

project(wave_attention_lab VERSION 0.1.0 LANGUAGES CXX)

option(WAVE_ENABLE_HIP "Build the HIP backend" OFF)

add_library(wave_attention
    src/attention_cpu.cpp
)
add_library(wave::attention ALIAS wave_attention)

target_include_directories(wave_attention PUBLIC include)
target_compile_features(wave_attention PUBLIC cxx_std_17)

if(WAVE_ENABLE_HIP)
    enable_language(HIP)
    target_sources(wave_attention PRIVATE src/attention_hip.hip)
    target_compile_definitions(wave_attention PUBLIC WAVE_HAS_HIP=1)
    set_target_properties(wave_attention PROPERTIES
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED YES
    )
endif()

if(MSVC)
    target_compile_options(wave_attention PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(wave_attention PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
endif()

add_executable(wave_bench src/main.cpp)
target_link_libraries(wave_bench PRIVATE wave::attention)

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE WAVE_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT WAVE_GIT_SHA)
    set(WAVE_GIT_SHA unknown)
endif()
target_compile_definitions(wave_bench PRIVATE WAVE_GIT_SHA="${WAVE_GIT_SHA}")

include(CTest)
if(BUILD_TESTING)
    add_executable(wave_tests tests/attention_tests.cpp)
    target_link_libraries(wave_tests PRIVATE wave::attention)
    add_test(NAME attention_cpu_correctness COMMAND wave_tests)
endif()
