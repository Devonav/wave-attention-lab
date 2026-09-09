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

# Regenerated on every build so a published CSV never claims a stale commit.
set(WAVE_GIT_SHA_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/wave/git_sha.h")
add_custom_target(wave_git_sha ALL
    COMMAND ${CMAKE_COMMAND}
        -DWAVE_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DWAVE_INPUT=${CMAKE_SOURCE_DIR}/cmake/git_sha.h.in
        -DWAVE_OUTPUT=${WAVE_GIT_SHA_HEADER}
        -P ${CMAKE_SOURCE_DIR}/cmake/WaveGitSha.cmake
    BYPRODUCTS "${WAVE_GIT_SHA_HEADER}"
    COMMENT "Resolving git revision"
    VERBATIM
)
add_dependencies(wave_bench wave_git_sha)
target_include_directories(wave_bench PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_definitions(wave_bench PRIVATE WAVE_HAS_GIT_SHA_HEADER=1)

include(CTest)
if(BUILD_TESTING)
    add_executable(wave_tests tests/attention_tests.cpp)
    target_link_libraries(wave_tests PRIVATE wave::attention)
    add_test(NAME attention_cpu_correctness COMMAND wave_tests)
endif()
