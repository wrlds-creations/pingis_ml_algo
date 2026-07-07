cmake_minimum_required(VERSION 3.22.1)

project(collector_edge_impulse_pingpong)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/edge_impulse_local/pingpong_cpp_mcu_v4")
set(EI_MODEL_METADATA "${EI_ROOT}/model-parameters/model_metadata.h")

add_library(edge_impulse_pingpong SHARED edge_impulse_pingpong_jni.cpp)

target_include_directories(edge_impulse_pingpong PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(edge_impulse_pingpong PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-missing-field-initializers
)

if(EXISTS "${EI_MODEL_METADATA}")
    message(STATUS "Building with local Edge Impulse pingpong SDK at ${EI_ROOT}")
    target_compile_definitions(edge_impulse_pingpong PRIVATE
        HAVE_EDGE_IMPULSE_PINGPONG=1
        EI_PORTING_ANDROID=1
        EI_CLASSIFIER_PRINT_STATE=0
    )
    target_include_directories(edge_impulse_pingpong PRIVATE "${EI_ROOT}")

    file(GLOB EI_MODEL_SOURCE CONFIGURE_DEPENDS
        "${EI_ROOT}/tflite-model/*.cpp"
    )
    file(GLOB_RECURSE EI_CLASSIFIER_SOURCE CONFIGURE_DEPENDS
        "${EI_ROOT}/edge-impulse-sdk/classifier/*.cpp"
    )
    file(GLOB_RECURSE EI_DSP_CPP_SOURCE CONFIGURE_DEPENDS
        "${EI_ROOT}/edge-impulse-sdk/dsp/*.cpp"
    )
    file(GLOB_RECURSE EI_TFLITE_SOURCE CONFIGURE_DEPENDS
        "${EI_ROOT}/edge-impulse-sdk/tensorflow/lite/*.cc"
    )
    file(GLOB EI_PORTING_SOURCE CONFIGURE_DEPENDS
        "${EI_ROOT}/edge-impulse-sdk/porting/android/*.cpp"
    )
    file(GLOB_RECURSE EI_CMSIS_DSP_SOURCE CONFIGURE_DEPENDS
        "${EI_ROOT}/edge-impulse-sdk/CMSIS/DSP/Source/TransformFunctions/*.c"
        "${EI_ROOT}/edge-impulse-sdk/CMSIS/DSP/Source/CommonTables/*.c"
        "${EI_ROOT}/edge-impulse-sdk/CMSIS/DSP/Source/BasicMathFunctions/*.c"
        "${EI_ROOT}/edge-impulse-sdk/CMSIS/DSP/Source/ComplexMathFunctions/*.c"
        "${EI_ROOT}/edge-impulse-sdk/CMSIS/DSP/Source/FastMathFunctions/*.c"
        "${EI_ROOT}/edge-impulse-sdk/CMSIS/DSP/Source/SupportFunctions/*.c"
        "${EI_ROOT}/edge-impulse-sdk/CMSIS/DSP/Source/MatrixFunctions/*.c"
        "${EI_ROOT}/edge-impulse-sdk/CMSIS/DSP/Source/StatisticsFunctions/*.c"
    )
    set(EI_EXTRA_C_SOURCE
        "${EI_ROOT}/edge-impulse-sdk/tensorflow/lite/c/common.c"
    )

    target_sources(edge_impulse_pingpong PRIVATE
        ${EI_MODEL_SOURCE}
        ${EI_CLASSIFIER_SOURCE}
        ${EI_DSP_CPP_SOURCE}
        ${EI_TFLITE_SOURCE}
        ${EI_PORTING_SOURCE}
        ${EI_CMSIS_DSP_SOURCE}
        ${EI_EXTRA_C_SOURCE}
    )
else()
    message(STATUS "Local Edge Impulse pingpong SDK missing; building JNI stub only.")
endif()

find_library(log-lib log)
target_link_libraries(edge_impulse_pingpong ${log-lib})
