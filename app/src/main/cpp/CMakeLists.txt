cmake_minimum_required(VERSION 3.22.1)
project("realcugan_ncnn_android")

# 支持的 ABI 列表
set(ANDROID_ABIS armeabi-v7a arm64-v8a riscv64 x86 x86_64)

# 为每个 ABI 预定义所有库的路径变量
foreach(ABI ${ANDROID_ABIS})
    # NCNN
    set(NCNN_LIB_${ABI}     ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libncnn.a)
    # glslang
    set(GLSLANG_LIB_${ABI}   ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libglslang.a)
    set(GCG_LIB_${ABI}       ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libGenericCodeGen.a)
    set(GRL_LIB_${ABI}       ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libglslang-default-resource-limits.a)
    set(MI_LIB_${ABI}        ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libMachineIndependent.a)
    set(OGLC_LIB_${ABI}      ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libOGLCompiler.a)
    set(OSD_LIB_${ABI}       ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libOSDependent.a)
    set(SPIRV_LIB_${ABI}     ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libSPIRV.a)
    # WebP 相关
    set(WEBP_CPUFEATURES_LIB_${ABI} ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libcpufeatures-webp.a)
    set(WEBP_EXAMPLEUTIL_LIB_${ABI} ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libexampleutil.a)
    set(WEBP_EXTRAS_LIB_${ABI}      ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libextras.a)
    set(WEBP_IMAGDEC_LIB_${ABI}     ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libimagedec.a)
    set(WEBP_IMAGEENC_LIB_${ABI}    ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libimageenc.a)
    set(WEBP_IMAGEIUTIL_LIB_${ABI}  ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libimageioutil.a)
    set(WEBP_WEBP_LIB_${ABI}        ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libwebp.a)
    set(WEBP_DECODER_LIB_${ABI}     ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libwebpdecoder.a)
    set(WEBP_DEMUX_LIB_${ABI}       ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libwebpdemux.a)
    set(WEBP_MUX_LIB_${ABI}         ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libwebpmux.a)
    set(WEBP_SHARPYUV_LIB_${ABI}    ${CMAKE_SOURCE_DIR}/lib/${ABI}/lib/libsharpyuv.a)
endforeach()

# 头文件搜索路径
include_directories(
        ${CMAKE_SOURCE_DIR}/include/ncnn
        ${CMAKE_SOURCE_DIR}/include/glslang
        ${CMAKE_SOURCE_DIR}/include/realcugan
        ${CMAKE_SOURCE_DIR}/include/webp
)

# 启用 OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message(STATUS "OpenMP found, enabling it.")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 生成主共享库
add_library(${PROJECT_NAME} SHARED
        native-lib.cpp
        realcugan.cpp
)

# 导入所有静态库为 CMake 目标
macro(import_static name var)
    add_library(${name} STATIC IMPORTED)
    set_target_properties(${name} PROPERTIES IMPORTED_LOCATION ${${var}})
endmacro()

# NCNN 和 glslang 系列
import_static(ncnn             NCNN_LIB_${ANDROID_ABI})
import_static(glslang          GLSLANG_LIB_${ANDROID_ABI})
import_static(GenericCodeGen   GCG_LIB_${ANDROID_ABI})
import_static(glslangResLimits GRL_LIB_${ANDROID_ABI})
import_static(MachineIndependent MI_LIB_${ANDROID_ABI})
import_static(OGLCompiler      OGLC_LIB_${ANDROID_ABI})
import_static(OSDependent      OSD_LIB_${ANDROID_ABI})
import_static(SPIRV            SPIRV_LIB_${ANDROID_ABI})

# WebP 相关
import_static(cpufeatures_webp WEBP_CPUFEATURES_LIB_${ANDROID_ABI})
import_static(exampleutil      WEBP_EXAMPLEUTIL_LIB_${ANDROID_ABI})
import_static(extras           WEBP_EXTRAS_LIB_${ANDROID_ABI})
import_static(imagedec         WEBP_IMAGDEC_LIB_${ANDROID_ABI})
import_static(imageenc         WEBP_IMAGEENC_LIB_${ANDROID_ABI})
import_static(imageioutil      WEBP_IMAGEIUTIL_LIB_${ANDROID_ABI})
import_static(webp             WEBP_WEBP_LIB_${ANDROID_ABI})
import_static(webpdecoder      WEBP_DECODER_LIB_${ANDROID_ABI})
import_static(webpdemux        WEBP_DEMUX_LIB_${ANDROID_ABI})
import_static(webpmux          WEBP_MUX_LIB_${ANDROID_ABI})
import_static(sharpyuv         WEBP_SHARPYUV_LIB_${ANDROID_ABI})

# 最终链接顺序：NCNN/GLSLANG → Android 日志 → WebP 系列 → JNI 图形接口 → Android 平台库
target_link_libraries(${PROJECT_NAME} PRIVATE
        OpenMP::OpenMP_CXX

        ncnn
        glslang
        GenericCodeGen
        glslangResLimits
        MachineIndependent
        OGLCompiler
        OSDependent
        SPIRV

        log

        cpufeatures_webp
        exampleutil
        extras
        imagedec
        imageenc
        imageioutil
        webp
        webpdecoder
        webpdemux
        webpmux
        sharpyuv

        jnigraphics    # AndroidBitmap_* 定义在这里
        android        # 其它 Android NDK API
)