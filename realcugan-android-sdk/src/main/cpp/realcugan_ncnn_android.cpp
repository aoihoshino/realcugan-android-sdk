#include <jni.h>
#include <string>
#include <android/log.h>
#include <android/bitmap.h>
#include "realcugan.h"
#include "filesystem_utils.h"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "webp_image.h"

#define LOG_TAG "RealCUGAN_NCNN_ANDROID_NATIVE"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ncnn初始化锁
static std::mutex gpu_mutex;
static bool gpu_initialized = false;

// realcugan序列锁
static std::mutex g_map_mutex;
static std::unordered_map<jlong, RealCUGAN *> g_instances;

// handler
static std::mutex handler_mutex;
static jlong next_handle = 1;

// 确保ncnn gpu只初始化一次
void ensure_ncnn_gpu() {
    std::lock_guard<std::mutex> lg(gpu_mutex);
    if (!gpu_initialized) {
        ncnn::create_gpu_instance();
        gpu_initialized = true;
    }
}

// 如果空了，就销毁 GPU 并允许下次 re-init
void release_ncnn_gpu() {
    std::lock_guard<std::mutex> lg(gpu_mutex);
    if (g_instances.empty() && gpu_initialized) {
        ncnn::destroy_gpu_instance();
        gpu_initialized = false;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_aoihoshino_realcugan_1ncnn_1android_RealCUGAN_nativeInitialize(
        JNIEnv *env, jclass /* this */,
        jstring modelRootDir,
        jobject noiseObj,      // java.lang.Integer or null
        jobject scaleObj,      // java.lang.Integer or null
        jobject syncgapObj,    // java.lang.Integer or null
        jstring modelNameJ,    // java.lang.String or null
        jobject ttaModeObj,    // java.lang.Boolean or null
        jobject gpuidObj       // java.lang.Integer or null
) {

    // 先声明要抛出的异常类
    jclass runtimeExc = env->FindClass("java/lang/RuntimeException");
    if (!runtimeExc) {
        return -1; // 如果连 RuntimeException 都找不到，直接回
    }

    // —— 1. 解析 Integer 和 Boolean 包装类型 ——
    jclass integerCls = env->FindClass("java/lang/Integer");
    jmethodID intValueID = env->GetMethodID(integerCls, "intValue", "()I");
    jclass booleanCls = env->FindClass("java/lang/Boolean");
    jmethodID boolValueID = env->GetMethodID(booleanCls, "booleanValue", "()Z");

    int noise = noiseObj ? env->CallIntMethod(noiseObj, intValueID) : -1;
    int scale = scaleObj ? env->CallIntMethod(scaleObj, intValueID) : 2;
    int syncgap = syncgapObj ? env->CallIntMethod(syncgapObj, intValueID) : 3;
    bool ttaMode = ttaModeObj != nullptr && env->CallBooleanMethod(ttaModeObj, boolValueID);
    int numThreads = 1;

    // —— 2. 创建 GPU 实例 ——
    int gpuId;
    if (gpuidObj) {
        int gpu_count = ncnn::get_gpu_count();
        gpuId = env->CallIntMethod(gpuidObj, intValueID);
        if (gpuId < -1 || gpuId >= gpu_count) {
            LOGE("invalid gpu device id %d", gpuId);
            return -1;
        }
    } else {
        gpuId = ncnn::get_default_gpu_index();
    }
    if (gpuId > -1) {
        ensure_ncnn_gpu();
    }
    LOGI("initialize(): using GPU %d", gpuId);

    // —— 3. 参数合法性检查 ——
    if (noise < -1 || noise > 3) {
        LOGE("initialize(): invalid noise %d", noise);
        release_ncnn_gpu();
        return -1;
    }
    if (scale < 2 || scale > 4) {
        LOGE("initialize(): invalid scale %d", scale);
        release_ncnn_gpu();
        return -1;
    }
    if (syncgap < 0 || syncgap > 3) {
        LOGE("initialize(): invalid syncgap %d", syncgap);
        release_ncnn_gpu();
        return -1;
    }

    // —— 4. 解析 modelName ——
    std::string modelDir = "models-se";
    if (modelNameJ) {
        const char *tmp = env->GetStringUTFChars(modelNameJ, nullptr);
        if (tmp && *tmp) modelDir = tmp;
        env->ReleaseStringUTFChars(modelNameJ, tmp);
    }
    // main.cpp 里 nose 模型强制关 syncgap
    if (modelDir == "models-nose") {
        syncgap = 0;
    }

    // —— 5. 计算 prepadding ——
    int prepadding = 0;
    if (scale == 2) prepadding = 18;
    else if (scale == 3) prepadding = 14;
    else if (scale == 4) prepadding = 19;

    // —— 6. 计算 tilesize ——
    int tilesize = 0;
    if (gpuId == -1) {
        // cpu only, optimised for mobile chip
        tilesize = 200;
    } else {
        uint32_t heap = ncnn::get_gpu_device(gpuId)->get_heap_budget();
        if (scale == 2) {
            if (heap > 1300) tilesize = 400;
            else if (heap > 800) tilesize = 300;
            else if (heap > 400) tilesize = 200;
            else if (heap > 200) tilesize = 100;
            else tilesize = 32;
        } else if (scale == 3) {
            if (heap > 3300) tilesize = 400;
            else if (heap > 1900) tilesize = 300;
            else if (heap > 950) tilesize = 200;
            else if (heap > 320) tilesize = 100;
            else tilesize = 32;
        } else { // scale == 4
            if (heap > 1690) tilesize = 400;
            else if (heap > 980) tilesize = 300;
            else if (heap > 530) tilesize = 200;
            else if (heap > 240) tilesize = 100;
            else tilesize = 32;
        }
    }

    // —— 7. 构造模型文件路径 ——
    const char *root = env->GetStringUTFChars(modelRootDir, nullptr);
    char parampath[256], modelpath[256];
    if (noise == -1) {
        sprintf(parampath, "%s/%s/up%dx-conservative.param", root, modelDir.c_str(), scale);
        sprintf(modelpath, "%s/%s/up%dx-conservative.bin", root, modelDir.c_str(), scale);
    } else if (noise == 0) {
        sprintf(parampath, "%s/%s/up%dx-no-denoise.param", root, modelDir.c_str(), scale);
        sprintf(modelpath, "%s/%s/up%dx-no-denoise.bin", root, modelDir.c_str(), scale);
    } else {
        sprintf(parampath, "%s/%s/up%dx-denoise%dx.param", root, modelDir.c_str(), scale, noise);
        sprintf(modelpath, "%s/%s/up%dx-denoise%dx.bin", root, modelDir.c_str(), scale, noise);
    }
    path_t paramFull = sanitize_filepath(parampath);
    path_t modelFull = sanitize_filepath(modelpath);

    LOGI("initialize(): noise=%d scale=%d syncgap=%d ttaMode=%d numThreads=%d prepadding=%d tilesize=%d model=%s",
         noise, scale, syncgap, ttaMode, numThreads, prepadding, tilesize, modelDir.c_str());

    LOGI("param file: %s", paramFull.c_str());
    LOGI("model file: %s", modelFull.c_str());
    if (access(paramFull.c_str(), F_OK) != 0 || access(modelFull.c_str(), F_OK) != 0) {
        LOGE("model file not found！");
        return -1;
    } else {
        LOGI("model loaded successfully.");
    }

    env->ReleaseStringUTFChars(modelRootDir, root);

    // —— 8. 创建 RealCUGAN 并 load 模型 ——
    auto *inst = new RealCUGAN(gpuId, ttaMode, numThreads);
    inst->noise = noise;
    inst->scale = scale;
    inst->syncgap = syncgap;
    inst->prepadding = prepadding;
    inst->tilesize = tilesize;

    int ret = inst->load(paramFull, modelFull);
    if (ret != 0) {
        LOGE("initialize: RealCUGAN::load failed (%d)", ret);
        delete inst;
        release_ncnn_gpu();
        return ret;
    }
    jlong handle;
    try {
        std::lock_guard<std::mutex> lg_next_handle(handler_mutex);
        handle = next_handle++;
        {
            std::lock_guard lg_map(g_map_mutex);
            g_instances[handle] = inst;
        }
    }
    catch (const std::exception &e) {
        // C++ 异常
        env->ThrowNew(runtimeExc, e.what());
        return -1;
    }
    catch (...) {
        // 任何其他崩溃
        env->ThrowNew(runtimeExc, "Unknown native error in RealCUGAN");
        return -1;
    }
    LOGI("initialize: RealCUGAN::load succeeded");
    return handle;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_github_aoihoshino_realcugan_1ncnn_1android_RealCUGAN_nativeProcessImage(
        JNIEnv *env, jclass /*clazz*/,
        jlong handle,
        jbyteArray imageData) {
    // —— 1) 找到对应的 RealCUGAN 实例 —————————————
    // 1) Find the right instance under lock
    // 先声明要抛出的异常类
    jclass runtimeExc = env->FindClass("java/lang/RuntimeException");
    if (!runtimeExc) {
        return nullptr; // 如果连 RuntimeException 都找不到，直接回
    }
    RealCUGAN *inst = nullptr;
    {
        std::lock_guard<std::mutex> guard(g_map_mutex);
        auto it = g_instances.find(handle);
        if (it != g_instances.end())
            inst = it->second;
    }
    if (!inst) {
        LOGE("processImage: instance of handle %lld not found", handle);
        return nullptr;
    }
    LOGI("processImage realcugan instance: handle = %lld noise=%d scale=%d syncgap=%d prepadding=%d tilesize=%d",
         handle, inst->noise, inst->scale, inst->syncgap, inst->prepadding, inst->tilesize);

    // 1) 从 Java 拿到压缩后的字节
    jsize length = env->GetArrayLength(imageData);
    jbyte *buffer = env->GetByteArrayElements(imageData, nullptr);

    int w = 0, h = 0, c = 0;
    unsigned char *pixeldata = nullptr;

    // 2) 先尝试 WebP
    pixeldata = webp_load(reinterpret_cast<unsigned char *>(buffer), length, &w, &h, &c);

    // 3) 回退到 PNG/JPEG
    if (!pixeldata) {
        pixeldata = stbi_load_from_memory(
                reinterpret_cast<unsigned char *>(buffer),
                length, &w, &h, &c, 0
        );
        if (!pixeldata) {
            LOGE("processImage: not webp nor png/jpeg");
            env->ReleaseByteArrayElements(imageData, buffer, JNI_ABORT);
            return nullptr;
        }
    }

    // —— 强制把灰度/灰度+Alpha 转成 3/4 通道 ————————————
    if (c == 1 || c == 2) {
        // free 上一次的解码结果
        free(pixeldata);
        // 重新走一次 stb_image，指定通道数
        int want_chan = (c == 1 ? 3 : 4);
        pixeldata = stbi_load_from_memory(
                reinterpret_cast<unsigned char *>(buffer),
                length, &w, &h, &c, want_chan
        );
        c = want_chan; // 更新 c
        if (!pixeldata) {
            LOGE("processImage: re-stbi_load_from_memory failed");
            env->ReleaseByteArrayElements(imageData, buffer, JNI_ABORT);
            return nullptr;
        }
    }

    // 4) 现在可以释放 Java 的 byte[]
    env->ReleaseByteArrayElements(imageData, buffer, JNI_ABORT);

    // 5) 构造输入 Mat
    ncnn::Mat in_mat(w, h, (void *) pixeldata, (size_t) c, c);

    int scale = inst->scale;
    // 6) 构造输出 Mat
    ncnn::Mat out_mat(w * scale, h * scale, (size_t) in_mat.elemsize, (int) in_mat.elemsize);

    // 7) 运行模型
    try {
        LOGI("processImage: processing");
        if (inst->process(in_mat, out_mat) != 0) {
            LOGE("processImage: model process failed");
            free(in_mat);
            free(out_mat);
            return nullptr;
        }
        free(pixeldata);
        LOGI("processImage: process ends");
    } catch (const std::exception &e) {
        // C++ 异常
        env->ThrowNew(runtimeExc, e.what());
        return nullptr;
    } catch (...) {
        // 任何其他崩溃
        env->ThrowNew(runtimeExc, "Unknown native error in RealCUGAN");
        return nullptr;
    }

    // 8) 打包成 Java byte[]
    int out_w = out_mat.w;
    int out_h = out_mat.h;
    int out_c = out_mat.elempack;
    jsize outLen = out_w * out_h * out_c;

    jbyteArray outArray = env->NewByteArray(outLen);
    env->SetByteArrayRegion(outArray, 0, outLen, reinterpret_cast<jbyte *>(out_mat.data));

    LOGI("processImage: complete. output size: %d x %d elempack: %d", out_w, out_h, out_c);

    return outArray;
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_aoihoshino_realcugan_1ncnn_1android_RealCUGAN_nativeRelease(
        JNIEnv * /*env*/, jclass, jlong handle) {
    // 1) 删除RealCUGAN实例
    RealCUGAN *inst = nullptr;
    {
        std::lock_guard lk(g_map_mutex);
        auto it = g_instances.find(handle);
        if (it != g_instances.end()) {
            inst = it->second;
            g_instances.erase(it);
        }
    }
    delete inst;

    release_ncnn_gpu();
}
