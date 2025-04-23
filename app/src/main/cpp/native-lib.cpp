#include <jni.h>
#include <string>
#include <android/log.h>
#include <android/bitmap.h>
#include "realcugan.h"
#include "filesystem_utils.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "webp_image.h"

#define LOG_TAG "RealCUGAN_NCNN"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// 全局 RealCUGAN 单例
static RealCUGAN* g_realcugan = nullptr;

extern "C" JNIEXPORT jint JNICALL
Java_io_github_aoihoshino_realcugan_1ncnn_1android_MainActivity_initialize(
        JNIEnv* env, jobject /* this */,
        jstring modelRootDir,
        jobject noiseObj,      // java.lang.Integer or null
        jobject scaleObj,      // java.lang.Integer or null
        jobject syncgapObj,    // java.lang.Integer or null
        jstring modelNameJ,    // java.lang.String or null
        jobject ttaModeObj    // java.lang.Boolean or null
) {
    // —— 1. 如果已存在实例，先释放 ——
    if (g_realcugan) {
        LOGI("initialize(): deleting previous RealCUGAN");
        delete g_realcugan;
        g_realcugan = nullptr;
        ncnn::destroy_gpu_instance();
    }

    // —— 2. 创建 GPU 实例 ——
    ncnn::create_gpu_instance();
    int gpuIndex = ncnn::get_default_gpu_index();
    LOGI("initialize(): using GPU %d", gpuIndex);

    // —— 3. 解析 Integer 和 Boolean 包装类型 ——
    jclass integerCls     = env->FindClass("java/lang/Integer");
    jmethodID intValueID  = env->GetMethodID(integerCls, "intValue", "()I");
    jclass booleanCls     = env->FindClass("java/lang/Boolean");
    jmethodID boolValueID = env->GetMethodID(booleanCls, "booleanValue", "()Z");

    int noise    = noiseObj   ? env->CallIntMethod(noiseObj,   intValueID) : -1;
    int scale    = scaleObj   ? env->CallIntMethod(scaleObj,   intValueID) :  2;
    int syncgap  = syncgapObj ? env->CallIntMethod(syncgapObj, intValueID) :  3;
    bool ttaMode = ttaModeObj != nullptr && env->CallBooleanMethod(ttaModeObj, boolValueID);
    int numThreads = 1;

    // —— 4. 参数合法性检查 ——
    if (noise < -1 || noise > 3) {
        LOGE("initialize(): invalid noise %d", noise);
        return -1;
    }
    if (scale < 1 || scale > 4) {
        LOGE("initialize(): invalid scale %d", scale);
        return -1;
    }
    if (syncgap < 0 || syncgap > 3) {
        LOGE("initialize(): invalid syncgap %d", syncgap);
        return -1;
    }

    // —— 5. 解析 modelName ——
    std::string modelDir = "models-se";
    if (modelNameJ) {
        const char* tmp = env->GetStringUTFChars(modelNameJ, nullptr);
        if (tmp && *tmp) modelDir = tmp;
        env->ReleaseStringUTFChars(modelNameJ, tmp);
    }
    // main.cpp 里 nose 模型强制关 syncgap
    if (modelDir == "models-nose") {
        syncgap = 0;
    }

    // —— 6. 计算 prepadding ——
    int prepadding = 0;
    if      (scale == 2) prepadding = 18;
    else if (scale == 3) prepadding = 14;
    else if (scale == 4) prepadding = 19;

    // —— 7. 单 GPU 时根据 heap 预算自动算 tilesize ——
    int tilesize = 0;
    if (scale == 1) {
        tilesize = 0;
    } else if (gpuIndex >= 0) {
        uint32_t heap = ncnn::get_gpu_device(gpuIndex)->get_heap_budget();
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

    const char* root = env->GetStringUTFChars(modelRootDir, nullptr);

    // —— 8. 构造模型文件路径 ——
    char parampath[256], modelpath[256];
    if (noise == -1) {
        sprintf(parampath, "%s/%s/up%dx-conservative.param", root, modelDir.c_str(), scale);
        sprintf(modelpath, "%s/%s/up%dx-conservative.bin",      root, modelDir.c_str(), scale);
    }
    else if (noise == 0) {
        sprintf(parampath, "%s/%s/up%dx-no-denoise.param", root, modelDir.c_str(), scale);
        sprintf(modelpath, "%s/%s/up%dx-no-denoise.bin",       root, modelDir.c_str(), scale);
    }
    else {
        sprintf(parampath, "%s/%s/up%dx-denoise%dx.param",     root, modelDir.c_str(), scale, noise);
        sprintf(modelpath, "%s/%s/up%dx-denoise%dx.bin",       root, modelDir.c_str(), scale, noise);
    }
    path_t paramFull = sanitize_filepath(parampath);
    path_t modelFull = sanitize_filepath(modelpath);

    LOGI("initialize(): noise=%d scale=%d syncgap=%d ttaMode=%d numThreads=%d prepadding=%d tilesize=%d modelDir=%s/%s",
         noise, scale, syncgap, ttaMode, numThreads, prepadding, tilesize, root, modelDir.c_str());

    LOGI("param file: %s", paramFull.c_str());
    LOGI("model file: %s", modelFull.c_str());
    if (access(paramFull.c_str(), F_OK) != 0 || access(modelFull.c_str(), F_OK) != 0) {
        LOGE("model file not found！");
        return -1;
    } else {
        LOGI("model loaded successfully.");
    }

    // —— 9. 创建 RealCUGAN 并 load 模型 ——
    g_realcugan = new RealCUGAN(gpuIndex, ttaMode, numThreads);
    g_realcugan->noise      = noise;
    g_realcugan->scale      = scale;
    g_realcugan->syncgap    = syncgap;
    g_realcugan->prepadding = prepadding;
    g_realcugan->tilesize   = tilesize;

    int ret = g_realcugan->load(paramFull, modelFull);
    if (ret != 0) {
        LOGE("initialize: RealCUGAN::load failed (%d)", ret);
        // 释放实例，防止后续 process 访问野指针
        delete g_realcugan;
        g_realcugan = nullptr;
        ncnn::destroy_gpu_instance();
        return ret;
    }
    LOGI("initialize: RealCUGAN::load succeeded");
    return ret;
}

extern "C" JNIEXPORT jobject JNICALL
Java_io_github_aoihoshino_realcugan_1ncnn_1android_MainActivity_processImage(
        JNIEnv* env, jobject /* this */,
        jbyteArray imageData,
        jobject  bitmap_out) {

    if (!g_realcugan) {
        LOGE("processImage: g_realcugan == nullptr (load failed?)");
        return nullptr;
    }

    // 1) 拿到原始字节
    jsize length = env->GetArrayLength(imageData);
    jbyte* buffer = env->GetByteArrayElements(imageData, nullptr);

    int w=0, h=0, c=0;
    unsigned char* pixeldata = nullptr;

    // 2) 尝试 WebP
    pixeldata = webp_load(reinterpret_cast<unsigned char*>(buffer), length, &w, &h, &c);

    // 3) 回退到 PNG/JPEG
    if (!pixeldata) {
        pixeldata = stbi_load_from_memory(
                reinterpret_cast<unsigned char*>(buffer),
                length, &w, &h, &c, 0
        );
        if (!pixeldata) {
            LOGE("processImage: not webp nor png/jpeg");
            env->ReleaseByteArrayElements(imageData, buffer, JNI_ABORT);
            return nullptr;
        }
    }

    env->ReleaseByteArrayElements(imageData, buffer, JNI_ABORT);

    // 4) 转成 ncnn::Mat
    ncnn::Mat in(w, h, (void*) pixeldata, (size_t)c, c);

    int scale = g_realcugan->scale; // 或者 g_realcugan 中你存下来的 scale

    // 5) 调用模型
    ncnn::Mat out(w * scale, h * scale, (size_t)in.elemsize, c);
    if (g_realcugan->process(in, out) != 0) {
        LOGE("processImage: model process failed");
        return nullptr;
    }
    free(pixeldata);

    // 6) 写回 Bitmap
    // 1) Get the bitmap info so we know the real width/height/stride
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap_out, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("processImage: getInfo failed");
        return nullptr;
    }
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("processImage: bitmap is not RGBA_8888");
        return nullptr;
    }

    // 2) Lock pixels
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap_out, &pixels) < 0) {
        LOGE("processImage: lockPixels failed");
        return nullptr;
    }

    // 3) Copy each row from your 3‑channel Mat to the 4‑channel bitmap
    auto* dstBase = static_cast<uint8_t*>(pixels);
    const auto* srcBase = reinterpret_cast<const uint8_t*>(out.data);
    auto width  = info.width;
    auto height = info.height;
    auto dstStride = info.stride;       // bytes per row in the bitmap
    auto srcRowBytes = width * 3;       // bytes per row in your 3‑channel Mat

    for (int y = 0; y < height; y++) {
        uint8_t* dstRow = dstBase + y * dstStride;
        const uint8_t* srcRow = srcBase + y * srcRowBytes;
        for (int x = 0; x < width; x++) {
            // pick R, G, B from srcRow
            dstRow[x*4 + 0] = srcRow[x*3 + 0];
            dstRow[x*4 + 1] = srcRow[x*3 + 1];
            dstRow[x*4 + 2] = srcRow[x*3 + 2];
            dstRow[x*4 + 3] = 255;   // full alpha
        }
    }

    // 4) Unlock
    AndroidBitmap_unlockPixels(env, bitmap_out);
    return bitmap_out;
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_aoihoshino_realcugan_1ncnn_1android_MainActivity_release(
        JNIEnv* /*env*/, jobject /* this */) {
    if (g_realcugan) {
        LOGI("release(): deleting RealCUGAN instance");
        delete g_realcugan;
        g_realcugan = nullptr;
        ncnn::destroy_gpu_instance();
    } else {
        LOGI("release(): already released");
    }
}
