#include <jni.h>
#include <string>
#include <sstream>
#include <functional>
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
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct ProgressCallback {
    JavaVM* vm = nullptr;
    jobject cbGlobal = nullptr;         // Global ref to the ProgressListener object
    jmethodID midOnProgressF = nullptr; // void onProgress(float)
};

static void call_progress(ProgressCallback* pcb, float percent) {
    if (!pcb || !pcb->cbGlobal || !pcb->midOnProgressF) return;

    JNIEnv* env = nullptr;
    bool didAttach = false;
    if (pcb->vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        if (pcb->vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return;
        didAttach = true;
    }

    env->CallVoidMethod(pcb->cbGlobal, pcb->midOnProgressF, percent);

    if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); }
    if (didAttach) pcb->vm->DetachCurrentThread();
}

struct CUGANParams {
    int noise;
    int scale;
    int syncgap;
    bool ttaMode;
    int gpuId;
    std::string modelDir;

    bool operator==(const CUGANParams &o) const noexcept {
        return noise == o.noise
               && scale == o.scale
               && syncgap == o.syncgap
               && ttaMode == o.ttaMode
               && gpuId == o.gpuId
               && modelDir == o.modelDir;
    }

    [[nodiscard]] std::string toString() const {
        std::ostringstream oss;
        oss << "CUGANParams{"
            << "noise=" << noise << ", "
            << "scale=" << scale << ", "
            << "syncgap=" << syncgap << ", "
            << "ttaMode=" << (ttaMode ? "true" : "false") << ", "
            << "gpuId=" << gpuId << ", "
            << "modelDir=\"" << modelDir << "\""
            << "}";
        return oss.str();
    }
};

namespace std {
    template<>
    struct hash<CUGANParams> {
        size_t operator()(CUGANParams const &p) const noexcept {
            size_t h = std::hash<int>()(p.noise);
            auto mix = [&](auto v) {
                h ^= std::hash<decltype(v)>()(v)
                     + 0x9e3779b97f4a7c15 + (h << 6) + (h >> 2);
            };
            mix(p.scale);
            mix(p.syncgap);
            mix(p.ttaMode);
            mix(p.gpuId);
            mix(p.modelDir);
            return h;
        }
    };
}

struct CUGANEntry {
    jlong handle;
    RealCUGAN *inst;
};

// ncnn初始化锁
static std::mutex gpu_mutex;
static bool gpu_initialized = false;

// realcugan序列锁
static std::mutex g_cache_mutex;
static std::unordered_map<CUGANParams, CUGANEntry> g_cache;

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
    std::lock_guard<std::mutex> lg2(g_cache_mutex);
    if (g_cache.empty() && gpu_initialized) {
        ncnn::destroy_gpu_instance();
        gpu_initialized = false;
    }
}

RealCUGAN *find_realcugan(jlong handle) {
    RealCUGAN *inst = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_cache_mutex);
        for (auto &kv: g_cache) {
            if (kv.second.handle == handle) {
                inst = kv.second.inst;
                LOGI("found realcugan: handle=%ld inst=%p options=%s",
                     (long)handle, (void*)inst, kv.first.toString().c_str());
                break;
            }
        }
    }
    return inst;
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_aoihoshino_realcugan_1ncnn_1android_RealCUGAN_nativeInitialize(
        JNIEnv *env, jclass /* this */,
        jstring modelRootDir,
        jobject noiseObj,
        jobject scaleObj,
        jobject syncgapObj,
        jstring modelNameJ,
        jobject ttaModeObj,
        jobject gpuidObj
) {
    if (!g_cache.empty()) {
        LOGW("nativeInitialize: You have loaded more than one RealCUGAN instance. Too many RealCUGAN model being loaded can cause the heap to grow too large, leading to OOM.");
    }
    // 1. 异常类
    jclass runtimeExc = env->FindClass("java/lang/RuntimeException");
    if (!runtimeExc) return -1;

    // 2. 拆箱 Integer/Boolean
    jclass integerCls = env->FindClass("java/lang/Integer");
    jmethodID intValueID = env->GetMethodID(integerCls, "intValue", "()I");
    jclass booleanCls = env->FindClass("java/lang/Boolean");
    jmethodID boolValueID = env->GetMethodID(booleanCls, "booleanValue", "()Z");

    int noise = noiseObj ? env->CallIntMethod(noiseObj, intValueID) : -1;
    int scale = scaleObj ? env->CallIntMethod(scaleObj, intValueID) : 2;
    int syncgap = syncgapObj ? env->CallIntMethod(syncgapObj, intValueID) : 3;
    bool ttaMode = ttaModeObj && env->CallBooleanMethod(ttaModeObj, boolValueID);
    int numThreads = 1;

    std::string modelDir;
    if (modelNameJ) {
        const char *tmp = env->GetStringUTFChars(modelNameJ, nullptr);
        if (tmp && *tmp) modelDir = tmp;
        env->ReleaseStringUTFChars(modelNameJ, tmp);
    }

    // 3. GPU 初始化
    int gpuId;
    ensure_ncnn_gpu();
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
    if (gpuId == -1) release_ncnn_gpu();
    LOGI("initialize(): using GPU %d", gpuId);

    CUGANParams key{noise, scale, syncgap, ttaMode, gpuId, modelDir};
    {
        std::lock_guard<std::mutex> lk(g_cache_mutex);
        auto it = g_cache.find(key);
        if (it != g_cache.end()) {
            return it->second.handle;
        }
    }

    // 4. 基本边界检查（syncgap, gpuId 先行）
    if (syncgap < 0 || syncgap > 3) {
        LOGE("initialize(): invalid syncgap %d", syncgap);
        release_ncnn_gpu();
        return -1;
    }

    // 5. nose 模型强制关 syncgap
    if (modelDir == "models-nose") {
        syncgap = 0;
    }

    // 6. 分模型校验 scale/noise
    if (modelDir == "models-nose") {
        // 只允许 up2x-no-denoise
        if (scale != 2 || noise != 0) {
            LOGE("initialize(): models-nose only supports scale=2, noise=0");
            release_ncnn_gpu();
            return -1;
        }
    } else if (modelDir == "models-pro") {
        // 允许 up{2,3}x-{conservative,no-denoise,denoise3x}
        bool okScale = (scale == 2 || scale == 3);
        bool okNoise = (noise == -1 || noise == 0 || noise == 3);
        if (!okScale || !okNoise) {
            LOGE("initialize(): models-pro supports scale=2..3, noise in {-1,0,3}");
            release_ncnn_gpu();
            return -1;
        }
    } else if (modelDir == "models-se") {
        // 允许 up{2,3,4}x-{conservative,no-denoise,denoise1x..3x}
        bool okScale = (scale >= 2 && scale <= 4);
        bool okNoise = (noise >= -1 && noise <= 3);
        if (!okScale || !okNoise) {
            LOGE("initialize(): models-se supports scale=2..4, noise in -1..3");
            release_ncnn_gpu();
            return -1;
        }
    } else {
        LOGE("initialize(): unknown modelDir '%s'", modelDir.c_str());
        release_ncnn_gpu();
        return -1;
    }

    // 7. 计算 prepadding
    int prepadding = 0;
    if (scale == 2) prepadding = 18;
    else if (scale == 3) prepadding = 14;
    else if (scale == 4) prepadding = 19;

    // 8. 计算 tilesize (不变)

    int tilesize = 200;
    if (gpuId == -1) {
        tilesize = 200;
    } else {
        ncnn::VulkanDevice *vkdev = ncnn::get_gpu_device(gpuId);

        // 读取 Vulkan 物理设备属性
        const ncnn::GpuInfo &props = vkdev->info;

        uint32_t heap;
        // Qualcomm 的 vendorID 通常是 0x5143，或者 deviceName 中包含 "Adreno"
        bool isQualcomm = props.vendor_id() == 0x5143
                          || std::string(props.device_name()).find("Adreno") != std::string::npos;
        if (isQualcomm) {
            // … 前面初始化 VulkanDevice vkdev …
            float heapMB = 0.f;
            if (props.support_VK_EXT_memory_priority() && props.support_VK_EXT_memory_budget()) {
                // 使用扩展取真实预算
                VkPhysicalDeviceMemoryProperties2 mem2{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
                VkPhysicalDeviceMemoryBudgetPropertiesEXT bud{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
                mem2.pNext = &bud;
                ncnn::vkGetPhysicalDeviceMemoryProperties(props.physical_device(),
                                                          &mem2.memoryProperties);
                heapMB = bud.heapBudget[0] / float(1024 * 1024);
            } else {
                // Fallback：用总显存的 80%
                VkPhysicalDeviceMemoryProperties mp{};
                ncnn::vkGetPhysicalDeviceMemoryProperties(props.physical_device(), &mp);
                for (uint32_t i = 0; i < mp.memoryHeapCount; i++) {
                    if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                        heapMB = (mp.memoryHeaps[i].size * 0.8f) / float(1024 * 1024 * 8);
                        break;
                    }
                }
            }
            heap = heapMB;
            tilesize = (int) (heap / scale);
        } else {
            heap = vkdev->get_heap_budget();
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
            } else {
                if (heap > 1690) tilesize = 400;
                else if (heap > 980) tilesize = 300;
                else if (heap > 530) tilesize = 200;
                else if (heap > 240) tilesize = 100;
                else tilesize = 32;
            }
        }
        LOGI("heap=%u", heap);
    }


    // 9. 构造 param/bin 路径 & load 模型 … (不变)
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
    env->ReleaseStringUTFChars(modelRootDir, root);

    path_t paramFull = sanitize_filepath(parampath);
    path_t modelFull = sanitize_filepath(modelpath);
    if (access(paramFull.c_str(), F_OK) || access(modelFull.c_str(), F_OK)) {
        LOGE("model file not found: %s / %s", paramFull.c_str(), modelFull.c_str());
        release_ncnn_gpu();
        return -1;
    }

    // 10. 实例化 RealCUGAN 并 load
    auto *inst = new RealCUGAN(gpuId, ttaMode, numThreads);
    inst->noise = noise;
    inst->scale = scale;
    inst->syncgap = syncgap;
    inst->prepadding = prepadding;
    inst->tilesize = tilesize;

    try {
        int ret = inst->load(paramFull, modelFull);
        if (ret != 0) {
            LOGE("initialize: RealCUGAN::load failed (%d)", ret);
            delete inst;
            release_ncnn_gpu();
            return ret;
        }
    }
    catch (const std::exception &e) {
        env->ThrowNew(runtimeExc, e.what());
        return -1;
    }

    // 11. 注册实例并返回 handle
    jlong handle;
    {
        std::lock_guard<std::mutex> lg(handler_mutex);
        handle = next_handle++;
    }
    {
        std::lock_guard<std::mutex> lk2(g_cache_mutex);
        g_cache[key] = CUGANEntry{handle, inst};
    }
    return handle;
}


extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_github_aoihoshino_realcugan_1ncnn_1android_RealCUGAN_nativeProcessImage(
        JNIEnv *env, jclass /*clazz*/,
        jlong handle,
        jbyteArray imageData,
        jobject onProgressFunction) {
    // —— 1) 找到对应的 RealCUGAN 实例 —————————————
    // 1) Find the right instance under lock
    // 先声明要抛出的异常类
    jclass runtimeExc = env->FindClass("java/lang/RuntimeException");
    if (!runtimeExc) {
        return nullptr; // 如果连 RuntimeException 都找不到，直接回
    }
    RealCUGAN *inst = find_realcugan(handle);
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
    // ---- Thread-safe progress callback (ProgressListener.onProgress(float)) ----
    ProgressCallback* pcb = nullptr;
    std::function<void(float)> progressCb; // empty by default

    if (onProgressFunction) {
        pcb = new ProgressCallback();
        env->GetJavaVM(&pcb->vm);

        // Promote the ProgressListener object to a GlobalRef
        pcb->cbGlobal = env->NewGlobalRef(onProgressFunction);
        jclass cbCls = env->GetObjectClass(onProgressFunction);

        // Expect a method: void onProgress(float)
        pcb->midOnProgressF = env->GetMethodID(cbCls, "onProgress", "(F)V");
        env->DeleteLocalRef(cbCls);

        if (!pcb->midOnProgressF) {
            LOGW("ProgressListener does not implement void onProgress(float); progress will be ignored.");
        } else {
            // Lambda that is safe to call from any thread used by ncnn/RealCUGAN
            progressCb = [pcb](float percent) { call_progress(pcb, percent); };
        }
    }

    try {
        LOGI("processImage: processing");
        if (inst->process(in_mat, out_mat, progressCb) != 0) {
            LOGE("processImage: model process failed");
            free(in_mat);
            free(out_mat);
            if (pcb) {
                env->DeleteGlobalRef(pcb->cbGlobal);
                delete pcb;
                pcb = nullptr;
            }
            return nullptr;
        }
        free(pixeldata);
        LOGI("processImage: process ends");
        if (pcb) {
            env->DeleteGlobalRef(pcb->cbGlobal);
            delete pcb;
            pcb = nullptr;
        }
    } catch (const std::exception &e) {
        env->ThrowNew(runtimeExc, e.what());
        if (pcb) {
            env->DeleteGlobalRef(pcb->cbGlobal);
            delete pcb;
            pcb = nullptr;
        }
        return nullptr;
    } catch (...) {
        env->ThrowNew(runtimeExc, "Unknown native error in RealCUGAN");
        if (pcb) {
            env->DeleteGlobalRef(pcb->cbGlobal);
            delete pcb;
            pcb = nullptr;
        }
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
    std::lock_guard<std::mutex> lk(g_cache_mutex);
    for (auto it = g_cache.begin(); it != g_cache.end(); ++it) {
        if (it->second.handle == handle) {
            delete it->second.inst;
            g_cache.erase(it);
            release_ncnn_gpu();
            break;
        }
    }
}
