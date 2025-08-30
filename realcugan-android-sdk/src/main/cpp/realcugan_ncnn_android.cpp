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
    JavaVM *vm = nullptr;
    jobject cbGlobal = nullptr;         // Global ref to the ProgressListener object
    jmethodID midOnProgressF = nullptr; // void onProgress(float)
};

static void call_progress(ProgressCallback *pcb, float percent) {
    if (!pcb || !pcb->cbGlobal || !pcb->midOnProgressF) return;

    JNIEnv *env = nullptr;
    bool didAttach = false;
    if (pcb->vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK) {
        if (pcb->vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return;
        didAttach = true;
    }

    env->CallVoidMethod(pcb->cbGlobal, pcb->midOnProgressF, percent);

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
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
                     (long) handle, (void *) inst, kv.first.toString().c_str());
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


// ---------- Bitmap lock helpers (RGBA_8888 + SOFTWARE) ----------
static bool lockBitmapRGBA8888(JNIEnv *env, jobject bitmap,
                               AndroidBitmapInfo *info, void **pixels) {
    *pixels = nullptr;
    memset(info, 0, sizeof(*info));

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    const int gi = AndroidBitmap_getInfo(env, bitmap, info);
    if (gi != ANDROID_BITMAP_RESULT_SUCCESS) return false;
#if __ANDROID_API__ >= 26
    if (info->flags & ANDROID_BITMAP_FLAGS_IS_HARDWARE) return false;
#endif
    if (info->format != ANDROID_BITMAP_FORMAT_RGBA_8888) return false;

    const int rc = AndroidBitmap_lockPixels(env, bitmap, pixels);
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
        return false;
    }
    return rc == ANDROID_BITMAP_RESULT_SUCCESS;
}

static void unlockBitmap(JNIEnv *env, jobject bitmap) {
    (void) AndroidBitmap_unlockPixels(env, bitmap);
}

// ---------- Preferred path: Bitmap -> ncnn::Mat -> Bitmap ----------
extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_aoihoshino_realcugan_1ncnn_1android_RealCUGAN_nativeProcessBitmap(
        JNIEnv *env, jclass /*clazz*/,
        jlong handle,
        jobject inBitmap,   // ARGB_8888, mutable, SOFTWARE
        jobject outBitmap,  // ARGB_8888, mutable, SOFTWARE (size = in*scale)
        jobject onProgressFunction) {

    jclass runtimeExc = env->FindClass("java/lang/RuntimeException");
    if (!runtimeExc) return JNI_FALSE;

    RealCUGAN *inst = find_realcugan(handle);
    if (!inst) {
        LOGE("nativeProcessBitmap: instance of handle %lld not found", handle);
        return JNI_FALSE;
    }

    AndroidBitmapInfo inInfo{}, outInfo{};
    void *inPix = nullptr;
    void *outPix = nullptr;

    if (!lockBitmapRGBA8888(env, inBitmap, &inInfo, &inPix)) return JNI_FALSE;
    if (!lockBitmapRGBA8888(env, outBitmap, &outInfo, &outPix)) {
        unlockBitmap(env, inBitmap);
        return JNI_FALSE;
    }

    bool ok = true;
    ProgressCallback *pcb = nullptr;
    std::function<void(float)> progressCb; // empty by default

    if (onProgressFunction) {
        pcb = new ProgressCallback();
        env->GetJavaVM(&pcb->vm);
        pcb->cbGlobal = env->NewGlobalRef(onProgressFunction);
        jclass cbCls = env->GetObjectClass(onProgressFunction);
        pcb->midOnProgressF = env->GetMethodID(cbCls, "onProgress", "(F)V");
        env->DeleteLocalRef(cbCls);
        if (!pcb->midOnProgressF) {
            LOGW("ProgressListener missing void onProgress(float); progress ignored.");
        } else {
            progressCb = [pcb](float percent) { call_progress(pcb, percent); };
        }
    }

    try {
        const int w = static_cast<int>(inInfo.width);
        const int h = static_cast<int>(inInfo.height);

        // Debug: verify dimensions and strides to catch potential mismatch early
        LOGI("nativeProcessBitmap: in=%dx%d stride=%u, out=%dx%d stride=%u, channels=%d, scale(inst)=%d",
             w, h, (unsigned) inInfo.stride, (int) outInfo.width, (int) outInfo.height,
             (unsigned) outInfo.stride,
             4, inst->scale);
        if ((int) outInfo.width != w * inst->scale || (int) outInfo.height != h * inst->scale) {
            LOGE("nativeProcessBitmap: outBitmap size mismatch: expected %dx%d, got %dx%d",
                 w * inst->scale, h * inst->scale, (int) outInfo.width, (int) outInfo.height);
            ok = false;
        }
        if (!ok) {
            throw std::runtime_error("Output Bitmap size mismatch");
        }

        // NOTE:
        // realcugan.cpp 的 CPU 路径期望 inimage.data 指向的是 **紧凑连续的交错像素**(RGB/RGBA)，
        // 并通过 from_pixels_roi(pixeldata, PIXEL_..., w, h, ...) 自己再转 planar。
        // 因此，这里不能直接传入带 stride 的 Bitmap 像素，也不能先 from_pixels 成为 planar。
        // 我们拷贝一份去除 stride 的紧凑缓冲，再用外部 packed Mat 包一层：elemsize=1, elempack=channels。

        const int channels = 4; // lock 限定为 RGBA_8888
        const int in_stride_bytes = static_cast<int>(inInfo.stride);
        const int tight_row_bytes = w * channels; // 交错紧凑：每行 w * 4 字节
        const size_t tight_size = (size_t) tight_row_bytes * h;

        auto *interleaved_tight = (unsigned char *) malloc(tight_size);
        if (!interleaved_tight) {
            LOGE("nativeProcessBitmap: OOM allocating %zu bytes for input", tight_size);
            ok = false;
        }

        if (ok) {
            const auto *src = static_cast<const unsigned char *>(inPix);
            for (int y = 0; y < h; y++) {
                // 把每一行从 Bitmap 的 stride 区域拷到紧凑缓冲
                memcpy(interleaved_tight + (size_t) y * tight_row_bytes,
                       src + (size_t) y * in_stride_bytes,
                       tight_row_bytes);
            }

            // 用外部 packed image 的构造：elemsize=1(u8), elempack=channels(3/4)
            ncnn::Mat in = ncnn::Mat(w, h, (void *) interleaved_tight, (size_t) 1u, channels);

            // 以 outBitmap 的实际尺寸为准，避免 scale 不一致导致越界
            const int outW = static_cast<int>(outInfo.width);
            const int outH = static_cast<int>(outInfo.height);

            // 为输出分配一块紧凑交错缓冲（我们自己持有，避免 allocator/packing 差异）
            const int out_tight_row_bytes = outW * channels;
            const size_t out_tight_size = (size_t) out_tight_row_bytes * outH;
            auto *out_tight = (unsigned char *) malloc(out_tight_size);
            if (!out_tight) {
                LOGE("nativeProcessBitmap: OOM allocating %zu bytes for output", out_tight_size);
                ok = false;
            }

            if (ok) {
                // 用外部 packed image 包装输出缓冲：elemsize=1(u8), elempack=channels(3/4)
                ncnn::Mat out(outW, outH, (void *) out_tight, (size_t) 1u, channels);

                int ret = inst->process(in, out, progressCb);
                if (ret != 0) {
                    LOGE("nativeProcessBitmap: RealCUGAN::process failed (%d)", ret);
                    ok = false;
                } else {
                    // 将紧凑交错输出按 Bitmap 的 stride 写回
                    auto *dst = static_cast<unsigned char *>(outPix);
                    const int out_stride_bytes = static_cast<int>(outInfo.stride);
                    for (int y = 0; y < outH; y++) {
                        memcpy(dst + (size_t) y * out_stride_bytes,
                               out_tight + (size_t) y * out_tight_row_bytes,
                               out_tight_row_bytes);
                    }
                }

                free(out_tight);
            }

            // 释放输入临时缓冲
            free(interleaved_tight);
        }
    } catch (const std::exception &e) {
        env->ThrowNew(runtimeExc, e.what());
        ok = false;
    } catch (...) {
        env->ThrowNew(runtimeExc, "Unknown native error in nativeProcessBitmap");
        ok = false;
    }

    if (pcb) {
        env->DeleteGlobalRef(pcb->cbGlobal);
        delete pcb;
        pcb = nullptr;
    }
    unlockBitmap(env, outBitmap);
    unlockBitmap(env, inBitmap);
    return ok ? JNI_TRUE : JNI_FALSE;
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
