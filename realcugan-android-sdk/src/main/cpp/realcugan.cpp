// realcugan implemented with ncnn library

#include "realcugan.h"

#include <algorithm>
#include <vector>
#include <map>

#define LOG_TAG "RealCUGAN_NCNN_ANDROID_NATIVE"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)


// ncnn
#include "cpu.h"

#include "realcugan_preproc.comp.hex.h"
#include "realcugan_postproc.comp.hex.h"
#include "realcugan_4x_postproc.comp.hex.h"
#include "realcugan_preproc_tta.comp.hex.h"
#include "realcugan_postproc_tta.comp.hex.h"
#include "realcugan_4x_postproc_tta.comp.hex.h"

class FeatureCache {
public:
    void clear() {
        gpu_cache.clear();
        cpu_cache.clear();
    }

    std::string make_key(int yi, int xi, int ti, const std::string &name) const {
        return std::to_string(yi) + "-" + std::to_string(xi) + "-" + std::to_string(ti) + "-" +
               name;
    }

    void load(int yi, int xi, int ti, const std::string &name, ncnn::VkMat &feat) {
        feat = gpu_cache[make_key(yi, xi, ti, name)];
    }

    void save(int yi, int xi, int ti, const std::string &name, ncnn::VkMat &feat) {
        gpu_cache[make_key(yi, xi, ti, name)] = feat;
    }

    void load(int yi, int xi, int ti, const std::string &name, ncnn::Mat &feat) {
        feat = cpu_cache[make_key(yi, xi, ti, name)];
    }

    void save(int yi, int xi, int ti, const std::string &name, ncnn::Mat &feat) {
        cpu_cache[make_key(yi, xi, ti, name)] = feat;
    }

public:
    std::map<std::string, ncnn::VkMat> gpu_cache;
    std::map<std::string, ncnn::Mat> cpu_cache;
};

RealCUGAN::RealCUGAN(int gpuid, bool _tta_mode, int num_threads) {
    vkdev = gpuid == -1 ? 0 : ncnn::get_gpu_device(gpuid);

    net.opt.num_threads = num_threads;

    realcugan_preproc = 0;
    realcugan_postproc = 0;
    realcugan_4x_postproc = 0;
    bicubic_2x = 0;
    bicubic_3x = 0;
    bicubic_4x = 0;
    tta_mode = _tta_mode;
}

RealCUGAN::~RealCUGAN() {
    // cleanup preprocess and postprocess pipeline
    {
        delete realcugan_preproc;
        delete realcugan_postproc;
    }

    bicubic_2x->destroy_pipeline(net.opt);
    delete bicubic_2x;

    bicubic_3x->destroy_pipeline(net.opt);
    delete bicubic_3x;

    bicubic_4x->destroy_pipeline(net.opt);
    delete bicubic_4x;
}

#if _WIN32
int RealCUGAN::load(const std::wstring& parampath, const std::wstring& modelpath)
#else

int RealCUGAN::load(const std::string &parampath, const std::string &modelpath)
#endif
{
    net.opt.use_vulkan_compute = (vkdev != nullptr);
    net.opt.use_fp16_packed = true;

    if (vkdev) {
        // Use device capability flags for safer defaults
        const auto &info = vkdev->info;
        net.opt.use_fp16_storage = info.support_fp16_storage();
        net.opt.use_fp16_arithmetic = info.support_fp16_arithmetic();
    } else {
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
    }

    // Only enable int8 storage when using int8 quantized models; default off for stability
    net.opt.use_int8_storage = false;

    net.set_vulkan_device(vkdev);

#if _WIN32
    {
        FILE* fp = _wfopen(parampath.c_str(), L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", parampath.c_str());
        }

        net.load_param(fp);

        fclose(fp);
    }
    {
        FILE* fp = _wfopen(modelpath.c_str(), L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", modelpath.c_str());
        }

        net.load_model(fp);

        fclose(fp);
    }
#else
    net.load_param(parampath.c_str());
    net.load_model(modelpath.c_str());
#endif

    // initialize preprocess and postprocess pipeline
    if (vkdev) {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty()) {
                    if (tta_mode)
                        compile_spirv_module(realcugan_preproc_tta_comp_data,
                                             sizeof(realcugan_preproc_tta_comp_data), net.opt,
                                             spirv);
                    else
                        compile_spirv_module(realcugan_preproc_comp_data,
                                             sizeof(realcugan_preproc_comp_data), net.opt, spirv);
                }
            }

            realcugan_preproc = new ncnn::Pipeline(vkdev);
            realcugan_preproc->set_optimal_local_size_xyz(8, 8, 3);
            realcugan_preproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty()) {
                    if (tta_mode)
                        compile_spirv_module(realcugan_postproc_tta_comp_data,
                                             sizeof(realcugan_postproc_tta_comp_data), net.opt,
                                             spirv);
                    else
                        compile_spirv_module(realcugan_postproc_comp_data,
                                             sizeof(realcugan_postproc_comp_data), net.opt, spirv);
                }
            }

            realcugan_postproc = new ncnn::Pipeline(vkdev);
            realcugan_postproc->set_optimal_local_size_xyz(8, 8, 3);
            realcugan_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty()) {
                    if (tta_mode)
                        compile_spirv_module(realcugan_4x_postproc_tta_comp_data,
                                             sizeof(realcugan_4x_postproc_tta_comp_data), net.opt,
                                             spirv);
                    else
                        compile_spirv_module(realcugan_4x_postproc_comp_data,
                                             sizeof(realcugan_4x_postproc_comp_data), net.opt,
                                             spirv);
                }
            }

            realcugan_4x_postproc = new ncnn::Pipeline(vkdev);
            realcugan_4x_postproc->set_optimal_local_size_xyz(8, 8, 3);
            realcugan_4x_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    // bicubic 2x/3x/4x for alpha channel
    {
        bicubic_2x = ncnn::create_layer("Interp");
        bicubic_2x->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 3);// bicubic
        pd.set(1, 2.f);
        pd.set(2, 2.f);
        bicubic_2x->load_param(pd);

        bicubic_2x->create_pipeline(net.opt);
    }
    {
        bicubic_3x = ncnn::create_layer("Interp");
        bicubic_3x->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 3);// bicubic
        pd.set(1, 3.f);
        pd.set(2, 3.f);
        bicubic_3x->load_param(pd);

        bicubic_3x->create_pipeline(net.opt);
    }
    {
        bicubic_4x = ncnn::create_layer("Interp");
        bicubic_4x->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 3);// bicubic
        pd.set(1, 4.f);
        pd.set(2, 4.f);
        bicubic_4x->load_param(pd);

        bicubic_4x->create_pipeline(net.opt);
    }

    return 0;
}

int RealCUGAN::process(const ncnn::Mat &inimage, ncnn::Mat &outimage,
                       const std::function<void(float)> &progress_listener) const {
    bool syncgap_needed = tilesize < std::max(inimage.w, inimage.h);

    if (!vkdev) {
        // cpu only
        if (syncgap_needed && syncgap) {
            if (syncgap == 1)
                return process_cpu_se(inimage, outimage, progress_listener);
            if (syncgap == 2)
                return process_cpu_se_rough(inimage, outimage, progress_listener);
            if (syncgap == 3)
                return process_cpu_se_very_rough(inimage, outimage, progress_listener);
        } else
            return process_cpu(inimage, outimage, progress_listener);
    }

    if (noise == -1 && scale == 1) {
        outimage = inimage;
        return 0;
    }

    if (syncgap_needed && syncgap) {
        if (syncgap == 1)
            return process_se(inimage, outimage, progress_listener);
        if (syncgap == 2)
            return process_se_rough(inimage, outimage, progress_listener);
        if (syncgap == 3)
            return process_se_very_rough(inimage, outimage, progress_listener);
    }

    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    ncnn::VkAllocator *blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator *staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++) {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3) {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4) {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        ncnn::Mat in;
        if (opt.use_fp16_storage && opt.use_int8_storage) {
            in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0),
                           (unsigned char *) pixeldata + in_tile_y0 * w * channels,
                           (size_t) channels, 1);
        } else {
            if (channels == 3) {
#if _WIN32
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR2RGB, w, (in_tile_y1 - in_tile_y0));
#else
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels,
                                            ncnn::Mat::PIXEL_RGB, w, (in_tile_y1 - in_tile_y0));
#endif
            }
            if (channels == 4) {
#if _WIN32
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGRA2RGBA, w, (in_tile_y1 - in_tile_y0));
#else
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels,
                                            ncnn::Mat::PIXEL_RGBA, w, (in_tile_y1 - in_tile_y0));
#endif
            }
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

        ncnn::VkMat out_gpu;
        if (opt.use_fp16_storage && opt.use_int8_storage) {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t) channels, 1,
                           blob_vkallocator);
        } else {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t) 4u, 1,
                           blob_vkallocator);
        }

        for (int xi = 0; xi < xtiles; xi++) {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3) {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4) {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            if (tta_mode) {
                // preproc
                ncnn::VkMat in_tile_gpu[8];
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, blob_vkallocator);

                    if (channels == 4) {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1,
                                                 in_out_tile_elemsize, 1, blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu[0];
                    bindings[2] = in_tile_gpu[1];
                    bindings[3] = in_tile_gpu[2];
                    bindings[4] = in_tile_gpu[3];
                    bindings[5] = in_tile_gpu[4];
                    bindings[6] = in_tile_gpu[5];
                    bindings[7] = in_tile_gpu[6];
                    bindings[8] = in_tile_gpu[7];
                    bindings[9] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < 8; ti++) {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("in0", in_tile_gpu[ti]);

                    ex.extract("out0", out_tile_gpu[ti], cmd);
                }

                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4) {
                    if (scale == 1) {
                        out_alpha_tile_gpu = in_alpha_tile_gpu;
                    }
                    if (scale == 2) {
                        bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 3) {
                        bicubic_3x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 4) {
                        bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                }

                // postproc
                if (scale == 4) {
                    std::vector<ncnn::VkMat> bindings(11);
                    bindings[0] = in_gpu;
                    bindings[1] = out_tile_gpu[0];
                    bindings[2] = out_tile_gpu[1];
                    bindings[3] = out_tile_gpu[2];
                    bindings[4] = out_tile_gpu[3];
                    bindings[5] = out_tile_gpu[4];
                    bindings[6] = out_tile_gpu[5];
                    bindings[7] = out_tile_gpu[6];
                    bindings[8] = out_tile_gpu[7];
                    bindings[9] = out_alpha_tile_gpu;
                    bindings[10] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(16);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = out_tile_gpu[0].w;
                    constants[4].i = out_tile_gpu[0].h;
                    constants[5].i = out_tile_gpu[0].cstep;
                    constants[6].i = out_gpu.w;
                    constants[7].i = out_gpu.h;
                    constants[8].i = out_gpu.cstep;
                    constants[9].i = xi * TILE_SIZE_X;
                    constants[10].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[11].i = xi * TILE_SIZE_X * scale;
                    constants[12].i = std::min(TILE_SIZE_X * scale,
                                               out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[13].i = channels;
                    constants[14].i = out_alpha_tile_gpu.w;
                    constants[15].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale,
                                            out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_4x_postproc, bindings, constants, dispatcher);
                } else {
                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = out_tile_gpu[0];
                    bindings[1] = out_tile_gpu[1];
                    bindings[2] = out_tile_gpu[2];
                    bindings[3] = out_tile_gpu[3];
                    bindings[4] = out_tile_gpu[4];
                    bindings[5] = out_tile_gpu[5];
                    bindings[6] = out_tile_gpu[6];
                    bindings[7] = out_tile_gpu[7];
                    bindings[8] = out_alpha_tile_gpu;
                    bindings[9] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = out_tile_gpu[0].w;
                    constants[1].i = out_tile_gpu[0].h;
                    constants[2].i = out_tile_gpu[0].cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale,
                                              out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = channels;
                    constants[9].i = out_alpha_tile_gpu.w;
                    constants[10].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale,
                                            out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_postproc, bindings, constants, dispatcher);
                }
            } else {
                // preproc
                ncnn::VkMat in_tile_gpu;
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                       in_out_tile_elemsize, 1, blob_vkallocator);

                    if (channels == 4) {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1,
                                                 in_out_tile_elemsize, 1, blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;
                    bindings[2] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu;
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("in0", in_tile_gpu);

                    ex.extract("out0", out_tile_gpu, cmd);
                }

                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4) {
                    if (scale == 1) {
                        out_alpha_tile_gpu = in_alpha_tile_gpu;
                    }
                    if (scale == 2) {
                        bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 3) {
                        bicubic_3x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 4) {
                        bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                }

                // postproc
                if (scale == 4) {
                    std::vector<ncnn::VkMat> bindings(4);
                    bindings[0] = in_gpu;
                    bindings[1] = out_tile_gpu;
                    bindings[2] = out_alpha_tile_gpu;
                    bindings[3] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(16);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = out_tile_gpu.w;
                    constants[4].i = out_tile_gpu.h;
                    constants[5].i = out_tile_gpu.cstep;
                    constants[6].i = out_gpu.w;
                    constants[7].i = out_gpu.h;
                    constants[8].i = out_gpu.cstep;
                    constants[9].i = xi * TILE_SIZE_X;
                    constants[10].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[11].i = xi * TILE_SIZE_X * scale;
                    constants[12].i = std::min(TILE_SIZE_X * scale,
                                               out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[13].i = channels;
                    constants[14].i = out_alpha_tile_gpu.w;
                    constants[15].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale,
                                            out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_4x_postproc, bindings, constants, dispatcher);
                } else {
                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = out_tile_gpu;
                    bindings[1] = out_alpha_tile_gpu;
                    bindings[2] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = out_tile_gpu.w;
                    constants[1].i = out_tile_gpu.h;
                    constants[2].i = out_tile_gpu.cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale,
                                              out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = channels;
                    constants[9].i = out_alpha_tile_gpu.w;
                    constants[10].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale,
                                            out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_postproc, bindings, constants, dispatcher);
                }
            }

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }

            report_progress(xtiles, ytiles, yi, xi, progress_listener);
        }

        // download
        {
            ncnn::Mat out;

            if (opt.use_fp16_storage && opt.use_int8_storage) {
                out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned char *) outimage.data +
                                                      yi * scale * TILE_SIZE_Y * w * scale *
                                                      channels, (size_t) channels, 1);
            }

            cmd.record_clone(out_gpu, out, opt);

            cmd.submit_and_wait();

            if (!(opt.use_fp16_storage && opt.use_int8_storage)) {
                if (channels == 3) {
#if _WIN32
                    out.to_pixels((unsigned char*)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, ncnn::Mat::PIXEL_RGB2BGR);
#else
                    out.to_pixels((unsigned char *) outimage.data +
                                  yi * scale * TILE_SIZE_Y * w * scale * channels,
                                  ncnn::Mat::PIXEL_RGB);
#endif
                }
                if (channels == 4) {
#if _WIN32
                    out.to_pixels((unsigned char*)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, ncnn::Mat::PIXEL_RGBA2BGRA);
#else
                    out.to_pixels((unsigned char *) outimage.data +
                                  yi * scale * TILE_SIZE_Y * w * scale * channels,
                                  ncnn::Mat::PIXEL_RGBA);
#endif
                }
            }
        }
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int RealCUGAN::process_cpu(const ncnn::Mat &inimage, ncnn::Mat &outimage,
                           const std::function<void(float)> &progress_listener) const {
    if (noise == -1 && scale == 1) {
        outimage = inimage;
        return 0;
    }

    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    ncnn::Option opt = net.opt;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    for (int yi = 0; yi < ytiles; yi++) {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3) {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4) {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        for (int xi = 0; xi < xtiles; xi++) {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3) {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4) {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            int in_tile_x0 = std::max(xi * TILE_SIZE_X - prepadding, 0);
            int in_tile_x1 = std::min((xi + 1) * TILE_SIZE_X + prepadding_right, w);

            // crop tile
            ncnn::Mat in;
            {
                if (channels == 3) {
#if _WIN32
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_BGR2RGB, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
#else
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_RGB, w, h,
                                                    in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0,
                                                    in_tile_y1 - in_tile_y0);
#endif
                }
                if (channels == 4) {
#if _WIN32
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_BGRA2RGBA, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
#else
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_RGBA, w, h,
                                                    in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0,
                                                    in_tile_y1 - in_tile_y0);
#endif
                }
            }

            ncnn::Mat out;

            if (tta_mode) {
                // split alpha and preproc
                ncnn::Mat in_tile[8];
                ncnn::Mat in_alpha_tile;
                {
                    in_tile[0].create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++) {
                        const float *ptr = in.channel(q);
                        float *outptr0 = in_tile[0].channel(q);

                        for (int i = 0; i < in.h; i++) {
                            for (int j = 0; j < in.w; j++) {
                                *outptr0++ = *ptr++ * (1 / 255.f);
                            }
                        }
                    }

                    if (channels == 4) {
                        in_alpha_tile = in.channel_range(3, 1).clone();
                    }
                }

                // alpha tile cut to nopad size (CPU tta fix)
                ncnn::Mat in_alpha_tile_nocrop;
                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(
                            std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h,
                                     prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w,
                                                      prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile[0], in_tile_padded, pad_top, pad_bottom,
                                           pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile[0] = in_tile_padded;

                    if (channels == 4) {
                        ncnn::copy_cut_border(in_alpha_tile,
                                              in_alpha_tile_nocrop,
                                              pad_top, pad_bottom, pad_left, pad_right,
                                              opt);
                        // ensure alpha tile matches nopad size to avoid OOB
                        if (in_alpha_tile_nocrop.w != tile_w_nopad ||
                            in_alpha_tile_nocrop.h != tile_h_nopad) {
                            int extra_w = in_alpha_tile_nocrop.w - tile_w_nopad;
                            int extra_h = in_alpha_tile_nocrop.h - tile_h_nopad;
                            int fix_left = extra_w > 0 ? extra_w / 2 : 0;
                            int fix_right = extra_w > 0 ? extra_w - fix_left : 0;
                            int fix_top = extra_h > 0 ? extra_h / 2 : 0;
                            int fix_bottom = extra_h > 0 ? extra_h - fix_top : 0;
                            if (fix_left || fix_right || fix_top || fix_bottom) {
                                ncnn::Mat _alpha_fix;
                                ncnn::copy_cut_border(in_alpha_tile_nocrop, _alpha_fix,
                                                      fix_top, fix_bottom, fix_left, fix_right,
                                                      opt);
                                in_alpha_tile_nocrop = _alpha_fix;
                            }
                        }
                    }
                }

                // the other 7 directions
                {
                    in_tile[1].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[2].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[3].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[4].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[5].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[6].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[7].create(in_tile[0].h, in_tile[0].w, 3);

                    for (int q = 0; q < 3; q++) {
                        const ncnn::Mat in_tile_0 = in_tile[0].channel(q);
                        ncnn::Mat in_tile_1 = in_tile[1].channel(q);
                        ncnn::Mat in_tile_2 = in_tile[2].channel(q);
                        ncnn::Mat in_tile_3 = in_tile[3].channel(q);
                        ncnn::Mat in_tile_4 = in_tile[4].channel(q);
                        ncnn::Mat in_tile_5 = in_tile[5].channel(q);
                        ncnn::Mat in_tile_6 = in_tile[6].channel(q);
                        ncnn::Mat in_tile_7 = in_tile[7].channel(q);

                        for (int i = 0; i < in_tile[0].h; i++) {
                            const float *outptr0 = in_tile_0.row(i);
                            float *outptr1 = in_tile_1.row(in_tile[0].h - 1 - i);
                            float *outptr2 = in_tile_2.row(i) + in_tile[0].w - 1;
                            float *outptr3 = in_tile_3.row(in_tile[0].h - 1 - i) + in_tile[0].w - 1;

                            for (int j = 0; j < in_tile[0].w; j++) {
                                float *outptr4 = in_tile_4.row(j) + i;
                                float *outptr5 = in_tile_5.row(in_tile[0].w - 1 - j) + i;
                                float *outptr6 = in_tile_6.row(j) + in_tile[0].h - 1 - i;
                                float *outptr7 =
                                        in_tile_7.row(in_tile[0].w - 1 - j) + in_tile[0].h - 1 - i;

                                float v = *outptr0++;

                                *outptr1++ = v;
                                *outptr2-- = v;
                                *outptr3-- = v;
                                *outptr4 = v;
                                *outptr5 = v;
                                *outptr6 = v;
                                *outptr7 = v;
                            }
                        }
                    }
                }

                // realcugan
                ncnn::Mat out_tile[8];
                for (int ti = 0; ti < 8; ti++) {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile[ti]);

                    ex.extract("out0", out_tile[ti]);
                }

                ncnn::Mat out_alpha_tile;
                if (channels == 4) {
                    if (scale == 1) {
                        out_alpha_tile = in_alpha_tile_nocrop;
                    }
                    if (scale == 2) {
                        bicubic_2x->forward(in_alpha_tile_nocrop, out_alpha_tile, opt);
                    }
                    if (scale == 3) {
                        bicubic_3x->forward(in_alpha_tile_nocrop, out_alpha_tile, opt);
                    }
                    if (scale == 4) {
                        bicubic_4x->forward(in_alpha_tile_nocrop, out_alpha_tile, opt);
                    }
                }

                // postproc and merge alpha
                {
                    out.create(tile_w_nopad * scale, tile_h_nopad * scale, channels);
                    if (scale == 4) {
                        for (int q = 0; q < 3; q++) {
                            const ncnn::Mat out_tile_0 = out_tile[0].channel(q);
                            const ncnn::Mat out_tile_1 = out_tile[1].channel(q);
                            const ncnn::Mat out_tile_2 = out_tile[2].channel(q);
                            const ncnn::Mat out_tile_3 = out_tile[3].channel(q);
                            const ncnn::Mat out_tile_4 = out_tile[4].channel(q);
                            const ncnn::Mat out_tile_5 = out_tile[5].channel(q);
                            const ncnn::Mat out_tile_6 = out_tile[6].channel(q);
                            const ncnn::Mat out_tile_7 = out_tile[7].channel(q);
                            float *outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++) {
                                const float *inptr =
                                        in_tile[0].channel(q).row(prepadding + i / 4) + prepadding;
                                const float *ptr0 = out_tile_0.row(i);
                                const float *ptr1 = out_tile_1.row(out_tile[0].h - 1 - i);
                                const float *ptr2 = out_tile_2.row(i) + out_tile[0].w - 1;
                                const float *ptr3 =
                                        out_tile_3.row(out_tile[0].h - 1 - i) + out_tile[0].w - 1;

                                for (int j = 0; j < out.w; j++) {
                                    const float *ptr4 = out_tile_4.row(j) + i;
                                    const float *ptr5 = out_tile_5.row(out_tile[0].w - 1 - j) + i;
                                    const float *ptr6 = out_tile_6.row(j) + out_tile[0].h - 1 - i;
                                    const float *ptr7 =
                                            out_tile_7.row(out_tile[0].w - 1 - j) + out_tile[0].h -
                                            1 - i;

                                    float v =
                                            (*ptr0++ + *ptr1++ + *ptr2-- + *ptr3-- + *ptr4 + *ptr5 +
                                             *ptr6 + *ptr7) / 8;

                                    *outptr++ = v * 255.f + 0.5f + inptr[j / 4] * 255.f;
                                }
                            }
                        }
                    } else {
                        for (int q = 0; q < 3; q++) {
                            const ncnn::Mat out_tile_0 = out_tile[0].channel(q);
                            const ncnn::Mat out_tile_1 = out_tile[1].channel(q);
                            const ncnn::Mat out_tile_2 = out_tile[2].channel(q);
                            const ncnn::Mat out_tile_3 = out_tile[3].channel(q);
                            const ncnn::Mat out_tile_4 = out_tile[4].channel(q);
                            const ncnn::Mat out_tile_5 = out_tile[5].channel(q);
                            const ncnn::Mat out_tile_6 = out_tile[6].channel(q);
                            const ncnn::Mat out_tile_7 = out_tile[7].channel(q);
                            float *outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++) {
                                const float *ptr0 = out_tile_0.row(i);
                                const float *ptr1 = out_tile_1.row(out_tile[0].h - 1 - i);
                                const float *ptr2 = out_tile_2.row(i) + out_tile[0].w - 1;
                                const float *ptr3 =
                                        out_tile_3.row(out_tile[0].h - 1 - i) + out_tile[0].w - 1;

                                for (int j = 0; j < out.w; j++) {
                                    const float *ptr4 = out_tile_4.row(j) + i;
                                    const float *ptr5 = out_tile_5.row(out_tile[0].w - 1 - j) + i;
                                    const float *ptr6 = out_tile_6.row(j) + out_tile[0].h - 1 - i;
                                    const float *ptr7 =
                                            out_tile_7.row(out_tile[0].w - 1 - j) + out_tile[0].h -
                                            1 - i;

                                    float v =
                                            (*ptr0++ + *ptr1++ + *ptr2-- + *ptr3-- + *ptr4 + *ptr5 +
                                             *ptr6 + *ptr7) / 8;

                                    *outptr++ = v * 255.f + 0.5f;
                                }
                            }
                        }
                    }

                    if (channels == 4) {
                        memcpy(out.channel_range(3, 1), out_alpha_tile,
                               out_alpha_tile.total() * sizeof(float));
                    }
                }
            } else {
                // split alpha and preproc
                ncnn::Mat in_tile;
                ncnn::Mat in_alpha_tile;
                ncnn::Mat in_alpha_tile_nocrop;
                {
                    in_tile.create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++) {
                        const float *ptr = in.channel(q);
                        float *outptr = in_tile.channel(q);

                        for (int i = 0; i < in.w * in.h; i++) {
                            *outptr++ = *ptr++ * (1 / 255.f);
                        }
                    }

                    if (channels == 4) {
                        in_alpha_tile = in.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(
                            std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h,
                                     prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w,
                                                      prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile, in_tile_padded, pad_top, pad_bottom, pad_left,
                                           pad_right, 2, 0.f, net.opt);
                    in_tile = in_tile_padded;

                    // alpha tile cut to nopad
                    if (channels == 4) {
                        ncnn::copy_cut_border(
                                in_alpha_tile,
                                in_alpha_tile_nocrop,
                                pad_top, pad_bottom, pad_left, pad_right,
                                opt);
                        // ensure alpha tile matches nopad size to avoid OOB
                        if (in_alpha_tile_nocrop.w != tile_w_nopad ||
                            in_alpha_tile_nocrop.h != tile_h_nopad) {
                            int extra_w = in_alpha_tile_nocrop.w - tile_w_nopad;
                            int extra_h = in_alpha_tile_nocrop.h - tile_h_nopad;
                            int fix_left = extra_w > 0 ? extra_w / 2 : 0;
                            int fix_right = extra_w > 0 ? extra_w - fix_left : 0;
                            int fix_top = extra_h > 0 ? extra_h / 2 : 0;
                            int fix_bottom = extra_h > 0 ? extra_h - fix_top : 0;
                            if (fix_left || fix_right || fix_top || fix_bottom) {
                                ncnn::Mat _alpha_fix;
                                ncnn::copy_cut_border(in_alpha_tile_nocrop, _alpha_fix,
                                                      fix_top, fix_bottom, fix_left, fix_right,
                                                      opt);
                                in_alpha_tile_nocrop = _alpha_fix;
                            }
                        }
                    }
                }

                // alpha tile cut to nopad size (CPU non-tta fix)
                if (channels == 4) {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(
                            std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h,
                                     prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w,
                                                      prepadding_right), 0);

                    ncnn::copy_cut_border(in_alpha_tile,
                                          in_alpha_tile_nocrop,
                                          pad_top, pad_bottom, pad_left, pad_right,
                                          net.opt);
                    // ensure alpha tile matches nopad size to avoid OOB
                    if (in_alpha_tile_nocrop.w != tile_w_nopad ||
                        in_alpha_tile_nocrop.h != tile_h_nopad) {
                        int extra_w = in_alpha_tile_nocrop.w - tile_w_nopad;
                        int extra_h = in_alpha_tile_nocrop.h - tile_h_nopad;
                        int fix_left = extra_w > 0 ? extra_w / 2 : 0;
                        int fix_right = extra_w > 0 ? extra_w - fix_left : 0;
                        int fix_top = extra_h > 0 ? extra_h / 2 : 0;
                        int fix_bottom = extra_h > 0 ? extra_h - fix_top : 0;
                        if (fix_left || fix_right || fix_top || fix_bottom) {
                            ncnn::Mat _alpha_fix;
                            ncnn::copy_cut_border(in_alpha_tile_nocrop, _alpha_fix,
                                                  fix_top, fix_bottom, fix_left, fix_right,
                                                  net.opt);
                            in_alpha_tile_nocrop = _alpha_fix;
                        }
                    }
                }

                // realcugan
                ncnn::Mat out_tile;
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile);

                    ex.extract("out0", out_tile);
                }

                ncnn::Mat out_alpha_tile;
                if (channels == 4) {
                    if (scale == 1) {
                        out_alpha_tile = in_alpha_tile_nocrop;
                    }
                    if (scale == 2) {
                        bicubic_2x->forward(in_alpha_tile_nocrop, out_alpha_tile, opt);
                    }
                    if (scale == 3) {
                        bicubic_3x->forward(in_alpha_tile_nocrop, out_alpha_tile, opt);
                    }
                    if (scale == 4) {
                        bicubic_4x->forward(in_alpha_tile_nocrop, out_alpha_tile, opt);
                    }
                }

                // postproc and merge alpha
                {
                    out.create(tile_w_nopad * scale, tile_h_nopad * scale, channels);
                    if (scale == 4) {
                        for (int q = 0; q < 3; q++) {
                            float *outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++) {
                                const float *inptr =
                                        in_tile.channel(q).row(prepadding + i / 4) + prepadding;
                                const float *ptr = out_tile.channel(q).row(i);

                                for (int j = 0; j < out.w; j++) {
                                    *outptr++ = *ptr++ * 255.f + 0.5f + inptr[j / 4] * 255.f;
                                }
                            }
                        }
                    } else {
                        for (int q = 0; q < 3; q++) {
                            float *outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++) {
                                const float *ptr = out_tile.channel(q).row(i);

                                for (int j = 0; j < out.w; j++) {
                                    *outptr++ = *ptr++ * 255.f + 0.5f;
                                }
                            }
                        }
                    }

                    if (channels == 4) {
                        memcpy(out.channel_range(3, 1), out_alpha_tile,
                               out_alpha_tile.total() * sizeof(float));
                    }
                }
            }

            {
                if (channels == 3) {
#if _WIN32
                    out.to_pixels((unsigned char*)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels + xi * scale * TILE_SIZE_X * channels, ncnn::Mat::PIXEL_RGB2BGR, w * scale * channels);
#else
                    out.to_pixels((unsigned char *) outimage.data +
                                  yi * scale * TILE_SIZE_Y * w * scale * channels +
                                  xi * scale * TILE_SIZE_X * channels, ncnn::Mat::PIXEL_RGB,
                                  w * scale * channels);
#endif
                }
                if (channels == 4) {
#if _WIN32
                    out.to_pixels((unsigned char*)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels + xi * scale * TILE_SIZE_X * channels, ncnn::Mat::PIXEL_RGBA2BGRA, w * scale * channels);
#else
                    out.to_pixels((unsigned char *) outimage.data +
                                  yi * scale * TILE_SIZE_Y * w * scale * channels +
                                  xi * scale * TILE_SIZE_X * channels, ncnn::Mat::PIXEL_RGBA,
                                  w * scale * channels);
#endif
                }
            }
            report_progress(xtiles, ytiles, yi, xi, progress_listener);
        }
    }

    return 0;
}

int RealCUGAN::process_se(const ncnn::Mat &inimage, ncnn::Mat &outimage,
                          const std::function<void(float)> &progress_listener) const {
    ncnn::VkAllocator *blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator *staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    FeatureCache cache;

    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0"};
    process_se_stage0(inimage, in0, out0, opt, cache, progress_listener);

    std::vector<std::string> gap0 = {"gap0"};
    process_se_sync_gap(inimage, gap0, opt, cache, progress_listener);

    std::vector<std::string> in1 = {"gap0"};
    std::vector<std::string> out1 = {"gap1"};
    process_se_stage0(inimage, in1, out1, opt, cache, progress_listener);

    std::vector<std::string> gap1 = {"gap1"};
    process_se_sync_gap(inimage, gap1, opt, cache, progress_listener);

    std::vector<std::string> in2 = {"gap0", "gap1"};
    std::vector<std::string> out2 = {"gap2"};
    process_se_stage0(inimage, in2, out2, opt, cache, progress_listener);

    std::vector<std::string> gap2 = {"gap2"};
    process_se_sync_gap(inimage, gap2, opt, cache, progress_listener);

    std::vector<std::string> in3 = {"gap0", "gap1", "gap2"};
    std::vector<std::string> out3 = {"gap3"};
    process_se_stage0(inimage, in3, out3, opt, cache, progress_listener);

    std::vector<std::string> gap3 = {"gap3"};
    process_se_sync_gap(inimage, gap3, opt, cache, progress_listener);

    std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_stage2(inimage, in4, outimage, opt, cache, progress_listener);

    cache.clear();

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int RealCUGAN::process_se_rough(const ncnn::Mat &inimage, ncnn::Mat &outimage,
                                const std::function<void(float)> &progress_listener) const {
    ncnn::VkAllocator *blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator *staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    FeatureCache cache;

    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_stage0(inimage, in0, out0, opt, cache, progress_listener);

    std::vector<std::string> gap0 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_sync_gap(inimage, gap0, opt, cache, progress_listener);

    std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_stage2(inimage, in4, outimage, opt, cache, progress_listener);

    cache.clear();

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int RealCUGAN::process_se_very_rough(const ncnn::Mat &inimage, ncnn::Mat &outimage,
                                     const std::function<void(float)> &progress_listener) const {
    ncnn::VkAllocator *blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator *staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    FeatureCache cache;

    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_very_rough_stage0(inimage, in0, out0, opt, cache, progress_listener);

    std::vector<std::string> gap0 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_very_rough_sync_gap(inimage, gap0, opt, cache, progress_listener);

    std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_stage2(inimage, in4, outimage, opt, cache, progress_listener);

    cache.clear();

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int RealCUGAN::process_cpu_se(const ncnn::Mat &inimage, ncnn::Mat &outimage,
                              const std::function<void(float)> &progress_listener) const {
    FeatureCache cache;

    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0"};
    process_cpu_se_stage0(inimage, in0, out0, cache, progress_listener);

    std::vector<std::string> gap0 = {"gap0"};
    process_cpu_se_sync_gap(inimage, gap0, cache, progress_listener);

    std::vector<std::string> in1 = {"gap0"};
    std::vector<std::string> out1 = {"gap1"};
    process_cpu_se_stage0(inimage, in1, out1, cache, progress_listener);

    std::vector<std::string> gap1 = {"gap1"};
    process_cpu_se_sync_gap(inimage, gap1, cache, progress_listener);

    std::vector<std::string> in2 = {"gap0", "gap1"};
    std::vector<std::string> out2 = {"gap2"};
    process_cpu_se_stage0(inimage, in2, out2, cache, progress_listener);

    std::vector<std::string> gap2 = {"gap2"};
    process_cpu_se_sync_gap(inimage, gap2, cache, progress_listener);

    std::vector<std::string> in3 = {"gap0", "gap1", "gap2"};
    std::vector<std::string> out3 = {"gap3"};
    process_cpu_se_stage0(inimage, in3, out3, cache, progress_listener);

    std::vector<std::string> gap3 = {"gap3"};
    process_cpu_se_sync_gap(inimage, gap3, cache, progress_listener);

    std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
    process_cpu_se_stage2(inimage, in4, outimage, cache, progress_listener);

    cache.clear();

    return 0;
}

int RealCUGAN::process_cpu_se_rough(const ncnn::Mat &inimage, ncnn::Mat &outimage,
                                    const std::function<void(float)> &progress_listener) const {
    FeatureCache cache;

    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0", "gap1", "gap2", "gap3"};
    process_cpu_se_stage0(inimage, in0, out0, cache, progress_listener);

    std::vector<std::string> gap0 = {"gap0", "gap1", "gap2", "gap3"};
    process_cpu_se_sync_gap(inimage, gap0, cache, progress_listener);

    std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
    process_cpu_se_stage2(inimage, in4, outimage, cache, progress_listener);

    cache.clear();

    return 0;
}

int RealCUGAN::process_cpu_se_very_rough(const ncnn::Mat &inimage, ncnn::Mat &outimage,
                                         const std::function<void(
                                                 float)> &progress_listener) const {
    FeatureCache cache;

    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0", "gap1", "gap2", "gap3"};
    process_cpu_se_very_rough_stage0(inimage, in0, out0, cache, progress_listener);

    std::vector<std::string> gap0 = {"gap0", "gap1", "gap2", "gap3"};
    process_cpu_se_very_rough_sync_gap(inimage, gap0, cache, progress_listener);

    std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
    process_cpu_se_stage2(inimage, in4, outimage, cache, progress_listener);

    cache.clear();

    return 0;
}

int RealCUGAN::process_se_stage0(const ncnn::Mat &inimage, const std::vector<std::string> &names,
                                 const std::vector<std::string> &outnames, const ncnn::Option &opt,
                                 FeatureCache &cache,
                                 const std::function<void(float)> &progress_listener) const {
    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++) {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3) {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4) {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        ncnn::Mat in;
        if (opt.use_fp16_storage && opt.use_int8_storage) {
            in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0),
                           (unsigned char *) pixeldata + in_tile_y0 * w * channels,
                           (size_t) channels, 1);
        } else {
            if (channels == 3) {
#if _WIN32
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR2RGB, w, (in_tile_y1 - in_tile_y0));
#else
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels,
                                            ncnn::Mat::PIXEL_RGB, w, (in_tile_y1 - in_tile_y0));
#endif
            }
            if (channels == 4) {
#if _WIN32
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGRA2RGBA, w, (in_tile_y1 - in_tile_y0));
#else
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels,
                                            ncnn::Mat::PIXEL_RGBA, w, (in_tile_y1 - in_tile_y0));
#endif
            }
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

        ncnn::VkMat out_gpu;
        if (opt.use_fp16_storage && opt.use_int8_storage) {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t) channels, 1,
                           opt.blob_vkallocator);
        } else {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t) 4u, 1,
                           opt.blob_vkallocator);
        }

        for (int xi = 0; xi < xtiles; xi++) {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3) {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4) {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            if (tta_mode) {
                // preproc
                ncnn::VkMat in_tile_gpu[8];
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    if (channels == 4) {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1,
                                                 in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu[0];
                    bindings[2] = in_tile_gpu[1];
                    bindings[3] = in_tile_gpu[2];
                    bindings[4] = in_tile_gpu[3];
                    bindings[5] = in_tile_gpu[4];
                    bindings[6] = in_tile_gpu[5];
                    bindings[7] = in_tile_gpu[6];
                    bindings[8] = in_tile_gpu[7];
                    bindings[9] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < 8; ti++) {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu[ti]);

                    for (size_t i = 0; i < names.size(); i++) {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++) {
                        ncnn::VkMat feat;
                        ex.extract(outnames[i].c_str(), feat, cmd);

                        cache.save(yi, xi, ti, outnames[i], feat);
                    }
                }
            } else {
                // preproc
                ncnn::VkMat in_tile_gpu;
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                       in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    if (channels == 4) {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1,
                                                 in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;
                    bindings[2] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu);

                    for (size_t i = 0; i < names.size(); i++) {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++) {
                        ncnn::VkMat feat;
                        ex.extract(outnames[i].c_str(), feat, cmd);

                        cache.save(yi, xi, 0, outnames[i], feat);
                    }
                }
            }

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        cmd.submit_and_wait();
        cmd.reset();
    }

    return 0;
}

int RealCUGAN::process_se_stage2(const ncnn::Mat &inimage, const std::vector<std::string> &names,
                                 ncnn::Mat &outimage, const ncnn::Option &opt, FeatureCache &cache,
                                 const std::function<void(float)> &progress_listener) const {
    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++) {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3) {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4) {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        ncnn::Mat in;
        if (opt.use_fp16_storage && opt.use_int8_storage) {
            in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0),
                           (unsigned char *) pixeldata + in_tile_y0 * w * channels,
                           (size_t) channels, 1);
        } else {
            if (channels == 3) {
#if _WIN32
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR2RGB, w, (in_tile_y1 - in_tile_y0));
#else
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels,
                                            ncnn::Mat::PIXEL_RGB, w, (in_tile_y1 - in_tile_y0));
#endif
            }
            if (channels == 4) {
#if _WIN32
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGRA2RGBA, w, (in_tile_y1 - in_tile_y0));
#else
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels,
                                            ncnn::Mat::PIXEL_RGBA, w, (in_tile_y1 - in_tile_y0));
#endif
            }
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

        ncnn::VkMat out_gpu;
        if (opt.use_fp16_storage && opt.use_int8_storage) {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t) channels, 1,
                           opt.blob_vkallocator);
        } else {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t) 4u, 1,
                           opt.blob_vkallocator);
        }

        for (int xi = 0; xi < xtiles; xi++) {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3) {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4) {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            if (tta_mode) {
                // preproc
                ncnn::VkMat in_tile_gpu[8];
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    if (channels == 4) {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1,
                                                 in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu[0];
                    bindings[2] = in_tile_gpu[1];
                    bindings[3] = in_tile_gpu[2];
                    bindings[4] = in_tile_gpu[3];
                    bindings[5] = in_tile_gpu[4];
                    bindings[6] = in_tile_gpu[5];
                    bindings[7] = in_tile_gpu[6];
                    bindings[8] = in_tile_gpu[7];
                    bindings[9] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < 8; ti++) {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu[ti]);

                    for (size_t i = 0; i < names.size(); i++) {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    ex.extract("out0", out_tile_gpu[ti], cmd);
                }

                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4) {
                    if (scale == 1) {
                        out_alpha_tile_gpu = in_alpha_tile_gpu;
                    }
                    if (scale == 2) {
                        bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 3) {
                        bicubic_3x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 4) {
                        bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                }

                // postproc
                if (scale == 4) {
                    std::vector<ncnn::VkMat> bindings(11);
                    bindings[0] = in_gpu;
                    bindings[1] = out_tile_gpu[0];
                    bindings[2] = out_tile_gpu[1];
                    bindings[3] = out_tile_gpu[2];
                    bindings[4] = out_tile_gpu[3];
                    bindings[5] = out_tile_gpu[4];
                    bindings[6] = out_tile_gpu[5];
                    bindings[7] = out_tile_gpu[6];
                    bindings[8] = out_tile_gpu[7];
                    bindings[9] = out_alpha_tile_gpu;
                    bindings[10] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(16);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = out_tile_gpu[0].w;
                    constants[4].i = out_tile_gpu[0].h;
                    constants[5].i = out_tile_gpu[0].cstep;
                    constants[6].i = out_gpu.w;
                    constants[7].i = out_gpu.h;
                    constants[8].i = out_gpu.cstep;
                    constants[9].i = xi * TILE_SIZE_X;
                    constants[10].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[11].i = xi * TILE_SIZE_X * scale;
                    constants[12].i = std::min(TILE_SIZE_X * scale,
                                               out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[13].i = channels;
                    constants[14].i = out_alpha_tile_gpu.w;
                    constants[15].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale,
                                            out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_4x_postproc, bindings, constants, dispatcher);
                } else {
                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = out_tile_gpu[0];
                    bindings[1] = out_tile_gpu[1];
                    bindings[2] = out_tile_gpu[2];
                    bindings[3] = out_tile_gpu[3];
                    bindings[4] = out_tile_gpu[4];
                    bindings[5] = out_tile_gpu[5];
                    bindings[6] = out_tile_gpu[6];
                    bindings[7] = out_tile_gpu[7];
                    bindings[8] = out_alpha_tile_gpu;
                    bindings[9] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = out_tile_gpu[0].w;
                    constants[1].i = out_tile_gpu[0].h;
                    constants[2].i = out_tile_gpu[0].cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale,
                                              out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = channels;
                    constants[9].i = out_alpha_tile_gpu.w;
                    constants[10].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale,
                                            out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_postproc, bindings, constants, dispatcher);
                }
            } else {
                // preproc
                ncnn::VkMat in_tile_gpu;
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                       in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    if (channels == 4) {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1,
                                                 in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;
                    bindings[2] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu;
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu);

                    for (size_t i = 0; i < names.size(); i++) {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    ex.extract("out0", out_tile_gpu, cmd);
                }

                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4) {
                    if (scale == 1) {
                        out_alpha_tile_gpu = in_alpha_tile_gpu;
                    }
                    if (scale == 2) {
                        bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 3) {
                        bicubic_3x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 4) {
                        bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                }

                // postproc
                if (scale == 4) {
                    std::vector<ncnn::VkMat> bindings(4);
                    bindings[0] = in_gpu;
                    bindings[1] = out_tile_gpu;
                    bindings[2] = out_alpha_tile_gpu;
                    bindings[3] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(16);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = out_tile_gpu.w;
                    constants[4].i = out_tile_gpu.h;
                    constants[5].i = out_tile_gpu.cstep;
                    constants[6].i = out_gpu.w;
                    constants[7].i = out_gpu.h;
                    constants[8].i = out_gpu.cstep;
                    constants[9].i = xi * TILE_SIZE_X;
                    constants[10].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[11].i = xi * TILE_SIZE_X * scale;
                    constants[12].i = std::min(TILE_SIZE_X * scale,
                                               out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[13].i = channels;
                    constants[14].i = out_alpha_tile_gpu.w;
                    constants[15].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale,
                                            out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_4x_postproc, bindings, constants, dispatcher);
                } else {
                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = out_tile_gpu;
                    bindings[1] = out_alpha_tile_gpu;
                    bindings[2] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = out_tile_gpu.w;
                    constants[1].i = out_tile_gpu.h;
                    constants[2].i = out_tile_gpu.cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale,
                                              out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = channels;
                    constants[9].i = out_alpha_tile_gpu.w;
                    constants[10].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale,
                                            out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_postproc, bindings, constants, dispatcher);
                }
            }

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }

            report_progress(xtiles, ytiles, yi, xi, progress_listener);
        }

        // download
        {
            ncnn::Mat out;

            if (opt.use_fp16_storage && opt.use_int8_storage) {
                out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned char *) outimage.data +
                                                      yi * scale * TILE_SIZE_Y * w * scale *
                                                      channels, (size_t) channels, 1);
            }

            cmd.record_clone(out_gpu, out, opt);

            cmd.submit_and_wait();

            if (!(opt.use_fp16_storage && opt.use_int8_storage)) {
                if (channels == 3) {
#if _WIN32
                    out.to_pixels((unsigned char*)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, ncnn::Mat::PIXEL_RGB2BGR);
#else
                    out.to_pixels((unsigned char *) outimage.data +
                                  yi * scale * TILE_SIZE_Y * w * scale * channels,
                                  ncnn::Mat::PIXEL_RGB);
#endif
                }
                if (channels == 4) {
#if _WIN32
                    out.to_pixels((unsigned char*)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, ncnn::Mat::PIXEL_RGBA2BGRA);
#else
                    out.to_pixels((unsigned char *) outimage.data +
                                  yi * scale * TILE_SIZE_Y * w * scale * channels,
                                  ncnn::Mat::PIXEL_RGBA);
#endif
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_se_sync_gap(const ncnn::Mat &inimage, const std::vector<std::string> &names,
                                   const ncnn::Option &opt, FeatureCache &cache,
                                   const std::function<void(float)> &progress_listener) const {
    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    std::vector<std::vector<ncnn::VkMat> > feats(names.size());
    for (int yi = 0; yi < ytiles; yi++) {
        for (int xi = 0; xi < xtiles; xi++) {
            {
                for (size_t i = 0; i < names.size(); i++) {
                    if (tta_mode) {
                        for (int ti = 0; ti < 8; ti++) {
                            ncnn::VkMat feat;
                            cache.load(yi, xi, ti, names[i], feat);

                            feats[i].push_back(feat);
                        }
                    } else {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        feats[i].push_back(feat);
                    }
                }
            }
        }
    }

    const int tiles = ytiles * xtiles * (tta_mode ? 8 : 1);

    ncnn::VkCompute cmd(vkdev);

    // download
    std::vector<std::vector<ncnn::Mat> > feats_cpu(names.size());
    for (size_t i = 0; i < names.size(); i++) {
        feats_cpu[i].resize(tiles);

        for (int j = 0; j < tiles; j++) {
            cmd.record_download(feats[i][j], feats_cpu[i][j], opt);
        }
    }

    cmd.submit_and_wait();
    cmd.reset();

    // global average
    // upload
    std::vector<ncnn::VkMat> avgfeats(names.size());
    for (size_t i = 0; i < names.size(); i++) {
        for (int j = 0; j < tiles; j++) {
            if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() &&
                feats_cpu[i][j].elembits() == 16) {
                ncnn::Mat feat_fp32;
                ncnn::cast_float16_to_float32(feats_cpu[i][j], feat_fp32, opt);
                feats_cpu[i][j] = feat_fp32;
            }

            if (opt.use_packing_layout && feats_cpu[i][j].elempack != 1) {
                ncnn::Mat feat_cpu_unpacked;
                ncnn::convert_packing(feats_cpu[i][j], feat_cpu_unpacked, 1, opt);
                feats_cpu[i][j] = feat_cpu_unpacked;
            }
        }

        // handle feats_cpu[i] vector
        {
            ncnn::Mat avgfeat;
            avgfeat.create_like(feats_cpu[i][0]);
            avgfeat.fill(0.f);

            int len = avgfeat.total();

            for (int j = 0; j < tiles; j++) {
                const ncnn::Mat f = feats_cpu[i][j];

                for (int k = 0; k < len; k++) {
                    avgfeat[k] += f[k];
                }
            }

            for (int k = 0; k < len; k++) {
                avgfeat[k] /= tiles;
            }

            cmd.record_upload(avgfeat, avgfeats[i], opt);
        }
    }

    cmd.submit_and_wait();
    cmd.reset();


    for (int yi = 0; yi < ytiles; yi++) {
        for (int xi = 0; xi < xtiles; xi++) {
            {
                for (size_t i = 0; i < names.size(); i++) {
                    if (tta_mode) {
                        for (int ti = 0; ti < 8; ti++) {
                            cache.save(yi, xi, ti, names[i], avgfeats[i]);
                        }
                    } else {
                        cache.save(yi, xi, 0, names[i], avgfeats[i]);
                    }
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_se_very_rough_stage0(const ncnn::Mat &inimage,
                                            const std::vector<std::string> &names,
                                            const std::vector<std::string> &outnames,
                                            const ncnn::Option &opt, FeatureCache &cache,
                                            const std::function<void(
                                                    float)> &progress_listener) const {
    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = 32;
    const int TILE_SIZE_Y = 32;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi + 2 < ytiles; yi += 3) {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3) {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4) {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        ncnn::Mat in;
        if (opt.use_fp16_storage && opt.use_int8_storage) {
            in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0),
                           (unsigned char *) pixeldata + in_tile_y0 * w * channels,
                           (size_t) channels, 1);
        } else {
            if (channels == 3) {
#if _WIN32
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR2RGB, w, (in_tile_y1 - in_tile_y0));
#else
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels,
                                            ncnn::Mat::PIXEL_RGB, w, (in_tile_y1 - in_tile_y0));
#endif
            }
            if (channels == 4) {
#if _WIN32
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGRA2RGBA, w, (in_tile_y1 - in_tile_y0));
#else
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels,
                                            ncnn::Mat::PIXEL_RGBA, w, (in_tile_y1 - in_tile_y0));
#endif
            }
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

        ncnn::VkMat out_gpu;
        if (opt.use_fp16_storage && opt.use_int8_storage) {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t) channels, 1,
                           opt.blob_vkallocator);
        } else {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t) 4u, 1,
                           opt.blob_vkallocator);
        }

        for (int xi = 0; xi + 2 < xtiles; xi += 3) {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3) {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4) {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            if (tta_mode) {
                // preproc
                ncnn::VkMat in_tile_gpu[8];
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                          in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    if (channels == 4) {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1,
                                                 in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu[0];
                    bindings[2] = in_tile_gpu[1];
                    bindings[3] = in_tile_gpu[2];
                    bindings[4] = in_tile_gpu[3];
                    bindings[5] = in_tile_gpu[4];
                    bindings[6] = in_tile_gpu[5];
                    bindings[7] = in_tile_gpu[6];
                    bindings[8] = in_tile_gpu[7];
                    bindings[9] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < 8; ti++) {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu[ti]);

                    for (size_t i = 0; i < names.size(); i++) {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++) {
                        ncnn::VkMat feat;
                        ex.extract(outnames[i].c_str(), feat, cmd);

                        cache.save(yi, xi, ti, outnames[i], feat);
                    }
                }
            } else {
                // preproc
                ncnn::VkMat in_tile_gpu;
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                       in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    if (channels == 4) {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1,
                                                 in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;
                    bindings[2] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu);

                    for (size_t i = 0; i < names.size(); i++) {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++) {
                        ncnn::VkMat feat;
                        ex.extract(outnames[i].c_str(), feat, cmd);

                        cache.save(yi, xi, 0, outnames[i], feat);
                    }
                }
            }

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        cmd.submit_and_wait();
        cmd.reset();
    }

    return 0;
}

int RealCUGAN::process_se_very_rough_sync_gap(const ncnn::Mat &inimage,
                                              const std::vector<std::string> &names,
                                              const ncnn::Option &opt, FeatureCache &cache,
                                              const std::function<void(
                                                      float)> &progress_listener) const {
    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = 32;
    const int TILE_SIZE_Y = 32;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    std::vector<std::vector<ncnn::VkMat> > feats(names.size());
    for (int yi = 0; yi + 2 < ytiles; yi += 3) {
        for (int xi = 0; xi + 2 < xtiles; xi += 3) {
            {
                for (size_t i = 0; i < names.size(); i++) {
                    if (tta_mode) {
                        for (int ti = 0; ti < 8; ti++) {
                            ncnn::VkMat feat;
                            cache.load(yi, xi, ti, names[i], feat);

                            feats[i].push_back(feat);
                        }
                    } else {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        feats[i].push_back(feat);
                    }
                }
            }
        }
    }

    const int tiles = (ytiles / 3) * (xtiles / 3) * (tta_mode ? 8 : 1);

    ncnn::VkCompute cmd(vkdev);

    // download
    std::vector<std::vector<ncnn::Mat> > feats_cpu(names.size());
    for (size_t i = 0; i < names.size(); i++) {
        feats_cpu[i].resize(tiles);

        for (int j = 0; j < tiles; j++) {
            cmd.record_download(feats[i][j], feats_cpu[i][j], opt);
        }
    }

    cmd.submit_and_wait();
    cmd.reset();

    // global average
    // upload
    std::vector<ncnn::VkMat> avgfeats(names.size());
    for (size_t i = 0; i < names.size(); i++) {
        for (int j = 0; j < tiles; j++) {
            if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() &&
                feats_cpu[i][j].elembits() == 16) {
                ncnn::Mat feat_fp32;
                ncnn::cast_float16_to_float32(feats_cpu[i][j], feat_fp32, opt);
                feats_cpu[i][j] = feat_fp32;
            }

            if (opt.use_packing_layout && feats_cpu[i][j].elempack != 1) {
                ncnn::Mat feat_cpu_unpacked;
                ncnn::convert_packing(feats_cpu[i][j], feat_cpu_unpacked, 1, opt);
                feats_cpu[i][j] = feat_cpu_unpacked;
            }
        }

        // handle feats_cpu[i] vector
        {
            ncnn::Mat avgfeat;
            avgfeat.create_like(feats_cpu[i][0]);
            avgfeat.fill(0.f);

            int len = avgfeat.total();

            for (int j = 0; j < tiles; j++) {
                const ncnn::Mat f = feats_cpu[i][j];

                for (int k = 0; k < len; k++) {
                    avgfeat[k] += f[k];
                }
            }

            for (int k = 0; k < len; k++) {
                avgfeat[k] /= tiles;
            }

            cmd.record_upload(avgfeat, avgfeats[i], opt);
        }
    }

    cmd.submit_and_wait();
    cmd.reset();


    for (int yi = 0; yi + 2 < ytiles; yi += 3) {
        for (int xi = 0; xi + 2 < xtiles; xi += 3) {
            {
                for (size_t i = 0; i < names.size(); i++) {
                    if (tta_mode) {
                        for (int ti = 0; ti < 8; ti++) {
                            cache.save(yi, xi, ti, names[i], avgfeats[i]);
                            cache.save(yi, xi + 1, ti, names[i], avgfeats[i]);
                            cache.save(yi, xi + 2, ti, names[i], avgfeats[i]);
                            cache.save(yi + 1, xi, ti, names[i], avgfeats[i]);
                            cache.save(yi + 1, xi + 1, ti, names[i], avgfeats[i]);
                            cache.save(yi + 1, xi + 2, ti, names[i], avgfeats[i]);
                            cache.save(yi + 2, xi, ti, names[i], avgfeats[i]);
                            cache.save(yi + 2, xi + 1, ti, names[i], avgfeats[i]);
                            cache.save(yi + 2, xi + 2, ti, names[i], avgfeats[i]);
                        }
                    } else {
                        cache.save(yi, xi, 0, names[i], avgfeats[i]);
                        cache.save(yi, xi + 1, 0, names[i], avgfeats[i]);
                        cache.save(yi, xi + 2, 0, names[i], avgfeats[i]);
                        cache.save(yi + 1, xi, 0, names[i], avgfeats[i]);
                        cache.save(yi + 1, xi + 1, 0, names[i], avgfeats[i]);
                        cache.save(yi + 1, xi + 2, 0, names[i], avgfeats[i]);
                        cache.save(yi + 2, xi, 0, names[i], avgfeats[i]);
                        cache.save(yi + 2, xi + 1, 0, names[i], avgfeats[i]);
                        cache.save(yi + 2, xi + 2, 0, names[i], avgfeats[i]);
                    }
                }
            }
        }
    }

    return 0;
}

int
RealCUGAN::process_cpu_se_stage0(const ncnn::Mat &inimage, const std::vector<std::string> &names,
                                 const std::vector<std::string> &outnames, FeatureCache &cache,
                                 const std::function<void(float)> &progress_listener) const {
    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    ncnn::Option opt = net.opt;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    for (int yi = 0; yi < ytiles; yi++) {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3) {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4) {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        for (int xi = 0; xi < xtiles; xi++) {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3) {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4) {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            int in_tile_x0 = std::max(xi * TILE_SIZE_X - prepadding, 0);
            int in_tile_x1 = std::min((xi + 1) * TILE_SIZE_X + prepadding_right, w);

            // crop tile
            ncnn::Mat in;
            {
                if (channels == 3) {
#if _WIN32
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_BGR2RGB, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
#else
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_RGB, w, h,
                                                    in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0,
                                                    in_tile_y1 - in_tile_y0);
#endif
                }
                if (channels == 4) {
#if _WIN32
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_BGRA2RGBA, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
#else
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_RGBA, w, h,
                                                    in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0,
                                                    in_tile_y1 - in_tile_y0);
#endif
                }
            }

            ncnn::Mat out;

            if (tta_mode) {
                // split alpha and preproc
                ncnn::Mat in_tile[8];
                ncnn::Mat in_alpha_tile;
                ncnn::Mat in_alpha_tile_nocrop;

                {
                    in_tile[0].create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++) {
                        const float *ptr = in.channel(q);
                        float *outptr0 = in_tile[0].channel(q);

                        for (int i = 0; i < in.h; i++) {
                            for (int j = 0; j < in.w; j++) {
                                *outptr0++ = *ptr++ * (1 / 255.f);
                            }
                        }
                    }

                    if (channels == 4) {
                        in_alpha_tile = in.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(
                            std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h,
                                     prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w,
                                                      prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile[0], in_tile_padded, pad_top, pad_bottom,
                                           pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile[0] = in_tile_padded;

                    // alpha tile cut to nopad
                    if (channels == 4) {
                        ncnn::copy_cut_border(in_alpha_tile,
                                              in_alpha_tile_nocrop,
                                              pad_top, pad_bottom, pad_left, pad_right,
                                              opt);
                        // ensure alpha tile matches nopad size to avoid OOB
                        if (in_alpha_tile_nocrop.w != tile_w_nopad ||
                            in_alpha_tile_nocrop.h != tile_h_nopad) {
                            int extra_w = in_alpha_tile_nocrop.w - tile_w_nopad;
                            int extra_h = in_alpha_tile_nocrop.h - tile_h_nopad;
                            int fix_left = extra_w > 0 ? extra_w / 2 : 0;
                            int fix_right = extra_w > 0 ? extra_w - fix_left : 0;
                            int fix_top = extra_h > 0 ? extra_h / 2 : 0;
                            int fix_bottom = extra_h > 0 ? extra_h - fix_top : 0;
                            if (fix_left || fix_right || fix_top || fix_bottom) {
                                ncnn::Mat _alpha_fix;
                                ncnn::copy_cut_border(in_alpha_tile_nocrop, _alpha_fix,
                                                      fix_top, fix_bottom, fix_left, fix_right,
                                                      opt);
                                in_alpha_tile_nocrop = _alpha_fix;
                            }
                        }
                    }
                }

                // the other 7 directions
                {
                    in_tile[1].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[2].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[3].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[4].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[5].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[6].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[7].create(in_tile[0].h, in_tile[0].w, 3);

                    for (int q = 0; q < 3; q++) {
                        const ncnn::Mat in_tile_0 = in_tile[0].channel(q);
                        ncnn::Mat in_tile_1 = in_tile[1].channel(q);
                        ncnn::Mat in_tile_2 = in_tile[2].channel(q);
                        ncnn::Mat in_tile_3 = in_tile[3].channel(q);
                        ncnn::Mat in_tile_4 = in_tile[4].channel(q);
                        ncnn::Mat in_tile_5 = in_tile[5].channel(q);
                        ncnn::Mat in_tile_6 = in_tile[6].channel(q);
                        ncnn::Mat in_tile_7 = in_tile[7].channel(q);

                        for (int i = 0; i < in_tile[0].h; i++) {
                            const float *outptr0 = in_tile_0.row(i);
                            float *outptr1 = in_tile_1.row(in_tile[0].h - 1 - i);
                            float *outptr2 = in_tile_2.row(i) + in_tile[0].w - 1;
                            float *outptr3 = in_tile_3.row(in_tile[0].h - 1 - i) + in_tile[0].w - 1;

                            for (int j = 0; j < in_tile[0].w; j++) {
                                float *outptr4 = in_tile_4.row(j) + i;
                                float *outptr5 = in_tile_5.row(in_tile[0].w - 1 - j) + i;
                                float *outptr6 = in_tile_6.row(j) + in_tile[0].h - 1 - i;
                                float *outptr7 =
                                        in_tile_7.row(in_tile[0].w - 1 - j) + in_tile[0].h - 1 - i;

                                float v = *outptr0++;

                                *outptr1++ = v;
                                *outptr2-- = v;
                                *outptr3-- = v;
                                *outptr4 = v;
                                *outptr5 = v;
                                *outptr6 = v;
                                *outptr7 = v;
                            }
                        }
                    }
                }

                // realcugan
                ncnn::Mat out_tile[8];
                for (int ti = 0; ti < 8; ti++) {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile[ti]);

                    for (size_t i = 0; i < names.size(); i++) {
                        ncnn::Mat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++) {
                        ncnn::Mat feat;
                        ex.extract(outnames[i].c_str(), feat);

                        cache.save(yi, xi, ti, outnames[i], feat);
                    }
                }
            } else {
                // split alpha and preproc
                ncnn::Mat in_tile;
                ncnn::Mat in_alpha_tile;
                ncnn::Mat in_alpha_tile_nocrop;
                {
                    in_tile.create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++) {
                        const float *ptr = in.channel(q);
                        float *outptr = in_tile.channel(q);

                        for (int i = 0; i < in.w * in.h; i++) {
                            *outptr++ = *ptr++ * (1 / 255.f);
                        }
                    }

                    if (channels == 4) {
                        in_alpha_tile = in.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(
                            std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h,
                                     prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w,
                                                      prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile, in_tile_padded, pad_top, pad_bottom, pad_left,
                                           pad_right, 2, 0.f, net.opt);
                    in_tile = in_tile_padded;

                    // alpha tile cut to nopad
                    if (channels == 4) {
                        ncnn::copy_cut_border(
                                in_alpha_tile,
                                in_alpha_tile_nocrop,
                                pad_top, pad_bottom, pad_left, pad_right,
                                opt);
                        // ensure alpha tile matches nopad size to avoid OOB
                        if (in_alpha_tile_nocrop.w != tile_w_nopad ||
                            in_alpha_tile_nocrop.h != tile_h_nopad) {
                            int extra_w = in_alpha_tile_nocrop.w - tile_w_nopad;
                            int extra_h = in_alpha_tile_nocrop.h - tile_h_nopad;
                            int fix_left = extra_w > 0 ? extra_w / 2 : 0;
                            int fix_right = extra_w > 0 ? extra_w - fix_left : 0;
                            int fix_top = extra_h > 0 ? extra_h / 2 : 0;
                            int fix_bottom = extra_h > 0 ? extra_h - fix_top : 0;
                            if (fix_left || fix_right || fix_top || fix_bottom) {
                                ncnn::Mat _alpha_fix;
                                ncnn::copy_cut_border(in_alpha_tile_nocrop, _alpha_fix,
                                                      fix_top, fix_bottom, fix_left, fix_right,
                                                      opt);
                                in_alpha_tile_nocrop = _alpha_fix;
                            }
                        }
                    }
                }

                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile);

                    for (size_t i = 0; i < names.size(); i++) {
                        ncnn::Mat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++) {
                        ncnn::Mat feat;
                        ex.extract(outnames[i].c_str(), feat);

                        cache.save(yi, xi, 0, outnames[i], feat);
                    }
                }
            }
        }
    }

    return 0;
}

int
RealCUGAN::process_cpu_se_stage2(const ncnn::Mat &inimage, const std::vector<std::string> &names,
                                 ncnn::Mat &outimage, FeatureCache &cache,
                                 const std::function<void(float)> &progress_listener) const {
    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    ncnn::Option opt = net.opt;
    const size_t __out_total_bytes = (size_t) outimage.w * (size_t) outimage.h * (size_t) channels;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    for (int yi = 0; yi < ytiles; yi++) {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3) {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4) {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        for (int xi = 0; xi < xtiles; xi++) {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3) {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4) {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            int in_tile_x0 = std::max(xi * TILE_SIZE_X - prepadding, 0);
            int in_tile_x1 = std::min((xi + 1) * TILE_SIZE_X + prepadding_right, w);

            // crop tile
            ncnn::Mat in;
            {
                if (channels == 3) {
#if _WIN32
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_BGR2RGB, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
#else
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_RGB, w, h,
                                                    in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0,
                                                    in_tile_y1 - in_tile_y0);
#endif
                }
                if (channels == 4) {
#if _WIN32
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_BGRA2RGBA, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
#else
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_RGBA, w, h,
                                                    in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0,
                                                    in_tile_y1 - in_tile_y0);
#endif
                }
            }

            ncnn::Mat out;

            if (tta_mode) {
                // split alpha and preproc
                ncnn::Mat in_tile[8];
                ncnn::Mat in_alpha_tile;
                ncnn::Mat in_alpha_tile_nocrop;

                {
                    in_tile[0].create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++) {
                        const float *ptr = in.channel(q);
                        float *outptr0 = in_tile[0].channel(q);

                        for (int i = 0; i < in.h; i++) {
                            for (int j = 0; j < in.w; j++) {
                                *outptr0++ = *ptr++ * (1 / 255.f);
                            }
                        }
                    }

                    if (channels == 4) {
                        in_alpha_tile = in.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(
                            std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h,
                                     prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w,
                                                      prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile[0], in_tile_padded, pad_top, pad_bottom,
                                           pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile[0] = in_tile_padded;

                    // alpha tile cut to nopad
                    if (channels == 4) {
                        ncnn::copy_cut_border(in_alpha_tile,
                                              in_alpha_tile_nocrop,
                                              pad_top, pad_bottom, pad_left, pad_right,
                                              opt);
                        // ensure alpha tile matches nopad size to avoid OOB
                        if (in_alpha_tile_nocrop.w != tile_w_nopad ||
                            in_alpha_tile_nocrop.h != tile_h_nopad) {
                            int extra_w = in_alpha_tile_nocrop.w - tile_w_nopad;
                            int extra_h = in_alpha_tile_nocrop.h - tile_h_nopad;
                            int fix_left = extra_w > 0 ? extra_w / 2 : 0;
                            int fix_right = extra_w > 0 ? extra_w - fix_left : 0;
                            int fix_top = extra_h > 0 ? extra_h / 2 : 0;
                            int fix_bottom = extra_h > 0 ? extra_h - fix_top : 0;
                            if (fix_left || fix_right || fix_top || fix_bottom) {
                                ncnn::Mat _alpha_fix;
                                ncnn::copy_cut_border(in_alpha_tile_nocrop, _alpha_fix,
                                                      fix_top, fix_bottom, fix_left, fix_right,
                                                      opt);
                                in_alpha_tile_nocrop = _alpha_fix;
                            }
                        }
                    }
                }

                // the other 7 directions
                {
                    in_tile[1].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[2].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[3].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[4].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[5].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[6].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[7].create(in_tile[0].h, in_tile[0].w, 3);

                    for (int q = 0; q < 3; q++) {
                        const ncnn::Mat in_tile_0 = in_tile[0].channel(q);
                        ncnn::Mat in_tile_1 = in_tile[1].channel(q);
                        ncnn::Mat in_tile_2 = in_tile[2].channel(q);
                        ncnn::Mat in_tile_3 = in_tile[3].channel(q);
                        ncnn::Mat in_tile_4 = in_tile[4].channel(q);
                        ncnn::Mat in_tile_5 = in_tile[5].channel(q);
                        ncnn::Mat in_tile_6 = in_tile[6].channel(q);
                        ncnn::Mat in_tile_7 = in_tile[7].channel(q);

                        for (int i = 0; i < in_tile[0].h; i++) {
                            const float *outptr0 = in_tile_0.row(i);
                            float *outptr1 = in_tile_1.row(in_tile[0].h - 1 - i);
                            float *outptr2 = in_tile_2.row(i) + in_tile[0].w - 1;
                            float *outptr3 = in_tile_3.row(in_tile[0].h - 1 - i) + in_tile[0].w - 1;

                            for (int j = 0; j < in_tile[0].w; j++) {
                                float *outptr4 = in_tile_4.row(j) + i;
                                float *outptr5 = in_tile_5.row(in_tile[0].w - 1 - j) + i;
                                float *outptr6 = in_tile_6.row(j) + in_tile[0].h - 1 - i;
                                float *outptr7 =
                                        in_tile_7.row(in_tile[0].w - 1 - j) + in_tile[0].h - 1 - i;

                                float v = *outptr0++;

                                *outptr1++ = v;
                                *outptr2-- = v;
                                *outptr3-- = v;
                                *outptr4 = v;
                                *outptr5 = v;
                                *outptr6 = v;
                                *outptr7 = v;
                            }
                        }
                    }
                }

                // realcugan
                ncnn::Mat out_tile[8];
                for (int ti = 0; ti < 8; ti++) {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile[ti]);

                    for (size_t i = 0; i < names.size(); i++) {
                        ncnn::Mat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    ex.extract("out0", out_tile[ti]);
                }

                ncnn::Mat out_alpha_tile;
                if (channels == 4) {
                    if (scale == 1) {
                        out_alpha_tile = in_alpha_tile_nocrop;
                    }
                    if (scale == 2) {
                        bicubic_2x->forward(in_alpha_tile_nocrop, out_alpha_tile, opt);
                    }
                    if (scale == 3) {
                        bicubic_3x->forward(in_alpha_tile_nocrop, out_alpha_tile, opt);
                    }
                    if (scale == 4) {
                        bicubic_4x->forward(in_alpha_tile_nocrop, out_alpha_tile, opt);
                    }
                }

                // postproc and merge alpha
                {
                    out.create(tile_w_nopad * scale, tile_h_nopad * scale, channels);
                    if (scale == 4) {
                        for (int q = 0; q < 3; q++) {
                            const ncnn::Mat out_tile_0 = out_tile[0].channel(q);
                            const ncnn::Mat out_tile_1 = out_tile[1].channel(q);
                            const ncnn::Mat out_tile_2 = out_tile[2].channel(q);
                            const ncnn::Mat out_tile_3 = out_tile[3].channel(q);
                            const ncnn::Mat out_tile_4 = out_tile[4].channel(q);
                            const ncnn::Mat out_tile_5 = out_tile[5].channel(q);
                            const ncnn::Mat out_tile_6 = out_tile[6].channel(q);
                            const ncnn::Mat out_tile_7 = out_tile[7].channel(q);
                            float *outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++) {
                                const float *inptr =
                                        in_tile[0].channel(q).row(prepadding + i / 4) + prepadding;
                                const float *ptr0 = out_tile_0.row(i);
                                const float *ptr1 = out_tile_1.row(out_tile[0].h - 1 - i);
                                const float *ptr2 = out_tile_2.row(i) + out_tile[0].w - 1;
                                const float *ptr3 =
                                        out_tile_3.row(out_tile[0].h - 1 - i) + out_tile[0].w - 1;

                                for (int j = 0; j < out.w; j++) {
                                    const float *ptr4 = out_tile_4.row(j) + i;
                                    const float *ptr5 = out_tile_5.row(out_tile[0].w - 1 - j) + i;
                                    const float *ptr6 = out_tile_6.row(j) + out_tile[0].h - 1 - i;
                                    const float *ptr7 =
                                            out_tile_7.row(out_tile[0].w - 1 - j) + out_tile[0].h -
                                            1 - i;

                                    float v =
                                            (*ptr0++ + *ptr1++ + *ptr2-- + *ptr3-- + *ptr4 + *ptr5 +
                                             *ptr6 + *ptr7) / 8;

                                    *outptr++ = v * 255.f + 0.5f + inptr[j / 4] * 255.f;
                                }
                            }
                        }
                    } else {
                        for (int q = 0; q < 3; q++) {
                            const ncnn::Mat out_tile_0 = out_tile[0].channel(q);
                            const ncnn::Mat out_tile_1 = out_tile[1].channel(q);
                            const ncnn::Mat out_tile_2 = out_tile[2].channel(q);
                            const ncnn::Mat out_tile_3 = out_tile[3].channel(q);
                            const ncnn::Mat out_tile_4 = out_tile[4].channel(q);
                            const ncnn::Mat out_tile_5 = out_tile[5].channel(q);
                            const ncnn::Mat out_tile_6 = out_tile[6].channel(q);
                            const ncnn::Mat out_tile_7 = out_tile[7].channel(q);
                            float *outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++) {
                                const float *ptr0 = out_tile_0.row(i);
                                const float *ptr1 = out_tile_1.row(out_tile[0].h - 1 - i);
                                const float *ptr2 = out_tile_2.row(i) + out_tile[0].w - 1;
                                const float *ptr3 =
                                        out_tile_3.row(out_tile[0].h - 1 - i) + out_tile[0].w - 1;

                                for (int j = 0; j < out.w; j++) {
                                    const float *ptr4 = out_tile_4.row(j) + i;
                                    const float *ptr5 = out_tile_5.row(out_tile[0].w - 1 - j) + i;
                                    const float *ptr6 = out_tile_6.row(j) + out_tile[0].h - 1 - i;
                                    const float *ptr7 =
                                            out_tile_7.row(out_tile[0].w - 1 - j) + out_tile[0].h -
                                            1 - i;

                                    float v =
                                            (*ptr0++ + *ptr1++ + *ptr2-- + *ptr3-- + *ptr4 + *ptr5 +
                                             *ptr6 + *ptr7) / 8;

                                    *outptr++ = v * 255.f + 0.5f;
                                }
                            }
                        }
                    }

                    if (channels == 4) {
                        float *__alpha_dst = out.channel(3);
                        const float *__alpha_src = (const float *) out_alpha_tile.data;
                        const size_t __dst_elems = (size_t) out.w * (size_t) out.h;
                        const size_t __src_elems = (size_t) out_alpha_tile.total();
                        const size_t __copy_elems =
                                __src_elems < __dst_elems ? __src_elems : __dst_elems;
                        if (__src_elems != __dst_elems) {
                            LOGW("stage2(TTA) alpha size mismatch: dst=%zu src=%zu", __dst_elems,
                                 __src_elems);
                        }
                        if (__copy_elems > 0)
                            memcpy((void *) __alpha_dst, (const void *) __alpha_src,
                                   __copy_elems * sizeof(float));
                    }
                }
            } else {
                // split alpha and preproc
                ncnn::Mat in_tile;
                ncnn::Mat in_alpha_tile;
                ncnn::Mat in_alpha_tile_nocrop;
                {
                    in_tile.create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++) {
                        const float *ptr = in.channel(q);
                        float *outptr = in_tile.channel(q);

                        for (int i = 0; i < in.w * in.h; i++) {
                            *outptr++ = *ptr++ * (1 / 255.f);
                        }
                    }

                    if (channels == 4) {
                        in_alpha_tile = in.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(
                            std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h,
                                     prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w,
                                                      prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile, in_tile_padded, pad_top, pad_bottom, pad_left,
                                           pad_right, 2, 0.f, net.opt);
                    in_tile = in_tile_padded;

                    // alpha tile cut to nopad
                    if (channels == 4) {
                        ncnn::copy_cut_border(
                                in_alpha_tile,
                                in_alpha_tile_nocrop,
                                pad_top, pad_bottom, pad_left, pad_right,
                                opt);
                        // ensure alpha tile matches nopad size to avoid OOB
                        if (in_alpha_tile_nocrop.w != tile_w_nopad ||
                            in_alpha_tile_nocrop.h != tile_h_nopad) {
                            int extra_w = in_alpha_tile_nocrop.w - tile_w_nopad;
                            int extra_h = in_alpha_tile_nocrop.h - tile_h_nopad;
                            int fix_left = extra_w > 0 ? extra_w / 2 : 0;
                            int fix_right = extra_w > 0 ? extra_w - fix_left : 0;
                            int fix_top = extra_h > 0 ? extra_h / 2 : 0;
                            int fix_bottom = extra_h > 0 ? extra_h - fix_top : 0;
                            if (fix_left || fix_right || fix_top || fix_bottom) {
                                ncnn::Mat _alpha_fix;
                                ncnn::copy_cut_border(in_alpha_tile_nocrop, _alpha_fix,
                                                      fix_top, fix_bottom, fix_left, fix_right,
                                                      opt);
                                in_alpha_tile_nocrop = _alpha_fix;
                            }
                        }
                    }
                }

                // realcugan
                ncnn::Mat out_tile;
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile);

                    for (size_t i = 0; i < names.size(); i++) {
                        ncnn::Mat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    ex.extract("out0", out_tile);
                }

                ncnn::Mat out_alpha_tile;
                if (channels == 4) {
                    if (scale == 1) {
                        out_alpha_tile = in_alpha_tile_nocrop;
                    }
                    if (scale == 2) {
                        bicubic_2x->forward(in_alpha_tile_nocrop, out_alpha_tile, opt);
                    }
                    if (scale == 3) {
                        bicubic_3x->forward(in_alpha_tile_nocrop, out_alpha_tile, opt);
                    }
                    if (scale == 4) {
                        bicubic_4x->forward(in_alpha_tile_nocrop, out_alpha_tile, opt);
                    }
                }

                // postproc and merge alpha
                {
                    out.create(tile_w_nopad * scale, tile_h_nopad * scale, channels);
                    if (scale == 4) {
                        for (int q = 0; q < 3; q++) {
                            float *outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++) {
                                const float *inptr =
                                        in_tile.channel(q).row(prepadding + i / 4) + prepadding;
                                const float *ptr = out_tile.channel(q).row(i);

                                for (int j = 0; j < out.w; j++) {
                                    *outptr++ = *ptr++ * 255.f + 0.5f + inptr[j / 4] * 255.f;
                                }
                            }
                        }
                    } else {
                        for (int q = 0; q < 3; q++) {
                            float *outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++) {
                                const float *ptr = out_tile.channel(q).row(i);

                                for (int j = 0; j < out.w; j++) {
                                    *outptr++ = *ptr++ * 255.f + 0.5f;
                                }
                            }
                        }
                    }

                    if (channels == 4) {
                        float *__alpha_dst = out.channel(3);
                        const float *__alpha_src = (const float *) out_alpha_tile.data;
                        const size_t __dst_elems = (size_t) out.w * (size_t) out.h;
                        const size_t __src_elems = (size_t) out_alpha_tile.total();
                        const size_t __copy_elems =
                                __src_elems < __dst_elems ? __src_elems : __dst_elems;
                        if (__src_elems != __dst_elems) {
                            LOGW("stage2(TTA) alpha size mismatch: dst=%zu src=%zu", __dst_elems,
                                 __src_elems);
                        }
                        if (__copy_elems > 0)
                            memcpy((void *) __alpha_dst, (const void *) __alpha_src,
                                   __copy_elems * sizeof(float));
                    }
                }
            }
            {
                const size_t dst_stride = (size_t) w * (size_t) scale * (size_t) channels;
                unsigned char *__dst_base = (unsigned char *) outimage.data
                                            + (size_t) yi * (size_t) scale * (size_t) TILE_SIZE_Y *
                                              dst_stride
                                            + (size_t) xi * (size_t) scale * (size_t) TILE_SIZE_X *
                                              (size_t) channels;

                const size_t __need = (size_t) (out.h ? out.h - 1 : 0) * dst_stride +
                                      (size_t) out.w * (size_t) channels;
                const size_t __offset = (size_t) (__dst_base - (unsigned char *) outimage.data);

                if (__offset + __need > __out_total_bytes) {
                    LOGE("stage2: to_pixels OOB prevented (tile=%dx%d yi=%d xi=%d) off=%zu need=%zu buf=%zu; skip this tile",
                         out.w, out.h, yi, xi, __offset, __need, __out_total_bytes);
                    // 保守处理：不写回，避免崩溃。上层仍继续处理其它 tile。
                } else {
#if _WIN32
                    if (channels == 3)
                        out.to_pixels(__dst_base, ncnn::Mat::PIXEL_RGB2BGR, (int)dst_stride);
                    else
                        out.to_pixels(__dst_base, ncnn::Mat::PIXEL_RGBA2BGRA, (int)dst_stride);
#else
                    if (channels == 3)
                        out.to_pixels(__dst_base, ncnn::Mat::PIXEL_RGB, (int) dst_stride);
                    else
                        out.to_pixels(__dst_base, ncnn::Mat::PIXEL_RGBA, (int) dst_stride);
#endif
                }
            }
            report_progress(xtiles, ytiles, yi, xi, progress_listener);
        }
    }

    return 0;
}

int
RealCUGAN::process_cpu_se_sync_gap(const ncnn::Mat &inimage, const std::vector<std::string> &names,
                                   FeatureCache &cache,
                                   const std::function<void(float)> &progress_listener) const {
    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    ncnn::Option opt = net.opt;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    std::vector<std::vector<ncnn::Mat> > feats(names.size());
    for (int yi = 0; yi < ytiles; yi++) {
        for (int xi = 0; xi < xtiles; xi++) {
            {
                for (size_t i = 0; i < names.size(); i++) {
                    if (tta_mode) {
                        for (int ti = 0; ti < 8; ti++) {
                            ncnn::Mat feat;
                            cache.load(yi, xi, ti, names[i], feat);

                            feats[i].push_back(feat);
                        }
                    } else {
                        ncnn::Mat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        feats[i].push_back(feat);
                    }
                }
            }
        }
    }

    const int tiles = ytiles * xtiles * (tta_mode ? 8 : 1);

    // global average
    std::vector<ncnn::Mat> avgfeats(names.size());
    for (size_t i = 0; i < names.size(); i++) {
        // handle feats[i] vector
        {
            ncnn::Mat avgfeat;
            avgfeat.create_like(feats[i][0]);
            avgfeat.fill(0.f);

            int len = avgfeat.total();

            for (int j = 0; j < tiles; j++) {
                const ncnn::Mat f = feats[i][j];

                for (int k = 0; k < len; k++) {
                    avgfeat[k] += f[k];
                }
            }

            for (int k = 0; k < len; k++) {
                avgfeat[k] /= tiles;
            }

            avgfeats[i] = avgfeat;
        }
    }

    for (int yi = 0; yi < ytiles; yi++) {
        for (int xi = 0; xi < xtiles; xi++) {
            {
                for (size_t i = 0; i < names.size(); i++) {
                    if (tta_mode) {
                        for (int ti = 0; ti < 8; ti++) {
                            cache.save(yi, xi, ti, names[i], avgfeats[i]);
                        }
                    } else {
                        cache.save(yi, xi, 0, names[i], avgfeats[i]);
                    }
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_cpu_se_very_rough_stage0(const ncnn::Mat &inimage,
                                                const std::vector<std::string> &names,
                                                const std::vector<std::string> &outnames,
                                                FeatureCache &cache, const std::function<void(
        float)> &progress_listener) const {
    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = 32;
    const int TILE_SIZE_Y = 32;

    ncnn::Option opt = net.opt;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    for (int yi = 0; yi + 2 < ytiles; yi += 3) {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3) {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4) {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        for (int xi = 0; xi + 2 < xtiles; xi += 3) {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3) {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4) {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            int in_tile_x0 = std::max(xi * TILE_SIZE_X - prepadding, 0);
            int in_tile_x1 = std::min((xi + 1) * TILE_SIZE_X + prepadding_right, w);

            // crop tile
            ncnn::Mat in;
            {
                if (channels == 3) {
#if _WIN32
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_BGR2RGB, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
#else
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_RGB, w, h,
                                                    in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0,
                                                    in_tile_y1 - in_tile_y0);
#endif
                }
                if (channels == 4) {
#if _WIN32
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_BGRA2RGBA, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
#else
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_RGBA, w, h,
                                                    in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0,
                                                    in_tile_y1 - in_tile_y0);
#endif
                }
            }

            ncnn::Mat out;

            if (tta_mode) {
                // split alpha and preproc
                ncnn::Mat in_tile[8];
                ncnn::Mat in_alpha_tile;
                ncnn::Mat in_alpha_tile_nocrop;

                {
                    in_tile[0].create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++) {
                        const float *ptr = in.channel(q);
                        float *outptr0 = in_tile[0].channel(q);

                        for (int i = 0; i < in.h; i++) {
                            for (int j = 0; j < in.w; j++) {
                                *outptr0++ = *ptr++ * (1 / 255.f);
                            }
                        }
                    }

                    if (channels == 4) {
                        in_alpha_tile = in.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(
                            std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h,
                                     prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w,
                                                      prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile[0], in_tile_padded, pad_top, pad_bottom,
                                           pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile[0] = in_tile_padded;

                    // alpha tile cut to nopad
                    if (channels == 4) {
                        ncnn::copy_cut_border(in_alpha_tile,
                                              in_alpha_tile_nocrop,
                                              pad_top, pad_bottom, pad_left, pad_right,
                                              opt);
                        // ensure alpha tile matches nopad size to avoid OOB
                        if (in_alpha_tile_nocrop.w != tile_w_nopad ||
                            in_alpha_tile_nocrop.h != tile_h_nopad) {
                            int extra_w = in_alpha_tile_nocrop.w - tile_w_nopad;
                            int extra_h = in_alpha_tile_nocrop.h - tile_h_nopad;
                            int fix_left = extra_w > 0 ? extra_w / 2 : 0;
                            int fix_right = extra_w > 0 ? extra_w - fix_left : 0;
                            int fix_top = extra_h > 0 ? extra_h / 2 : 0;
                            int fix_bottom = extra_h > 0 ? extra_h - fix_top : 0;
                            if (fix_left || fix_right || fix_top || fix_bottom) {
                                ncnn::Mat _alpha_fix;
                                ncnn::copy_cut_border(in_alpha_tile_nocrop, _alpha_fix,
                                                      fix_top, fix_bottom, fix_left, fix_right,
                                                      opt);
                                in_alpha_tile_nocrop = _alpha_fix;
                            }
                        }
                    }
                }

                // the other 7 directions
                {
                    in_tile[1].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[2].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[3].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[4].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[5].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[6].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[7].create(in_tile[0].h, in_tile[0].w, 3);

                    for (int q = 0; q < 3; q++) {
                        const ncnn::Mat in_tile_0 = in_tile[0].channel(q);
                        ncnn::Mat in_tile_1 = in_tile[1].channel(q);
                        ncnn::Mat in_tile_2 = in_tile[2].channel(q);
                        ncnn::Mat in_tile_3 = in_tile[3].channel(q);
                        ncnn::Mat in_tile_4 = in_tile[4].channel(q);
                        ncnn::Mat in_tile_5 = in_tile[5].channel(q);
                        ncnn::Mat in_tile_6 = in_tile[6].channel(q);
                        ncnn::Mat in_tile_7 = in_tile[7].channel(q);

                        for (int i = 0; i < in_tile[0].h; i++) {
                            const float *outptr0 = in_tile_0.row(i);
                            float *outptr1 = in_tile_1.row(in_tile[0].h - 1 - i);
                            float *outptr2 = in_tile_2.row(i) + in_tile[0].w - 1;
                            float *outptr3 = in_tile_3.row(in_tile[0].h - 1 - i) + in_tile[0].w - 1;

                            for (int j = 0; j < in_tile[0].w; j++) {
                                float *outptr4 = in_tile_4.row(j) + i;
                                float *outptr5 = in_tile_5.row(in_tile[0].w - 1 - j) + i;
                                float *outptr6 = in_tile_6.row(j) + in_tile[0].h - 1 - i;
                                float *outptr7 =
                                        in_tile_7.row(in_tile[0].w - 1 - j) + in_tile[0].h - 1 - i;

                                float v = *outptr0++;

                                *outptr1++ = v;
                                *outptr2-- = v;
                                *outptr3-- = v;
                                *outptr4 = v;
                                *outptr5 = v;
                                *outptr6 = v;
                                *outptr7 = v;
                            }
                        }
                    }
                }

                // realcugan
                ncnn::Mat out_tile[8];
                for (int ti = 0; ti < 8; ti++) {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile[ti]);

                    for (size_t i = 0; i < names.size(); i++) {
                        ncnn::Mat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++) {
                        ncnn::Mat feat;
                        ex.extract(outnames[i].c_str(), feat);

                        cache.save(yi, xi, ti, outnames[i], feat);
                    }
                }
            } else {
                // split alpha and preproc
                ncnn::Mat in_tile;
                ncnn::Mat in_alpha_tile;
                ncnn::Mat in_alpha_tile_nocrop;
                {
                    in_tile.create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++) {
                        const float *ptr = in.channel(q);
                        float *outptr = in_tile.channel(q);

                        for (int i = 0; i < in.w * in.h; i++) {
                            *outptr++ = *ptr++ * (1 / 255.f);
                        }
                    }

                    if (channels == 4) {
                        in_alpha_tile = in.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(
                            std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h,
                                     prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w,
                                                      prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile, in_tile_padded, pad_top, pad_bottom, pad_left,
                                           pad_right, 2, 0.f, net.opt);
                    in_tile = in_tile_padded;

                    // alpha tile cut to nopad
                    if (channels == 4) {
                        ncnn::copy_cut_border(in_alpha_tile,
                                              in_alpha_tile_nocrop,
                                              pad_top, pad_bottom, pad_left, pad_right,
                                              opt);
                        // ensure alpha tile matches nopad size to avoid OOB
                        if (in_alpha_tile_nocrop.w != tile_w_nopad ||
                            in_alpha_tile_nocrop.h != tile_h_nopad) {
                            int extra_w = in_alpha_tile_nocrop.w - tile_w_nopad;
                            int extra_h = in_alpha_tile_nocrop.h - tile_h_nopad;
                            int fix_left = extra_w > 0 ? extra_w / 2 : 0;
                            int fix_right = extra_w > 0 ? extra_w - fix_left : 0;
                            int fix_top = extra_h > 0 ? extra_h / 2 : 0;
                            int fix_bottom = extra_h > 0 ? extra_h - fix_top : 0;
                            if (fix_left || fix_right || fix_top || fix_bottom) {
                                ncnn::Mat _alpha_fix;
                                ncnn::copy_cut_border(in_alpha_tile_nocrop, _alpha_fix,
                                                      fix_top, fix_bottom, fix_left, fix_right,
                                                      opt);
                                in_alpha_tile_nocrop = _alpha_fix;
                            }
                        }
                    }
                }

                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile);

                    for (size_t i = 0; i < names.size(); i++) {
                        ncnn::Mat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++) {
                        ncnn::Mat feat;
                        ex.extract(outnames[i].c_str(), feat);

                        cache.save(yi, xi, 0, outnames[i], feat);
                    }
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_cpu_se_very_rough_sync_gap(const ncnn::Mat &inimage,
                                                  const std::vector<std::string> &names,
                                                  FeatureCache &cache, const std::function<void(
        float)> progress_listener) const {
    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = 32;
    const int TILE_SIZE_Y = 32;

    ncnn::Option opt = net.opt;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    std::vector<std::vector<ncnn::Mat> > feats(names.size());
    for (int yi = 0; yi + 2 < ytiles; yi += 3) {
        for (int xi = 0; xi + 2 < xtiles; xi += 3) {
            {
                for (size_t i = 0; i < names.size(); i++) {
                    if (tta_mode) {
                        for (int ti = 0; ti < 8; ti++) {
                            ncnn::Mat feat;
                            cache.load(yi, xi, ti, names[i], feat);

                            feats[i].push_back(feat);
                        }
                    } else {
                        ncnn::Mat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        feats[i].push_back(feat);
                    }
                }
            }
        }
    }

    const int tiles = (ytiles / 3) * (xtiles / 3) * (tta_mode ? 8 : 1);

    // global average
    std::vector<ncnn::Mat> avgfeats(names.size());
    for (size_t i = 0; i < names.size(); i++) {
        // handle feats[i] vector
        {
            ncnn::Mat avgfeat;
            avgfeat.create_like(feats[i][0]);
            avgfeat.fill(0.f);

            int len = avgfeat.total();

            for (int j = 0; j < tiles; j++) {
                const ncnn::Mat f = feats[i][j];

                for (int k = 0; k < len; k++) {
                    avgfeat[k] += f[k];
                }
            }

            for (int k = 0; k < len; k++) {
                avgfeat[k] /= tiles;
            }

            avgfeats[i] = avgfeat;
        }
    }

    for (int yi = 0; yi + 2 < ytiles; yi += 3) {
        for (int xi = 0; xi + 2 < xtiles; xi += 3) {
            {
                for (size_t i = 0; i < names.size(); i++) {
                    if (tta_mode) {
                        for (int ti = 0; ti < 8; ti++) {
                            cache.save(yi, xi, ti, names[i], avgfeats[i]);
                            cache.save(yi, xi + 1, ti, names[i], avgfeats[i]);
                            cache.save(yi, xi + 2, ti, names[i], avgfeats[i]);
                            cache.save(yi + 1, xi, ti, names[i], avgfeats[i]);
                            cache.save(yi + 1, xi + 1, ti, names[i], avgfeats[i]);
                            cache.save(yi + 1, xi + 2, ti, names[i], avgfeats[i]);
                            cache.save(yi + 2, xi, ti, names[i], avgfeats[i]);
                            cache.save(yi + 2, xi + 1, ti, names[i], avgfeats[i]);
                            cache.save(yi + 2, xi + 2, ti, names[i], avgfeats[i]);
                        }
                    } else {
                        cache.save(yi, xi, 0, names[i], avgfeats[i]);
                        cache.save(yi, xi + 1, 0, names[i], avgfeats[i]);
                        cache.save(yi, xi + 2, 0, names[i], avgfeats[i]);
                        cache.save(yi + 1, xi, 0, names[i], avgfeats[i]);
                        cache.save(yi + 1, xi + 1, 0, names[i], avgfeats[i]);
                        cache.save(yi + 1, xi + 2, 0, names[i], avgfeats[i]);
                        cache.save(yi + 2, xi, 0, names[i], avgfeats[i]);
                        cache.save(yi + 2, xi + 1, 0, names[i], avgfeats[i]);
                        cache.save(yi + 2, xi + 2, 0, names[i], avgfeats[i]);
                    }
                }
            }
        }
    }

    return 0;
}

void RealCUGAN::report_progress(int xtiles, int ytiles, int yi, int xi,
                                const std::function<void(float)> &on_progress) {
    const int total_tiles = xtiles * ytiles;
    const int done = yi * xtiles + xi + 1;
    const float percent = (static_cast<float>(done) * 100.f) / (float) total_tiles;
    LOGI("processing... %.2f%%\n", percent);
    if (on_progress) {
        try {
            on_progress(percent);
        } catch (const std::exception &e) {
            LOGE("native onProgressListener threw exception: %s", e.what());
        } catch (...) {
            LOGE("native onProgressListener threw unknown exception");
        }
    }
}
