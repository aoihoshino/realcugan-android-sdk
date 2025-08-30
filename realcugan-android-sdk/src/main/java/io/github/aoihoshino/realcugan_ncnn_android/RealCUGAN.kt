package io.github.aoihoshino.realcugan_ncnn_android

import RealCUGANOption
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.annotation.Keep
import androidx.core.graphics.createBitmap
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

@Keep
fun interface ProgressListener {
    fun onProgress(percent: Float)
}

class RealCUGAN private constructor(
    private val nativeHandle: Long, private val scaleFactor: Int
) {
    // 只有调用 nativeProcessImage 部分运行在这个 dispatcher 上
    private val gpuDispatcher: CoroutineDispatcher by lazy {
        Executors.newCachedThreadPool {
            Thread(it, "RealCUGAN-GPU").apply {
                priority = Thread.MAX_PRIORITY
                isDaemon = true
            }
        }.asCoroutineDispatcher()
    }

    /**
     * 对一段 PNG/JPEG/WebP 的字节做推理，返回 ARGB_8888 的 Bitmap。
     * 全流程：头部解码 → JNI 计算(切到 gpuDispatcher) → 拼装输出 Bitmap
     */
    suspend fun process(
        imageData: ByteArray, onProgressListener: ProgressListener? = null
    ): Bitmap = withContext(Dispatchers.IO) {
// 1) 解码为 SOFTWARE + ARGB_8888 + 可写的 Bitmap（避免硬件位图导致无法锁像素）
        val inBmp = BitmapFactory.decodeByteArray(
            imageData, 0, imageData.size, BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.ARGB_8888
                inMutable = true
            }) ?: throw RuntimeException("decode bitmap failed")

        val outW = inBmp.width * scaleFactor
        val outH = inBmp.height * scaleFactor
        val outBmp = createBitmap(outW, outH, Bitmap.Config.ARGB_8888)

// 2) 真正跑 native 推理，只在 gpuDispatcher 线程池
        val ok = withContext(gpuDispatcher) {
            nativeProcessBitmap(nativeHandle, inBmp, outBmp, onProgressListener)
        }
        if (!ok) throw RuntimeException("nativeProcessBitmap failed")

        Log.i("RealCUGAN", "process(Bitmap path) → $outW×$outH ready")
        outBmp
    }

    fun release() {
        nativeRelease(nativeHandle)
    }

    companion object {
        @JvmStatic
        private external fun nativeInitialize(
            modelRoot: String?,
            noise: Int?,
            scale: Int?,
            syncgap: Int?,
            modelName: String?,
            ttaMode: Boolean?,
            gpuId: Int?
        ): Long

        @JvmStatic
        private external fun nativeProcessBitmap(
            handle: Long,
            inBitmap: Bitmap,
            outBitmap: Bitmap,
            progressListener: ProgressListener? = null
        ): Boolean

        @JvmStatic
        private external fun nativeRelease(handle: Long)

        /**
         * 创建一个RealCUGAN实例。
         * - 非必要请只创建一个实例
         * - 请不要创建太多实例，以免导致堆栈溢出
         *
         * @param realCUGANOption 创建 RealCUGAN 的配置
         * @see RealCUGANOption
         * 拷贝了 assets/models → filesDir/models
         */
        suspend fun create(realCUGANOption: RealCUGANOption): RealCUGAN =
            withContext(Dispatchers.Default) {
                System.loadLibrary("realcugan_ncnn_android")
                val destRoot = copyModels(context = realCUGANOption.context)
                val handle = nativeInitialize(
                    destRoot.absolutePath,
                    realCUGANOption.noise,
                    realCUGANOption.scale,
                    realCUGANOption.syncgap,
                    realCUGANOption.modelName.dir,
                    realCUGANOption.ttaMode,
                    realCUGANOption.gpuId,
                )
                require(handle >= 1L) { "RealCUGAN nativeInitialize failed: $handle" }
                return@withContext RealCUGAN(handle, realCUGANOption.scale)
            }

        internal fun copyModels(context: Context): File {
            val destRoot = File(context.filesDir, "models")
            if (!destRoot.exists()) {
                destRoot.mkdirs()
                context.assets.list("models")?.forEach { subdir ->
                    val dstSub = File(destRoot, subdir).apply { mkdirs() }
                    context.assets.list("models/$subdir")?.forEach { fname ->
                        context.assets.open("models/$subdir/$fname").use { inp ->
                            FileOutputStream(File(dstSub, fname)).use { out ->
                                inp.copyTo(out)
                            }
                        }
                    }
                }
            }
            return destRoot
        }
    }
}
