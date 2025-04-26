package io.github.aoihoshino.realcugan_ncnn_android

import RealCUGANOption
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.core.graphics.createBitmap
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.Executors

class RealCUGAN private constructor(
    private val nativeHandle: Long,
    private val scaleFactor: Int
) {
    // 只有调用 nativeProcessImage 部分运行在这个 dispatcher 上
    private val gpuDispatcher: CoroutineDispatcher by lazy {
        Executors.newCachedThreadPool { Thread(it, "RealCUGAN-GPU") }
            .asCoroutineDispatcher()
    }

    /**
     * 对一段 PNG/JPEG/WebP 的字节做推理，返回 ARGB_8888 的 Bitmap。
     * 全流程：头部解码 → JNI 计算(切到 gpuDispatcher) → 拼装输出 Bitmap
     */
    suspend fun process(imageData: ByteArray): Bitmap = withContext(Dispatchers.IO) {
        // 1) 先解一次头，拿到原始尺寸
        val srcBmp = BitmapFactory.decodeByteArray(imageData, 0, imageData.size)
        val outW = srcBmp.width * scaleFactor
        val outH = srcBmp.height * scaleFactor

        // 2) 真正跑 native 推理，只在 gpuDispatcher 线程池
        val raw = withContext(gpuDispatcher) {
            nativeProcessImage(nativeHandle, imageData)
        }

        // 3) 拼装输出 Bitmap（IO 线程）
        val outBmp = createBitmap(outW, outH)
        when (raw.size) {
            // RGBA
            outW * outH * 4 -> {
                ByteBuffer
                    .wrap(raw)
                    .let { outBmp.copyPixelsFromBuffer(it) }
            }
            // RGB → 补 alpha
            outW * outH * 3 -> {
                val buf = ByteBuffer.allocate(outW * outH * 4)
                var i = 0
                repeat(outW * outH) {
                    buf.put(raw[i++])
                    buf.put(raw[i++])
                    buf.put(raw[i++])
                    buf.put(0xFF.toByte())
                }
                buf.rewind()
                outBmp.copyPixelsFromBuffer(buf)
            }

            else -> throw RuntimeException(
                "Unexpected pixel buffer size: ${raw.size}, expected ${outW * outH * 3} or ${outW * outH * 4}"
            )
        }
        Log.i("RealCUGAN", "process → Bitmap ready $outW×$outH")
        outBmp
    }

    fun release() {
        nativeRelease(nativeHandle)
    }

    companion object {
        init {
            System.loadLibrary("realcugan_ncnn_android")
        }

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
        private external fun nativeProcessImage(
            handle: Long,
            imageData: ByteArray
        ): ByteArray

        @JvmStatic
        private external fun nativeRelease(handle: Long)

        /**
         * 拷贝 assets/models → filesDir/models，然后 new 出一句柄
         */
        fun create(realCUGANOption: RealCUGANOption): RealCUGAN {
            val option = realCUGANOption
            val destRoot = File(option.context.filesDir, "models")
            if (!destRoot.exists()) {
                destRoot.mkdirs()
                option.context.assets.list("models")?.forEach { subdir ->
                    val dstSub = File(destRoot, subdir).apply { mkdirs() }
                    option.context.assets.list("models/$subdir")?.forEach { fname ->
                        option.context.assets.open("models/$subdir/$fname").use { inp ->
                            FileOutputStream(File(dstSub, fname)).use { out ->
                                inp.copyTo(out)
                            }
                        }
                    }
                }
            }
            val handle = nativeInitialize(
                destRoot.absolutePath,
                option.noise,
                option.scale,
                option.syncgap,
                option.modelName.dir,
                option.ttaMode,
                option.gpuId
            )
            require(handle >= 1L) { "RealCUGAN nativeInitialize failed: $handle" }
            return RealCUGAN(handle, option.scale)
        }
    }
}
