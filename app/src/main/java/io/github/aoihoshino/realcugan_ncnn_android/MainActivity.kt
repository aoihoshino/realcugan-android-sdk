package io.github.aoihoshino.realcugan_ncnn_android

import RealCUGANOption
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.time.measureTime

class MainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "MainActivity"
    }

    // 待测 scale 列表（可以根据需要调）
    private val scales = listOf(2, 3, 4)
    private lateinit var realCUGANs: List<RealCUGAN>
    private val testFiles = listOf("test1.png", "test2.jpeg", "test3.png")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val imageView = findViewById<ImageView>(R.id.resultImageView)

        // 1) 根据 scales 创建多个实例
        realCUGANs = scales.map { scale ->
            RealCUGAN.create(
                RealCUGANOption(
                    context = this,
                    scale = scale
                ),
            )
        }

        // 2) 并发跑每个实例对应一个文件
        lifecycleScope.launch {
            // zip 保证文件名和实例一一对应
            val jobs = testFiles.zip(realCUGANs).map { (filename, cg) ->
                async(Dispatchers.IO) {
                    // IO 线程读取压缩数据
                    val bytes = assets.open(filename).use { input ->
                        input.readBytes()
                    }
                    // 调用 suspend 版 process（内部会切到 gpuDispatcher）
                    val bmp = cg.process(bytes)
                    filename to bmp
                }
            }

            // 等待并依次更新 UI
            val timeCost = measureTime {
            for (job in jobs) {
                    val (filename, bmp) = job.await()
                    // Main 线程更新界面
                    withContext(Dispatchers.Main) {
                        imageView.setImageBitmap(bmp)
                    }
                    Log.i(TAG, "Processed $filename → ${bmp.width}×${bmp.height}")
                }
            }
            Log.i(TAG, "Processed all jobs in $timeCost")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // 释放所有实例
        realCUGANs.forEach { it.release() }
    }
}
