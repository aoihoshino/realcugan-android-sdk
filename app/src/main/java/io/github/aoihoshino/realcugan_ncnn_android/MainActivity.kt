package io.github.aoihoshino.realcugan_ncnn_android

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.widget.ImageView
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File

class MainActivity : AppCompatActivity() {

    companion object {
        init {
            System.loadLibrary("realcugan_ncnn_android")
        }
    }

    external fun initialize(
        modelRoot: String,
        noise: Int? = null,
        scale: Int? = null,
        syncgap: Int? = null,
        modelName: String? = null,
        ttaMode: Boolean? = null
    ): Int

    external fun processImage(imageData: ByteArray, bitmapOut: Bitmap): Bitmap
    external fun release()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        unpackModelsToFilesDir()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                    data = Uri.parse("package:$packageName")
                }
                startActivity(intent)
            }
        }

        val imageView = findViewById<ImageView>(R.id.resultImageView)

        // 初始化
        val modelRoot = File(filesDir, "models").absolutePath
        if (initialize(modelRoot, scale = 2) != 0) {
            throw RuntimeException("RealCUGAN 初始化失败")
        }

        // 读原图 bytes
        val inputBytes = assets.open("test.png").use { input ->
            ByteArrayOutputStream().also { out ->
                input.copyTo(out)
            }.toByteArray()
        }

        // 准备输出 Bitmap
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(inputBytes, 0, inputBytes.size, opts)
        val w = opts.outWidth
        val h = opts.outHeight
        val outputBmp = Bitmap.createBitmap(w * 2, h * 2, Bitmap.Config.ARGB_8888)

        // ——重点：在后台线程调用 Native，再回到主线程更新 UI——
        lifecycleScope.launch(Dispatchers.Default) {
            // 这里跑 JNI，会锁 Bitmap 像素、写数据
            val resultBmp = processImage(inputBytes, outputBmp)

            // 切回主线程
            withContext(Dispatchers.Main) {
                imageView.setImageBitmap(resultBmp)
            }
        }
    }

    private fun unpackModelsToFilesDir() {
        val assetMgr = assets
        val destRoot = File(filesDir, "models")
        if (!destRoot.exists()) destRoot.mkdirs()
        assetMgr.list("models")?.forEach { modelName ->
            val srcDir = "models/$modelName"
            val dstDir = File(destRoot, modelName).apply { mkdirs() }
            assetMgr.list(srcDir)?.forEach { filename ->
                assetMgr.open("$srcDir/$filename").use { input ->
                    File(dstDir, filename).outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        release()
    }
}
