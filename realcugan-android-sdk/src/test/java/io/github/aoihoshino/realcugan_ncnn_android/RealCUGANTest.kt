package io.github.aoihoshino.realcugan_ncnn_android

import ModelName
import RealCUGANOption
import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.mockito.Mockito.mock
import org.mockito.Mockito.`when`
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File

class ModelNameTest {

    @Test
    fun `from should return correct enum for valid dirs`() {
        assertEquals(ModelName.NOSE, ModelName.from("models-nose"))
        assertEquals(ModelName.PRO, ModelName.from("models-pro"))
        assertEquals(ModelName.SE, ModelName.from("models-se"))
    }

    @Test(expected = IllegalArgumentException::class)
    fun `from should throw for unknown dir`() {
        ModelName.from("models-xyz")
    }
}

class RealCUGANOptionTest {

    private val context: Context = mock(Context::class.java)

    @Test
    fun `default constructor yields valid option`() {
        val opts = RealCUGANOption(context)
        // 默认值：noise=-1, scale=2, syncgap=3, modelName=SE, ttaMode=false, gpuId=0
        assertEquals(-1, opts.noise)
        assertEquals(2, opts.scale)
        assertEquals(3, opts.syncgap)
        assertEquals(ModelName.SE, opts.modelName)
        assertFalse(opts.ttaMode)
        assertEquals(0, opts.gpuId)
    }

    // —— syncgap 边界测试 ——
    @Test
    fun `syncgap lower bound allowed`() {
        RealCUGANOption(context, syncgap = 0)
    }

    @Test
    fun `syncgap upper bound allowed`() {
        RealCUGANOption(context, syncgap = 3)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `syncgap below 0 should throw`() {
        RealCUGANOption(context, syncgap = -1)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `syncgap above 3 should throw`() {
        RealCUGANOption(context, syncgap = 4)
    }

    // —— gpuId 边界测试 ——
    @Test
    fun `gpuId -1 allowed (CPU only)`() {
        RealCUGANOption(context, gpuId = -1)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `gpuId below -1 should throw`() {
        RealCUGANOption(context, gpuId = -2)
    }

    // —— 针对 ModelName.NOSE 的 scale/noise 测试 ——
    @Test
    fun `ModelNameNOSE valid combo`() {
        RealCUGANOption(context, noise = 0, scale = 2, modelName = ModelName.NOSE)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `ModelNameNOSE invalid noise should throw`() {
        RealCUGANOption(context, noise = -1, scale = 2, modelName = ModelName.NOSE)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `ModelNameNOSE invalid scale should throw`() {
        RealCUGANOption(context, noise = 0, scale = 3, modelName = ModelName.NOSE)
    }

    // —— 针对 ModelName.PRO 的 scale/noise 测试 ——
    @Test
    fun `ModelNamePRO valid combos`() {
        RealCUGANOption(context, noise = -1, scale = 2, modelName = ModelName.PRO)
        RealCUGANOption(context, noise = 0, scale = 3, modelName = ModelName.PRO)
        RealCUGANOption(context, noise = 3, scale = 2, modelName = ModelName.PRO)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `ModelNamePRO invalid noise should throw`() {
        RealCUGANOption(context, noise = 1, scale = 2, modelName = ModelName.PRO)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `ModelNamePRO invalid scale should throw`() {
        RealCUGANOption(context, noise = 0, scale = 4, modelName = ModelName.PRO)
    }

    // —— 针对 ModelName.SE 的 scale/noise 测试 ——
    @Test
    fun `ModelNameSE valid combos`() {
        RealCUGANOption(context, noise = -1, scale = 2, modelName = ModelName.SE)
        RealCUGANOption(context, noise = 0, scale = 4, modelName = ModelName.SE)
        RealCUGANOption(context, noise = 3, scale = 3, modelName = ModelName.SE)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `ModelNameSE invalid noise should throw`() {
        RealCUGANOption(context, noise = 4, scale = 2, modelName = ModelName.SE)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `ModelNameSE invalid scale should throw`() {
        RealCUGANOption(context, noise = 0, scale = 5, modelName = ModelName.SE)
    }
}

class RealCUGANTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    private val modelStructure = mapOf(
        "models-nose" to listOf(
            "up2x-no-denoise.bin",
            "up2x-no-denoise.param"
        ),
        "models-pro" to listOf(
            "up2x-conservative.bin", "up2x-conservative.param",
            "up2x-denoise3x.bin", "up2x-denoise3x.param",
            "up2x-no-denoise.bin", "up2x-no-denoise.param",
            "up3x-conservative.bin", "up3x-conservative.param",
            "up3x-denoise3x.bin", "up3x-denoise3x.param",
            "up3x-no-denoise.bin", "up3x-no-denoise.param"
        ),
        "models-se" to listOf(
            "up2x-conservative.bin", "up2x-conservative.param",
            "up2x-denoise1x.bin", "up2x-denoise1x.param",
            "up2x-denoise2x.bin", "up2x-denoise2x.param",
            "up2x-denoise3x.bin", "up2x-denoise3x.param",
            "up2x-no-denoise.bin", "up2x-no-denoise.param",
            "up3x-conservative.bin", "up3x-conservative.param",
            "up3x-denoise3x.bin", "up3x-denoise3x.param",
            "up3x-no-denoise.bin", "up3x-no-denoise.param",
            "up4x-conservative.bin", "up4x-conservative.param",
            "up4x-denoise3x.bin", "up4x-denoise3x.param",
            "up4x-no-denoise.bin", "up4x-no-denoise.param"
        )
    )
}