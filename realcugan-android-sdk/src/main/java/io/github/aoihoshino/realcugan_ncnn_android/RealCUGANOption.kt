import android.content.Context

/** 对应 assets/models 下的目录名，并维护了每个模型允许的 scale/noise */
enum class ModelName(val dir: String, val allowedScales: IntRange, val allowedNoises: Set<Int>) {
    NOSE("models-nose", 2..2, setOf(0)),
    PRO("models-pro", 2..3, setOf(-1, 0, 3)),
    SE("models-se", 2..4, setOf(-1, 0, 1, 2, 3));

    companion object {
        fun from(dir: String): ModelName =
            entries.find { it.dir == dir }
                ?: throw IllegalArgumentException("未知的 modelName: $dir")
    }
}

/**
 * 封装调用本地 RealCUGAN 初始化所需的各项参数，并在构造时进行严格校验。
 *
 * @param context
 *   Android 上下文，用于后续可能的资源加载或文件路径解析。不能为空。
 *
 * @param noise
 *   去噪等级。取值含义如下：
 *   - -1：保守模式（conservative）
 *   - 0 ：无去噪（no-denoise）
 *   - 1..3：去噪强度（denoise1x、denoise2x、denoise3x）
 *   边界校验：必须在 [modelName.allowedNoises] 中，否则抛 IllegalArgumentException。
 *
 * @param scale
 *   放大倍数，支持 2、3、4（对应 up2x、up3x、up4x）。
 *   边界校验：必须在 [modelName.allowedScales] 中，否则抛 IllegalArgumentException。
 *
 * @param syncgap
 *   同步间隔模式，同步特征融合级别：0..3。
 *   - 0：不使用 SE 同步（fastest）
 *   - 1..3：不同程度的同步，数字越大越“粗糙”。
 *   边界校验：必须在 0..3 之间，否则抛 IllegalArgumentException。
 *
 * @param modelName
 *   预定义模型名，枚举类型 [ModelName]，仅支持：
 *   - ModelName.NOSE  （models-nose）
 *   - ModelName.PRO   （models-pro）
 *   - ModelName.SE    （models-se）
 *   枚举中已维护各自允许的 scale/noise 范围，无效值会在构造时抛出 IllegalArgumentException。
 *
 * @param ttaMode
 *   是否启用 TTA（Test-Time Augmentation）模式。
 *   - false：关闭（默认）
 *   - true ：开启
 *
 * @param gpuId
 *   GPU 设备 ID：
 *   - -1：仅 CPU
 *   - >=0：对应 ncnn 可用的 GPU 索引
 *   边界校验：必须 >= -1，否则抛 IllegalArgumentException。
 *
 * 使用示例：
 * ```
 * // 双倍放大 + 保守去噪 + 序列化模型-se + 开启 TTA + 默认 GPU
 * val opts = RealCUGANOption(
 *   context = this,
 *   noise    = -1,
 *   scale    = 2,
 *   syncgap  = 3,
 *   modelName= ModelName.SE,
 *   ttaMode  = true,
 *   gpuId    = 0
 * )
 * ```
 */
class RealCUGANOption(
    val context: Context,
    val noise: Int = -1,
    val scale: Int = 2,
    val syncgap: Int = 3,
    val modelName: ModelName = ModelName.SE,
    val ttaMode: Boolean = false,
    val gpuId: Int = 0
) {

    init {
        require(syncgap in 0..3) { "syncgap 必须在 0..3 之间，但传入是 $syncgap" }
        require(gpuId >= -1) { "gpuId 必须 >= -1，但传入是 $gpuId" }

        require(scale in modelName.allowedScales) {
            "在 ${modelName.dir} 下，scale 必须在 ${modelName.allowedScales} 中，但传入的是 $scale"
        }
        require(noise in modelName.allowedNoises) {
            "在 ${modelName.dir} 下，noise 必须是 ${modelName.allowedNoises} 中的值，但传入的是 $noise"
        }
    }
}
