# RealCUGAN Android SDK

RealCUGAN Android SDK 是一个面向 Android 的 RealCUGAN 图片超分 SDK 与示例应用。它基于 ncnn + Vulkan 的本地推理后端，提供 Kotlin 友好的调用接口、明确的资源释放流程，以及适合长任务的前台 Service 封装。

底层推理流程基于 nihui 的 Real-CUGAN ncnn/Vulkan 实现，并针对 Android 设备的内存、显存和后台任务场景做了封装。

## 功能特性

- 基于 ncnn + Vulkan 运行 RealCUGAN 超分，不依赖 CUDA 或 PyTorch 运行时。
- 面向移动端设备调整默认配置，适配 CPU 与 Qualcomm / Adreno GPU 场景。
- 对 `noise`、`scale`、`syncgap`、`gpuId` 等参数做范围校验。
- 自动调整 tile 大小，降低部分设备输出黑图的概率。
- 提供前台 Service 封装，可在后台处理任务并显示进度通知。
- 提供 Binder API：`process()`、`configureConcurrency()`、`dispose()`。
- 通过 `release()` / `dispose()` 显式释放 native 与 Vulkan 资源。

## 安装

添加 JitPack 仓库：

```groovy
allprojects {
    repositories {
        google()
        mavenCentral()
        maven { url 'https://jitpack.io' }
    }
}
```

添加依赖：

```groovy
dependencies {
    implementation 'io.github.aoihoshino:realcugan-android-sdk:1.1'
}
```

## 基础用法

创建推理实例：

```kotlin
val opts = RealCUGANOption(
    context = this,
    noise = -1,
    scale = 2,
    syncgap = 3,
    modelName = ModelName.SE,
    ttaMode = false,
    gpuId = 0
)

val engine = RealCUGAN.create(opts)
```

执行处理：

```kotlin
val output = engine.process(inputBytes) { percent ->
    // 进度回调
}

imageView.setImageBitmap(output)
```

使用结束后释放资源：

```kotlin
engine.release()
```

## 前台 Service

如果任务可能在后台持续运行，或者需要在通知栏显示进度，可以使用内置的前台 Service 封装。

启动 Service：

```kotlin
val intent = Intent(this, RealCUGANService::class.java)
    .putExtra(RealCUGANService.EXTRA_ENABLE_NOTIFICATION, true)
    .putExtra(RealCUGANService.EXTRA_MAX_CONCURRENT, 2)
    .putExtra(RealCUGANService.EXTRA_QUEUE_ENABLED, true)

ContextCompat.startForegroundService(this, intent)
```

绑定并提交任务：

```kotlin
bindService(intent, conn, Context.BIND_AUTO_CREATE)

binder.process(bytes, "image.png", listener) { result ->
    result.onSuccess { bitmap -> imageView.setImageBitmap(bitmap) }
    result.onFailure { error -> Log.e("RealCUGAN", "processing failed", error) }
}
```

回收 Service：

```kotlin
binder.dispose()
unbindService(conn)
stopService(intent)
```

## 注意事项

- 建议同一进程内尽量只保留一个推理实例，避免 GPU 显存竞争。
- 批量处理或压测结束后务必调用 `release()` 或 `dispose()`。
- 大批量处理前建议先在目标设备上测试内存和显存表现。
