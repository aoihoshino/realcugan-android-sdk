# RealCUGAN Android SDK

RealCUGAN Android SDK is an Android library and sample app for running
RealCUGAN image upscaling locally on mobile devices. It wraps an ncnn/Vulkan
native backend with a Kotlin-friendly API, lifecycle-aware cleanup, and an
optional foreground Service for long-running jobs.

The native pipeline is based on the Real-CUGAN ncnn/Vulkan implementation by
nihui, adapted for Android deployment and mobile resource constraints.

## Features

- RealCUGAN upscaling through ncnn + Vulkan, without CUDA or PyTorch.
- Mobile-oriented defaults for CPU and Qualcomm/Adreno GPU devices.
- Strict option validation for `noise`, `scale`, `syncgap`, and `gpuId`.
- Automatic tile sizing to reduce device-specific black-output failures.
- Foreground Service wrapper with progress notification support.
- Binder API for `process()`, `configureConcurrency()`, and `dispose()`.
- Explicit native and Vulkan resource cleanup through `release()` / `dispose()`.

## Installation

Add JitPack to your repositories:

```groovy
allprojects {
    repositories {
        google()
        mavenCentral()
        maven { url 'https://jitpack.io' }
    }
}
```

Add the dependency:

```groovy
dependencies {
    implementation 'io.github.aoihoshino:realcugan-android-sdk:1.1'
}
```

## Basic Usage

Create an engine:

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

Run processing:

```kotlin
val output = engine.process(inputBytes) { percent ->
    // progress callback
}

imageView.setImageBitmap(output)
```

Release native resources when finished:

```kotlin
engine.release()
```

## Foreground Service

Use the foreground Service wrapper when processing may continue while the app is
in the background or when progress should be shown in a notification.

Start the service:

```kotlin
val intent = Intent(this, RealCUGANService::class.java)
    .putExtra(RealCUGANService.EXTRA_ENABLE_NOTIFICATION, true)
    .putExtra(RealCUGANService.EXTRA_MAX_CONCURRENT, 2)
    .putExtra(RealCUGANService.EXTRA_QUEUE_ENABLED, true)

ContextCompat.startForegroundService(this, intent)
```

Bind and submit work:

```kotlin
bindService(intent, conn, Context.BIND_AUTO_CREATE)

binder.process(bytes, "image.png", listener) { result ->
    result.onSuccess { bitmap -> imageView.setImageBitmap(bitmap) }
    result.onFailure { error -> Log.e("RealCUGAN", "processing failed", error) }
}
```

Clean up:

```kotlin
binder.dispose()
unbindService(conn)
stopService(intent)
```

## Notes

- Prefer a single engine instance per process to avoid GPU memory contention.
- Always call `release()` or `dispose()` after batch jobs or stress tests.
- Test memory behavior on target devices before processing large batches.
