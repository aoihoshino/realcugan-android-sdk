package io.github.aoihoshino.realcugan_ncnn_android

import RealCUGANOption
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.graphics.Bitmap
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Semaphore
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.roundToInt

class RealCUGANService : Service() {

    companion object {
        const val CHANNEL_ID = "realcugan_fg"
        const val NOTIF_ID = 1001
        const val EXTRA_ENABLE_NOTIFICATION = "EXTRA_ENABLE_NOTIFICATION"
        const val EXTRA_MAX_CONCURRENT = "EXTRA_MAX_CONCURRENT"   // Int, default 1
        const val EXTRA_QUEUE_ENABLED = "EXTRA_QUEUE_ENABLED"    // Boolean, default true
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var rcg: RealCUGAN? = null

    private val tasks = ConcurrentHashMap<Int, Job>()
    private var nextTaskId = 1

    // 通知相关
    private var notificationsEnabled: Boolean = true
    private var builder: NotificationCompat.Builder? = null
    private var currentTitle: String = ""
    private var currentText: String = ""

    // 并发闸（Kotlin 侧）
    private var maxConcurrent: Int = 1
    private var queueEnabled: Boolean = true
    private var gate: Semaphore = Semaphore(maxConcurrent)

    override fun onCreate() {
        super.onCreate()
        ensureChannel()
        // 初始化通知标题/文案（改用资源）
        currentTitle = getString(R.string.notif_initial_title)
        currentText = getString(R.string.notif_initial_text)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        notificationsEnabled = intent?.getBooleanExtra(EXTRA_ENABLE_NOTIFICATION, true) ?: true
        if (notificationsEnabled) {
            startAsForeground(initial = true)
        } else {
            // 不启用前台：清掉可能已有的前台状态
            stopForeground(STOP_FOREGROUND_REMOVE)
        }
        // —— 并发闸配置 ——
        val newMax = (intent?.getIntExtra(EXTRA_MAX_CONCURRENT, maxConcurrent)
            ?: maxConcurrent).coerceAtLeast(1)
        val newQueue = intent?.getBooleanExtra(EXTRA_QUEUE_ENABLED, queueEnabled) ?: queueEnabled
        if (newMax != maxConcurrent || newQueue != queueEnabled) {
            maxConcurrent = newMax
            queueEnabled = newQueue
            gate = Semaphore(maxConcurrent)
        }
        return START_STICKY
    }

    override fun onDestroy() {
        tasks.values.forEach { it.cancel() }
        scope.cancel()
        rcg?.release()
        super.onDestroy()
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)

        when {
            // 应用进入后台：尽量降并发，必要时退出前台
            level == TRIM_MEMORY_UI_HIDDEN -> {
                maxConcurrent = 1
                gate = Semaphore(maxConcurrent)
                // 非必须：若无任务，退出前台
                if (tasks.isEmpty()) {
                    try {
                        stopForeground(STOP_FOREGROUND_REMOVE)
                    } catch (_: Throwable) {
                    }
                    (getSystemService(NOTIFICATION_SERVICE) as? NotificationManager)?.cancel(
                        NOTIF_ID
                    )
                }
            }

            // 系统要主动回收内存（后台/中度/完全）：优先释放
            level >= TRIM_MEMORY_BACKGROUND -> {
                if (tasks.isEmpty()) {
                    try {
                        rcg?.release()
                    } catch (_: Throwable) {
                    }
                    rcg = null
                    try {
                        stopForeground(STOP_FOREGROUND_REMOVE)
                    } catch (_: Throwable) {
                    }
                    (getSystemService(NOTIFICATION_SERVICE) as? NotificationManager)?.cancel(
                        NOTIF_ID
                    )
                    stopSelf()
                } else {
                    maxConcurrent = 1
                    gate = Semaphore(maxConcurrent)
                }
            }

            else -> {
                // 兼容旧版（已弃用）的 RUNNING_* 等级：仅降并发，不做重操作
                @Suppress("DEPRECATION")
                val runningLow = level == TRIM_MEMORY_RUNNING_LOW ||
                        level == TRIM_MEMORY_RUNNING_CRITICAL ||
                        level == TRIM_MEMORY_RUNNING_MODERATE
                if (runningLow) {
                    maxConcurrent = 1
                    gate = Semaphore(maxConcurrent)
                }
            }
        }
    }

    override fun onLowMemory() {
        super.onLowMemory()
        // 与 TRIM_* 一致：若空闲则尽量释放
        if (tasks.isEmpty()) {
            try {
                rcg?.release()
            } catch (_: Throwable) {
            }
            rcg = null
            try {
                stopForeground(STOP_FOREGROUND_REMOVE)
            } catch (_: Throwable) {
            }
            (getSystemService(NOTIFICATION_SERVICE) as? NotificationManager)?.cancel(NOTIF_ID)
            stopSelf()
        } else {
            maxConcurrent = 1
            gate = Semaphore(maxConcurrent)
        }
    }

    override fun onBind(intent: Intent?): IBinder = LocalBinder()

    inner class LocalBinder : Binder() {
        suspend fun init(opt: RealCUGANOption) {
            if (rcg == null) rcg = RealCUGAN.create(opt)
        }

        fun configureConcurrency(max: Int, queue: Boolean) {
            val safeMax = max.coerceAtLeast(1)
            maxConcurrent = safeMax
            queueEnabled = queue
            gate = Semaphore(safeMax)
        }

        fun concurrencyInfo(): Pair<Int, Boolean> = maxConcurrent to queueEnabled

        /**
         * 主动释放：可选取消正在运行的任务，并释放 native 实例与前台。
         */
        fun dispose(cancelRunning: Boolean = false) {
            // 不再接受新的并发
            maxConcurrent = 1
            gate = Semaphore(1)

            if (cancelRunning) {
                tasks.keys.toList().forEach { id ->
                    try {
                        tasks[id]?.cancel()
                    } catch (_: Throwable) {
                    }
                }
            }

            try {
                rcg?.release()
            } catch (_: Throwable) {
            }
            rcg = null

            try {
                stopForeground(STOP_FOREGROUND_REMOVE)
            } catch (_: Throwable) {
            }
            (getSystemService(NOTIFICATION_SERVICE) as? NotificationManager)?.cancel(NOTIF_ID)
            stopSelf()
        }

        /**
         * 提交一次任务
         * @param imageData 输入字节
         * @param displayName 用于通知栏显示的文件名（建议传原文件名），可以为 null
         * @param listener 额外的进度回调（可选）
         * @param onDone 结果回调
         */
        fun process(
            imageData: ByteArray,
            displayName: String? = null,
            listener: ProgressListener? = null,
            onDone: (Result<Bitmap>) -> Unit
        ): Int {
            val id = nextTaskId++
            val job = scope.launch {
                val inst = rcg ?: run {
                    onDone(Result.failure(IllegalStateException("Service not initialized")))
                    return@launch
                }

                if (queueEnabled) {
                    // 阻塞等待可用槽位
                    gate.acquire()
                } else {
                    // 不排队：满额即失败
                    // 不排队：满额即失败
                    if (!gate.tryAcquire()) {
                        onDone(Result.failure(IllegalStateException("Too many concurrent tasks")))
                        return@launch
                    }
                }

                // 通知：开始任务
                val title = displayName ?: getString(R.string.notif_title_processing_fallback)
                updateNotification(titleText = title, percent = 0f, active = true)

                // 复合进度回调：既更新通知，也转发给调用方
                val composite = ProgressListener { p ->
                    if (notificationsEnabled) {
                        updateNotification(titleText = title, percent = p, active = true)
                    }
                    listener?.onProgress(p)
                }

                try {
                    val bmp = inst.process(imageData, composite)
                    onDone(Result.success(bmp))
                    // 完成：进度 100% 并变更文案
                    updateNotification(
                        titleText = title,
                        percent = 100f,
                        active = false,
                        done = true
                    )
                } catch (t: Throwable) {
                    onDone(Result.failure(t))
                    // 失败：标记失败文案
                    if (notificationsEnabled) {
                        builder?.setContentTitle("$title · ${getString(R.string.notif_failed)}")
                            ?.setContentText(t.message ?: getString(R.string.notif_failed))
                            ?.setProgress(0, 0, false)
                        notifyNow()
                    }
                } finally {
                    // 释放并发槽位
                    try {
                        gate.release()
                    } catch (_: Throwable) {
                    }
                    tasks.remove(id)
                }
            }
            tasks[id] = job
            return id
        }

        fun cancel(taskId: Int) {
            tasks[taskId]?.cancel()
        }

        /** 运行中动态开/关通知 */
        fun setNotificationsEnabled(enabled: Boolean) {
            notificationsEnabled = enabled
            if (enabled) {
                startAsForeground(initial = false)
            } else {
                stopForeground(STOP_FOREGROUND_REMOVE)
                // 同时清掉状态通知
                (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).cancel(NOTIF_ID)
            }
        }
    }

    /** —— 通知相关 —— */

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT >= 26) {
            val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            val ch = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notif_channel_name),
                NotificationManager.IMPORTANCE_LOW
            )
            nm.createNotificationChannel(ch)
        }
    }

    private fun startAsForeground(initial: Boolean) {
        if (!notificationsEnabled) return
        if (builder == null) {
            builder = NotificationCompat.Builder(this, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setOnlyAlertOnce(true)
                .setOngoing(true)
                .setCategory(Notification.CATEGORY_SERVICE)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .setSubText(getString(R.string.notif_subtext))
        }
        builder!!
            .setContentTitle(currentTitle)
            .setContentText(currentText)
            .setProgress(0, 0, true)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setOngoing(true)

        if (initial) {
            startForeground(NOTIF_ID, builder!!.build())
        } else {
            notifyNow()
        }
    }

    private fun updateNotification(
        titleText: String,
        percent: Float,
        active: Boolean,
        done: Boolean = false
    ) {
        if (!notificationsEnabled) return
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        val pInt = percent.coerceIn(0f, 100f).roundToInt()

        if (builder == null) {
            builder = NotificationCompat.Builder(this, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setOnlyAlertOnce(true)
                .setOngoing(true)
                .setCategory(Notification.CATEGORY_SERVICE)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .setSubText(getString(R.string.notif_subtext))
        }

        currentTitle = titleText
        currentText =
            if (done) getString(R.string.notif_done) else getString(R.string.notif_processing, pInt)

        builder!!
            .setContentTitle(currentTitle)
            .setContentText(currentText)
            .setSubText(getString(R.string.notif_subtext))
            .setCategory(Notification.CATEGORY_SERVICE)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(active && !done)
            .setSmallIcon(if (done) android.R.drawable.stat_sys_download_done else android.R.drawable.stat_notify_sync)
            .setProgress(
                if (done) 0 else 100,
                if (done) 0 else pInt,
                false
            )

        // 如果此前没有前台，补齐为前台；否则仅刷新
        if (notificationsEnabled) {
            try {
                startForeground(NOTIF_ID, builder!!.build())
            } catch (_: IllegalStateException) {
                // 已经在前台，则只 notify 刷新
                nm.notify(NOTIF_ID, builder!!.build())
            }
        }
    }

    private fun notifyNow() {
        if (!notificationsEnabled) return
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIF_ID, builder!!.build())
    }
}