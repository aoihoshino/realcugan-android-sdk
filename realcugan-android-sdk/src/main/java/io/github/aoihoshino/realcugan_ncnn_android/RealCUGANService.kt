package io.github.aoihoshino.realcugan_ncnn_android

import RealCUGANOption
import android.app.*
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.roundToInt

class RealCUGANService : Service() {

    companion object {
        const val CHANNEL_ID = "realcugan_fg"
        const val NOTIF_ID = 1001
        const val EXTRA_ENABLE_NOTIFICATION = "EXTRA_ENABLE_NOTIFICATION"
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var rcg: RealCUGAN? = null

    private val tasks = ConcurrentHashMap<Int, Job>()
    private var nextTaskId = 1

    // 通知相关
    private var notificationsEnabled: Boolean = true
    private var builder: NotificationCompat.Builder? = null
    private var currentTitle: String = "RealCUGAN 待命"
    private var currentText: String = "准备处理图像..."

    override fun onCreate() {
        super.onCreate()
        ensureChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        notificationsEnabled = intent?.getBooleanExtra(EXTRA_ENABLE_NOTIFICATION, true) ?: true
        if (notificationsEnabled) {
            startAsForeground(initial = true)
        } else {
            // 不启用前台：清掉可能已有的前台状态
            stopForeground(STOP_FOREGROUND_REMOVE)
        }
        return START_STICKY
    }

    override fun onDestroy() {
        tasks.values.forEach { it.cancel() }
        scope.cancel()
        rcg?.release()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder = LocalBinder()

    inner class LocalBinder : Binder() {
        suspend fun init(opt: RealCUGANOption) {
            if (rcg == null) rcg = RealCUGAN.create(opt)
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
            onDone: (Result<android.graphics.Bitmap>) -> Unit
        ): Int {
            val id = nextTaskId++
            val job = scope.launch {
                val inst = rcg ?: run {
                    onDone(Result.failure(IllegalStateException("Service not initialized")))
                    return@launch
                }

                // 通知：开始任务
                val title = displayName ?: "正在处理图片"
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
                    updateNotification(titleText = title, percent = 100f, active = false, done = true)
                } catch (t: Throwable) {
                    onDone(Result.failure(t))
                    // 失败：标记失败文案
                    if (notificationsEnabled) {
                        builder?.setContentTitle("$title · 失败")
                            ?.setContentText(t.message ?: "处理失败")
                            ?.setProgress(0, 0, false)
                        notifyNow()
                    }
                } finally {
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
                CHANNEL_ID, "RealCUGAN 处理", NotificationManager.IMPORTANCE_LOW
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
                .setSubText("RealCUGAN")
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
                .setSubText("RealCUGAN")
        }

        currentTitle = titleText
        currentText = if (done) "处理完成" else "处理中：$pInt%"

        builder!!
            .setContentTitle(currentTitle)
            .setContentText(currentText)
            .setSubText("RealCUGAN")
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