package com.collectorapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Base64
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.modules.core.DeviceEventManagerModule
import com.facebook.react.uimanager.SimpleViewManager
import com.facebook.react.uimanager.ThemedReactContext
import java.io.ByteArrayOutputStream
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object BounceSideCameraHolder {
    @Volatile var rootView: FrameLayout? = null

    @Volatile private var previewImageView: ImageView? = null
    private val displayLock = Any()
    private var lastDisplayBitmap: Bitmap? = null

    fun updateView(root: FrameLayout?, imageView: ImageView?) {
        val previousImageView = previewImageView
        rootView = root
        previewImageView = imageView
        if (previousImageView != null && previousImageView !== imageView) {
            previousImageView.post { previousImageView.setImageDrawable(null) }
        }
        if (imageView == null) {
            clearDisplayedFrame()
        }
    }

    fun displayFrame(bitmap: Bitmap): Boolean {
        val imageView = previewImageView ?: return false
        imageView.post {
            if (previewImageView === imageView) {
                val previous = synchronized(displayLock) {
                    val old = lastDisplayBitmap
                    lastDisplayBitmap = bitmap
                    old
                }
                imageView.setImageBitmap(bitmap)
                previous?.recycle()
            } else {
                bitmap.recycle()
            }
        }
        return true
    }

    fun clearDisplayedFrame() {
        val imageView = previewImageView
        val previous = synchronized(displayLock) {
            val old = lastDisplayBitmap
            lastDisplayBitmap = null
            old
        }
        if (imageView != null) {
            imageView.post {
                if (previewImageView === imageView) {
                    imageView.setImageDrawable(null)
                }
                previous?.recycle()
            }
        } else {
            previous?.recycle()
        }
    }
}

class BounceSideCameraViewManager : SimpleViewManager<FrameLayout>() {
    override fun getName() = "BounceSideCameraView"

    override fun createViewInstance(ctx: ThemedReactContext): FrameLayout {
        val root = FrameLayout(ctx)
        root.setBackgroundColor(Color.BLACK)
        root.clipChildren = true

        val imageView = ImageView(ctx)
        imageView.setBackgroundColor(Color.BLACK)
        imageView.scaleType = ImageView.ScaleType.CENTER_CROP
        imageView.layoutParams = FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT,
        )
        root.addView(imageView)

        BounceSideCameraHolder.updateView(root, imageView)
        return root
    }

    override fun onDropViewInstance(view: FrameLayout) {
        if (BounceSideCameraHolder.rootView === view) {
            BounceSideCameraHolder.updateView(null, null)
        }
        super.onDropViewInstance(view)
    }
}

class BounceSideLiveModule(private val ctx: ReactApplicationContext)
    : ReactContextBaseJavaModule(ctx) {

    override fun getName() = "BounceSideLive"

    private companion object {
        const val TRACK_EVENT_NAME = "onBounceSideRacketTrack"
        const val TRACK_INTERVAL_MS = 110L
        const val TRACK_HOLD_MS = 420L
        const val TRACK_MAX_BUFFERED = 36
        const val TRACK_TARGET_WIDTH = 240
        const val TRACK_COLOR_CONFIDENCE = 0.52
        const val TRACK_BLACK_CONFIDENCE = 0.66
    }

    @Volatile private var cameraProvider: ProcessCameraProvider? = null
    @Volatile private var useFrontCameraForPreview: Boolean = true
    @Volatile private var trackBusy: Boolean = false
    @Volatile private var lastTrackAnalyzeMs: Long = 0L

    private class FrameRec(
        val tsMs: Long,
        val nv21: ByteArray,
        val width: Int,
        val height: Int,
        val rotation: Int,
    )

    private data class RacketTrackRec(
        val tsMs: Long,
        val tracked: Boolean,
        val label: String,
        val color: String,
        val confidence: Double,
        val x: Double,
        val y: Double,
        val width: Double,
        val height: Double,
        val source: String,
        val redScore: Double,
        val darkScore: Double,
        val areaRatio: Double,
        val fillRatio: Double,
    )

    private data class ComponentCandidate(
        val color: String,
        val x: Double,
        val y: Double,
        val width: Double,
        val height: Double,
        val confidence: Double,
        val redScore: Double,
        val darkScore: Double,
        val areaRatio: Double,
        val fillRatio: Double,
    )

    private val frameLock = Any()
    private val frameBuffer = ArrayDeque<FrameRec>()
    private val maxBufferedFrames = 16

    private val trackLock = Any()
    private val trackBuffer = ArrayDeque<RacketTrackRec>()
    private var lastGoodTrack: RacketTrackRec? = null

    private var landmarker: com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker? = null

    private fun ensureLandmarker(): com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker {
        var current = landmarker
        if (current == null) {
            val options = com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(
                    com.google.mediapipe.tasks.core.BaseOptions.builder()
                        .setModelAssetPath("pose_landmarker_lite.task")
                        .build()
                )
                .setRunningMode(com.google.mediapipe.tasks.vision.core.RunningMode.IMAGE)
                .build()
            current = com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker.createFromOptions(ctx, options)
            landmarker = current
        }
        return current
    }

    @ReactMethod
    fun addListener(eventName: String) {
        // Required by NativeEventEmitter on Android.
    }

    @ReactMethod
    fun removeListeners(count: Int) {
        // Required by NativeEventEmitter on Android.
    }

    @ReactMethod
    fun startCamera(useFrontCamera: Boolean, promise: Promise) {
        val activity = ctx.currentActivity
        if (activity !is LifecycleOwner) {
            promise.reject("NO_ACTIVITY", "Ingen aktiv activity for kamerabindning")
            return
        }
        useFrontCameraForPreview = useFrontCamera
        synchronized(frameLock) { frameBuffer.clear() }
        synchronized(trackLock) {
            trackBuffer.clear()
            lastGoodTrack = null
        }
        lastTrackAnalyzeMs = 0L

        val mainExecutor = ContextCompat.getMainExecutor(ctx)
        val providerFuture = ProcessCameraProvider.getInstance(ctx)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                cameraProvider = provider
                provider.unbindAll()

                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(720, 1280))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(ContextCompat.getMainExecutor(ctx)) { proxy ->
                    try {
                        val nv21 = imageProxyToNv21(proxy)
                        val record = FrameRec(
                            System.currentTimeMillis(),
                            nv21,
                            proxy.width,
                            proxy.height,
                            proxy.imageInfo.rotationDegrees,
                        )
                        synchronized(frameLock) {
                            frameBuffer.addLast(record)
                            while (frameBuffer.size > maxBufferedFrames) frameBuffer.removeFirst()
                        }
                        maybeAnalyzeRacketTrack(record)
                    } finally {
                        proxy.close()
                    }
                }

                val selector = if (useFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA
                               else CameraSelector.DEFAULT_BACK_CAMERA
                provider.bindToLifecycle(activity, selector, analysis)
                promise.resolve("started")
            } catch (error: Exception) {
                promise.reject("CAMERA_START_ERROR", error.message, error)
            }
        }, mainExecutor)
    }

    @ReactMethod
    fun stopCamera(promise: Promise) {
        val provider = cameraProvider
        if (provider != null) {
            ContextCompat.getMainExecutor(ctx).execute {
                try { provider.unbindAll() } catch (_: Exception) {}
            }
        }
        BounceSideCameraHolder.clearDisplayedFrame()
        synchronized(frameLock) { frameBuffer.clear() }
        synchronized(trackLock) {
            trackBuffer.clear()
            lastGoodTrack = null
        }
        trackBusy = false
        promise.resolve("stopped")
    }

    @ReactMethod
    fun getRacketTrack(targetTimeMs: Double, promise: Promise) {
        val target = targetTimeMs.toLong()
        val best = synchronized(trackLock) {
            trackBuffer.minByOrNull { abs(it.tsMs - target) }
        }
        val result = if (best != null && abs(best.tsMs - target) <= 650L) {
            best
        } else {
            lostTrack(target)
        }
        promise.resolve(trackToMap(result, System.currentTimeMillis(), target))
    }

    @ReactMethod
    fun captureCrop(targetTimeMs: Double, promise: Promise) {
        Thread {
            try {
                val frame: FrameRec?
                synchronized(frameLock) {
                    frame = if (targetTimeMs > 0) {
                        frameBuffer.minByOrNull { abs(it.tsMs - targetTimeMs.toLong()) }
                    } else {
                        frameBuffer.lastOrNull()
                    }
                }
                if (frame == null) {
                    promise.reject("NO_FRAME", "Ingen kameraframe tillganglig annu")
                    return@Thread
                }
                val frameDelayMs = frame.tsMs - targetTimeMs.toLong()
                var bitmap = frameToBitmap(frame, 92)

                val mpImage = com.google.mediapipe.framework.image.BitmapImageBuilder(bitmap).build()
                val poseResult = try { ensureLandmarker().detect(mpImage) } catch (_: Exception) { null }

                val bw = bitmap.width
                val bh = bitmap.height
                var x0: Int
                var y0: Int
                var x1: Int
                var y1: Int
                var source: String
                val landmarks = poseResult?.landmarks()?.firstOrNull()
                if (landmarks != null && landmarks.size > 16) {
                    val lw = landmarks[15]
                    val rw = landmarks[16]
                    val le = landmarks[13]
                    val re = landmarks[14]
                    val useRight = rw.visibility().orElse(0f) >= lw.visibility().orElse(0f)
                    val wristL = if (useRight) rw else lw
                    val elbowL = if (useRight) re else le
                    val wx = wristL.x() * bw
                    val wy = wristL.y() * bh
                    val fx = wx - elbowL.x() * bw
                    val fy = wy - elbowL.y() * bh
                    val flen = max(hypot(fx.toDouble(), fy.toDouble()).toFloat(), 1f)
                    val cx = wx + 0.8f * fx
                    val cy = wy + 0.8f * fy
                    val half = 1.3f * flen
                    x0 = max(0, (cx - half).toInt())
                    y0 = max(0, (cy - half).toInt())
                    x1 = min(bw, (cx + half).toInt())
                    y1 = min(bh, (cy + half).toInt())
                    source = "wrist_anchor"
                    if (x1 - x0 < 24 || y1 - y0 < 24) {
                        x0 = bw / 3
                        y0 = bh / 3
                        x1 = 2 * bw / 3
                        y1 = 2 * bh / 3
                        source = "center_fallback"
                    }
                } else {
                    x0 = bw / 3
                    y0 = bh / 3
                    x1 = 2 * bw / 3
                    y1 = 2 * bh / 3
                    source = "center_fallback"
                }

                val cropped = Bitmap.createBitmap(bitmap, x0, y0, x1 - x0, y1 - y0)
                val scaled = Bitmap.createScaledBitmap(cropped, 64, 64, true)
                val pixels = IntArray(64 * 64)
                scaled.getPixels(pixels, 0, 64, 0, 0, 64, 64)
                val rgb = ByteArray(64 * 64 * 3)
                var offset = 0
                for (pixel in pixels) {
                    rgb[offset++] = ((pixel shr 16) and 0xFF).toByte()
                    rgb[offset++] = ((pixel shr 8) and 0xFF).toByte()
                    rgb[offset++] = (pixel and 0xFF).toByte()
                }
                if (scaled !== cropped) scaled.recycle()
                if (cropped !== bitmap) cropped.recycle()
                bitmap.recycle()

                promise.resolve(Arguments.createMap().apply {
                    putString("rgb_b64", Base64.encodeToString(rgb, Base64.NO_WRAP))
                    putString("roi_source", source)
                    putDouble("frame_delay_ms", frameDelayMs.toDouble())
                })
            } catch (error: Exception) {
                promise.reject("CAPTURE_ERROR", error.message, error)
            }
        }.start()
    }

    private fun maybeAnalyzeRacketTrack(frame: FrameRec) {
        val now = System.currentTimeMillis()
        if (trackBusy || now - lastTrackAnalyzeMs < TRACK_INTERVAL_MS) return
        lastTrackAnalyzeMs = now
        trackBusy = true
        Thread {
            try {
                val track = analyzeRacketTrack(frame)
                recordTrack(track)
            } catch (_: Exception) {
                recordTrack(lostTrack(frame.tsMs))
            } finally {
                trackBusy = false
            }
        }.start()
    }

    private fun analyzeRacketTrack(frame: FrameRec): RacketTrackRec {
        var bitmap: Bitmap? = null
        var displayed = false
        return try {
            bitmap = frameToBitmap(frame, 72, useFrontCameraForPreview)
            val previous = synchronized(trackLock) { lastGoodTrack }
            val candidate = detectRacketCandidate(bitmap, previous, false)
            displayed = BounceSideCameraHolder.displayFrame(bitmap)
            if (candidate != null) {
                buildTrackedRecord(frame.tsMs, candidate, previous)
            } else {
                buildHoldOrLostRecord(frame.tsMs)
            }
        } finally {
            if (!displayed) bitmap?.recycle()
        }
    }

    private fun buildTrackedRecord(
        tsMs: Long,
        candidate: ComponentCandidate,
        previous: RacketTrackRec?,
    ): RacketTrackRec {
        var x = candidate.x
        var y = candidate.y
        var w = candidate.width
        var h = candidate.height
        if (previous != null && previous.tracked) {
            val cx = x + w / 2.0
            val cy = y + h / 2.0
            val pcx = previous.x + previous.width / 2.0
            val pcy = previous.y + previous.height / 2.0
            val dist = hypot(cx - pcx, cy - pcy)
            val gate = 0.18 + max(previous.width, previous.height) * 0.8
            if (dist <= gate) {
                val alpha = 0.42
                x = previous.x * (1.0 - alpha) + x * alpha
                y = previous.y * (1.0 - alpha) + y * alpha
                w = previous.width * (1.0 - alpha) + w * alpha
                h = previous.height * (1.0 - alpha) + h * alpha
            }
        }
        x = clamp(x, 0.0, 1.0)
        y = clamp(y, 0.0, 1.0)
        w = clamp(w, 0.02, 1.0 - x)
        h = clamp(h, 0.02, 1.0 - y)

        val color = when {
            candidate.color == "red" && candidate.confidence >= TRACK_COLOR_CONFIDENCE -> "red"
            candidate.color == "black" && candidate.confidence >= TRACK_BLACK_CONFIDENCE -> "black"
            else -> "uncertain"
        }
        val label = when (color) {
            "red" -> "racket-red"
            "black" -> "racket-black"
            else -> "racket"
        }
        val record = RacketTrackRec(
            tsMs = tsMs,
            tracked = true,
            label = label,
            color = color,
            confidence = candidate.confidence,
            x = x,
            y = y,
            width = w,
            height = h,
            source = "color_blob",
            redScore = candidate.redScore,
            darkScore = candidate.darkScore,
            areaRatio = candidate.areaRatio,
            fillRatio = candidate.fillRatio,
        )
        synchronized(trackLock) { lastGoodTrack = record }
        return record
    }

    private fun buildHoldOrLostRecord(tsMs: Long): RacketTrackRec {
        val previous = synchronized(trackLock) { lastGoodTrack }
        if (previous != null && tsMs - previous.tsMs <= TRACK_HOLD_MS) {
            val age = max(0L, tsMs - previous.tsMs).toDouble()
            val factor = clamp(1.0 - age / TRACK_HOLD_MS, 0.25, 0.8)
            return previous.copy(
                tsMs = tsMs,
                confidence = previous.confidence * factor,
                source = "hold",
            )
        }
        return lostTrack(tsMs)
    }

    private fun recordTrack(track: RacketTrackRec) {
        synchronized(trackLock) {
            trackBuffer.addLast(track)
            while (trackBuffer.size > TRACK_MAX_BUFFERED) trackBuffer.removeFirst()
        }
        emitTrack(track)
    }

    private fun emitTrack(track: RacketTrackRec) {
        if (!ctx.hasActiveCatalystInstance()) return
        ctx.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(TRACK_EVENT_NAME, trackToMap(track, System.currentTimeMillis(), track.tsMs))
    }

    private fun trackToMap(track: RacketTrackRec, nowMs: Long, targetMs: Long) =
        Arguments.createMap().apply {
            putBoolean("tracked", track.tracked)
            putString("label", track.label)
            putString("color", track.color)
            putDouble("confidence", track.confidence)
            putDouble("x", track.x)
            putDouble("y", track.y)
            putDouble("width", track.width)
            putDouble("height", track.height)
            putDouble("timestamp_ms", track.tsMs.toDouble())
            putDouble("age_ms", max(0L, nowMs - track.tsMs).toDouble())
            putDouble("frame_delay_ms", (track.tsMs - targetMs).toDouble())
            putString("source", track.source)
            putDouble("red_score", track.redScore)
            putDouble("dark_score", track.darkScore)
            putDouble("area_ratio", track.areaRatio)
            putDouble("fill_ratio", track.fillRatio)
        }

    private fun lostTrack(tsMs: Long) = RacketTrackRec(
        tsMs = tsMs,
        tracked = false,
        label = "lost",
        color = "uncertain",
        confidence = 0.0,
        x = 0.0,
        y = 0.0,
        width = 0.0,
        height = 0.0,
        source = "lost",
        redScore = 0.0,
        darkScore = 0.0,
        areaRatio = 0.0,
        fillRatio = 0.0,
    )

    private fun detectRacketCandidate(
        bitmap: Bitmap,
        previous: RacketTrackRec?,
        mirrorX: Boolean,
    ): ComponentCandidate? {
        val scale = min(1.0, TRACK_TARGET_WIDTH.toDouble() / bitmap.width.toDouble())
        val sw = max(1, (bitmap.width * scale).roundToInt())
        val sh = max(1, (bitmap.height * scale).roundToInt())
        val scaled = if (sw == bitmap.width && sh == bitmap.height) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, sw, sh, true)
        }
        try {
            val total = sw * sh
            val pixels = IntArray(total)
            scaled.getPixels(pixels, 0, sw, 0, 0, sw, sh)
            val mask = ByteArray(total)
            for (i in 0 until total) {
                mask[i] = pixelMask(pixels[i])
            }

            val visited = BooleanArray(total)
            val queue = IntArray(total)
            var best: ComponentCandidate? = null
            for (start in 0 until total) {
                val colorMask = mask[start]
                if (colorMask.toInt() == 0 || visited[start]) continue
                var head = 0
                var tail = 0
                queue[tail++] = start
                visited[start] = true
                var count = 0
                var minX = sw
                var minY = sh
                var maxX = 0
                var maxY = 0
                while (head < tail) {
                    val p = queue[head++]
                    val x = p % sw
                    val y = p / sw
                    count += 1
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y

                    val left = p - 1
                    val right = p + 1
                    val up = p - sw
                    val down = p + sw
                    if (x > 0 && !visited[left] && mask[left] == colorMask) {
                        visited[left] = true
                        queue[tail++] = left
                    }
                    if (x < sw - 1 && !visited[right] && mask[right] == colorMask) {
                        visited[right] = true
                        queue[tail++] = right
                    }
                    if (y > 0 && !visited[up] && mask[up] == colorMask) {
                        visited[up] = true
                        queue[tail++] = up
                    }
                    if (y < sh - 1 && !visited[down] && mask[down] == colorMask) {
                        visited[down] = true
                        queue[tail++] = down
                    }
                }
                val candidate = componentToCandidate(
                    colorMask = colorMask,
                    count = count,
                    minX = minX,
                    minY = minY,
                    maxX = maxX,
                    maxY = maxY,
                    sw = sw,
                    sh = sh,
                    previous = previous,
                    mirrorX = mirrorX,
                ) ?: continue
                if (best == null || candidate.confidence > best!!.confidence) {
                    best = candidate
                }
            }
            return best
        } finally {
            if (scaled !== bitmap) scaled.recycle()
        }
    }

    private fun componentToCandidate(
        colorMask: Byte,
        count: Int,
        minX: Int,
        minY: Int,
        maxX: Int,
        maxY: Int,
        sw: Int,
        sh: Int,
        previous: RacketTrackRec?,
        mirrorX: Boolean,
    ): ComponentCandidate? {
        val total = sw * sh
        val minPixels = max(42, (total * 0.00065).roundToInt())
        if (count < minPixels) return null

        val widthPx = maxX - minX + 1
        val heightPx = maxY - minY + 1
        if (widthPx < sw * 0.035 || heightPx < sh * 0.022) return null

        val bboxArea = widthPx * heightPx
        val boxRatio = bboxArea.toDouble() / total.toDouble()
        val fill = count.toDouble() / max(1, bboxArea).toDouble()
        if (fill < 0.15) return null

        val aspect = widthPx.toDouble() / max(1, heightPx).toDouble()
        if (aspect < 0.22 || aspect > 4.5) return null

        val isBlack = colorMask.toInt() == 2
        val areaRatio = count.toDouble() / total.toDouble()
        val widthNorm = widthPx.toDouble() / sw.toDouble()
        val heightNorm = heightPx.toDouble() / sh.toDouble()
        val closeUpBlackRubber = isBlack &&
            areaRatio <= 0.58 &&
            boxRatio <= 0.72 &&
            fill >= 0.38 &&
            aspect >= 0.45 &&
            aspect <= 2.05 &&
            widthNorm <= 0.93 &&
            heightNorm <= 0.82
        val maxArea = when {
            closeUpBlackRubber -> 0.58
            isBlack -> 0.09
            else -> 0.22
        }
        if (areaRatio > maxArea) return null

        val rawX = minX.toDouble() / sw.toDouble()
        val x = if (mirrorX) 1.0 - rawX - widthNorm else rawX
        val y = minY.toDouble() / sh.toDouble()
        val color = if (colorMask.toInt() == 1) "red" else "black"
        val cx = x + widthNorm / 2.0
        val cy = y + heightNorm / 2.0
        val previousDistance = if (previous != null && previous.tracked) {
            val pcx = previous.x + previous.width / 2.0
            val pcy = previous.y + previous.height / 2.0
            hypot(cx - pcx, cy - pcy)
        } else {
            null
        }
        val previousGate = if (previous != null && previous.tracked) {
            0.18 + max(previous.width, previous.height) * 0.8
        } else {
            0.0
        }
        val confidenceGate = if (previous != null && previous.tracked) {
            0.20 + max(previous.width, previous.height) * 0.8
        } else {
            0.0
        }
        val nearPrevious = previousDistance != null && previousDistance <= previousGate
        if (isBlack) {
            val touchesFrame = minX <= 1 || minY <= 1 || maxX >= sw - 2 || maxY >= sh - 2
            val tallBodyLike = heightNorm > 0.34 && heightNorm > widthNorm * 1.35
            val broadBodyLike = widthNorm > 0.54 || heightNorm > 0.46 || boxRatio > 0.14
            if (fill < 0.22) return null
            if (aspect < 0.32 || aspect > 3.4) return null
            if (broadBodyLike && !nearPrevious && !closeUpBlackRubber) return null
            if (tallBodyLike && !nearPrevious && !closeUpBlackRubber) return null
            if (touchesFrame && !nearPrevious && !closeUpBlackRubber) return null
            if (cy < 0.18 && !nearPrevious && !closeUpBlackRubber) return null
        }
        val aspectScore = 1.0 - min(1.0, abs(ln(max(0.05, aspect))) / 1.55)
        var confidence = 0.22 +
            min(1.0, areaRatio / 0.035) * 0.22 +
            min(1.0, boxRatio / 0.13) * 0.12 +
            fill * 0.26 +
            aspectScore * 0.22 +
            if (color == "red") 0.08 else 0.0
        if (color == "black") {
            confidence -= 0.10
            if (closeUpBlackRubber) confidence += 0.24
            if (boxRatio > 0.10 && !closeUpBlackRubber) confidence -= 0.12
            if (heightNorm > 0.30 && heightNorm > widthNorm * 1.20 && !closeUpBlackRubber) confidence -= 0.12
            if ((previous == null || !previous.tracked) && !closeUpBlackRubber) confidence -= 0.06
        }
        if (previousDistance != null) {
            if (previousDistance < confidenceGate) {
                confidence += 0.16
            } else if (previousDistance > 0.55) {
                confidence -= if (color == "black") 0.22 else 0.10
            }
        }
        confidence = clamp(confidence, 0.0, 0.99)
        return ComponentCandidate(
            color = color,
            x = clamp(x, 0.0, 1.0),
            y = clamp(y, 0.0, 1.0),
            width = clamp(widthNorm, 0.02, 1.0),
            height = clamp(heightNorm, 0.02, 1.0),
            confidence = confidence,
            redScore = if (color == "red") areaRatio else 0.0,
            darkScore = if (color == "black") areaRatio else 0.0,
            areaRatio = areaRatio,
            fillRatio = fill,
        )
    }

    private fun pixelMask(pixel: Int): Byte {
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        val v = max(r, max(g, b))
        val mn = min(r, min(g, b))
        val delta = v - mn
        val s = if (v == 0) 0 else (255 * delta) / v
        var h = 0.0
        if (delta > 0) {
            h = when (v) {
                r -> 60.0 * (g - b).toDouble() / delta.toDouble()
                g -> 120.0 + 60.0 * (b - r).toDouble() / delta.toDouble()
                else -> 240.0 + 60.0 * (r - g).toDouble() / delta.toDouble()
            }
            if (h < 0.0) h += 360.0
        }
        var h8 = (h / 2.0).roundToInt()
        if (h8 >= 180) h8 -= 180
        val redDominance = r - max(g, b)
        val red = (h8 <= 5 || h8 >= 175) && s >= 95 && v >= 45 && redDominance >= 42
        if (red) return 1
        val dark = v < 78 && s < 145
        if (dark) return 2
        return 0
    }

    private fun frameToBitmap(frame: FrameRec, jpegQuality: Int, mirrorX: Boolean = false): Bitmap {
        val yuv = YuvImage(frame.nv21, android.graphics.ImageFormat.NV21, frame.width, frame.height, null)
        val jpegStream = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, frame.width, frame.height), jpegQuality, jpegStream)
        val jpegBytes = jpegStream.toByteArray()
        var bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
        if (frame.rotation != 0) {
            val matrix = Matrix()
            matrix.postRotate(frame.rotation.toFloat())
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (rotated !== bitmap) bitmap.recycle()
            bitmap = rotated
        }
        if (mirrorX) {
            val matrix = Matrix()
            matrix.postScale(-1f, 1f)
            val mirrored = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (mirrored !== bitmap) bitmap.recycle()
            bitmap = mirrored
        }
        return bitmap
    }

    private fun clamp(value: Double, minValue: Double, maxValue: Double): Double =
        min(max(value, minValue), maxValue)

    private fun imageProxyToNv21(proxy: androidx.camera.core.ImageProxy): ByteArray {
        val width = proxy.width
        val height = proxy.height
        val out = ByteArray(width * height * 3 / 2)
        val yPlane = proxy.planes[0]
        val uPlane = proxy.planes[1]
        val vPlane = proxy.planes[2]

        var offset = 0
        val yBuf = yPlane.buffer
        val yRow = yPlane.rowStride
        val yPix = yPlane.pixelStride
        if (yPix == 1 && yRow == width) {
            yBuf.position(0)
            yBuf.get(out, 0, width * height)
            offset = width * height
        } else {
            for (row in 0 until height) {
                var pos = row * yRow
                for (col in 0 until width) {
                    out[offset++] = yBuf.get(pos)
                    pos += yPix
                }
            }
        }
        val chromaH = height / 2
        val chromaW = width / 2
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        val uRow = uPlane.rowStride
        val uPix = uPlane.pixelStride
        val vRow = vPlane.rowStride
        val vPix = vPlane.pixelStride
        for (row in 0 until chromaH) {
            for (col in 0 until chromaW) {
                out[offset++] = vBuf.get(row * vRow + col * vPix)
                out[offset++] = uBuf.get(row * uRow + col * uPix)
            }
        }
        return out
    }
}
