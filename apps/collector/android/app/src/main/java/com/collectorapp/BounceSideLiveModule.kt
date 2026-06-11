package com.collectorapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Base64
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.uimanager.SimpleViewManager
import com.facebook.react.uimanager.ThemedReactContext
import java.io.ByteArrayOutputStream

/**
 * Live FH/BH-studs: kameraström (CameraX) med senaste-frame-buffer.
 * JS-flödet: ljuddetektorn (AudioStream + Fable-modellen) larmar om en
 * bollträff -> JS anropar captureCrop() -> senaste kameraframen ->
 * MediaPipe-pose (samma motor/modellfil som träningen) -> handleds-
 * ankrad racket-crop 64x64 RGB -> sidomodellen i JS avgör FH/BH.
 */
object BounceSideCameraHolder {
    @Volatile var previewView: PreviewView? = null
}

class BounceSideCameraViewManager : SimpleViewManager<PreviewView>() {
    override fun getName() = "BounceSideCameraView"

    override fun createViewInstance(ctx: ThemedReactContext): PreviewView {
        val view = PreviewView(ctx)
        view.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        BounceSideCameraHolder.previewView = view
        return view
    }

    override fun onDropViewInstance(view: PreviewView) {
        if (BounceSideCameraHolder.previewView === view) {
            BounceSideCameraHolder.previewView = null
        }
        super.onDropViewInstance(view)
    }
}

class BounceSideLiveModule(private val ctx: ReactApplicationContext)
    : ReactContextBaseJavaModule(ctx) {

    override fun getName() = "BounceSideLive"

    @Volatile private var cameraProvider: ProcessCameraProvider? = null

    // Ringbuffert med senaste ~halvsekunden av frames (NV21 + tidsstämpel):
    // ljudkedjan hinner processa i 200-300 ms innan JS ber om bilden, och
    // racketen har då hunnit röra sig - vi vill ha bilden från TRÄFF-
    // ögonblicket (onset_time_ms), inte från "nu".
    private class FrameRec(
        val tsMs: Long,
        val nv21: ByteArray,
        val width: Int,
        val height: Int,
        val rotation: Int,
    )

    private val frameLock = Any()
    private val frameBuffer = ArrayDeque<FrameRec>()
    private val maxBufferedFrames = 16

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
    fun startCamera(useFrontCamera: Boolean, promise: Promise) {
        val activity = ctx.currentActivity
        if (activity !is LifecycleOwner) {
            promise.reject("NO_ACTIVITY", "Ingen aktiv activity för kamerabindning")
            return
        }
        val mainExecutor = ContextCompat.getMainExecutor(ctx)
        val providerFuture = ProcessCameraProvider.getInstance(ctx)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                cameraProvider = provider
                provider.unbindAll()

                val preview = Preview.Builder().build()
                BounceSideCameraHolder.previewView?.let { view ->
                    preview.setSurfaceProvider(view.surfaceProvider)
                }

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
                    } finally {
                        proxy.close()
                    }
                }

                val selector = if (useFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA
                               else CameraSelector.DEFAULT_BACK_CAMERA
                provider.bindToLifecycle(activity, selector, preview, analysis)
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
        synchronized(frameLock) { frameBuffer.clear() }
        promise.resolve("stopped")
    }

    /**
     * Ta racket-crop ur framen NÄRMAST `targetTimeMs` (träffögonblicket,
     * System.currentTimeMillis-bas, samma klocka som ljudets onset_time_ms):
     * MediaPipe-pose -> handleds-ankrad kvadrat -> 64x64 RGB (base64).
     * Samma geometri som träningen.
     */
    @ReactMethod
    fun captureCrop(targetTimeMs: Double, promise: Promise) {
        Thread {
            try {
                val frame: FrameRec?
                synchronized(frameLock) {
                    frame = if (targetTimeMs > 0) {
                        frameBuffer.minByOrNull { kotlin.math.abs(it.tsMs - targetTimeMs.toLong()) }
                    } else {
                        frameBuffer.lastOrNull()
                    }
                }
                if (frame == null) {
                    promise.reject("NO_FRAME", "Ingen kameraframe tillgänglig ännu")
                    return@Thread
                }
                val nv21 = frame.nv21
                val w = frame.width
                val h = frame.height
                val rotation = frame.rotation
                val frameDelayMs = frame.tsMs - targetTimeMs.toLong()

                // NV21 -> Bitmap via Androids egen JPEG-väg (korrekt färgrymd),
                // sedan rotation till stående.
                val yuv = YuvImage(nv21, android.graphics.ImageFormat.NV21, w, h, null)
                val jpegStream = ByteArrayOutputStream()
                yuv.compressToJpeg(Rect(0, 0, w, h), 92, jpegStream)
                val jpegBytes = jpegStream.toByteArray()
                var bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
                if (rotation != 0) {
                    val matrix = Matrix()
                    matrix.postRotate(rotation.toFloat())
                    val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                    if (rotated !== bitmap) bitmap.recycle()
                    bitmap = rotated
                }

                val mpImage = com.google.mediapipe.framework.image.BitmapImageBuilder(bitmap).build()
                val poseResult = try { ensureLandmarker().detect(mpImage) } catch (_: Exception) { null }

                val bw = bitmap.width
                val bh = bitmap.height
                var x0: Int; var y0: Int; var x1: Int; var y1: Int; var source: String
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
                    val flen = maxOf(Math.hypot(fx.toDouble(), fy.toDouble()).toFloat(), 1f)
                    val cx = wx + 0.8f * fx
                    val cy = wy + 0.8f * fy
                    val half = 1.3f * flen
                    x0 = maxOf(0, (cx - half).toInt()); y0 = maxOf(0, (cy - half).toInt())
                    x1 = minOf(bw, (cx + half).toInt()); y1 = minOf(bh, (cy + half).toInt())
                    source = "wrist_anchor"
                    if (x1 - x0 < 24 || y1 - y0 < 24) {
                        x0 = bw / 3; y0 = bh / 3; x1 = 2 * bw / 3; y1 = 2 * bh / 3; source = "center_fallback"
                    }
                } else {
                    x0 = bw / 3; y0 = bh / 3; x1 = 2 * bw / 3; y1 = 2 * bh / 3; source = "center_fallback"
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

    /** YUV_420_888 ImageProxy -> NV21 (hanterar strides). */
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
