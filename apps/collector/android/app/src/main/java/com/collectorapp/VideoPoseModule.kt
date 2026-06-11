package com.collectorapp

import android.media.Image
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.WritableArray
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import java.io.File
import kotlin.math.max

/**
 * Pose-extraktion ur video.
 *
 * Snabb väg: videon avkodas EN gång i presentationsordning med MediaCodec
 * och frames plockas ur strömmen vid samplingsintervallet. Den gamla vägen
 * (MediaMetadataRetriever.getFrameAtTime per sample med OPTION_CLOSEST)
 * tvingade fram en nyckelbilds-seek + omavkodning av alla mellanliggande
 * frames för VARJE sample - en 60 s video avkodades i praktiken 20-50
 * gånger om, vilket gjorde helvideo-skanningen ohållbart långsam.
 *
 * Sekventiell avkodning är dessutom bättre för kvaliteten: ML Kits
 * STREAM_MODE använder spårning mellan frames, som bara fungerar som tänkt
 * när framesen kommer i äkta tidsordning.
 *
 * Fallback: om MediaCodec-vägen misslyckas (ovanlig kodek etc.) körs den
 * gamla retriever-vägen så att inget flöde går sönder.
 */
class VideoPoseModule(private val ctx: ReactApplicationContext)
    : ReactContextBaseJavaModule(ctx) {

    override fun getName() = "VideoPose"

    @ReactMethod
    fun extractPose(videoPath: String, sampleFps: Double, promise: Promise) {
        runExtraction(videoPath, sampleFps, null, promise)
    }

    /**
     * Pose enbart i givna tidsfönster (flat array [start0, end0, start1, end1, ...]
     * i ms). Videon avkodas fortfarande sekventiellt (billigt), men konvertering
     * + ML Kit-inferens hoppar över frames utanför fönstren - stor vinst när
     * analysen är ljudankrad och bara behöver pose kring bollträffarna.
     */
    @ReactMethod
    fun extractPoseInWindows(videoPath: String, sampleFps: Double, windowsMs: com.facebook.react.bridge.ReadableArray, promise: Promise) {
        val windows = ArrayList<LongRange>(windowsMs.size() / 2)
        var index = 0
        while (index + 1 < windowsMs.size()) {
            val start = windowsMs.getDouble(index).toLong()
            val end = windowsMs.getDouble(index + 1).toLong()
            if (end > start) windows.add(start..end)
            index += 2
        }
        runExtraction(videoPath, sampleFps, windows.sortedBy { it.first }, promise)
    }

    private fun runExtraction(videoPath: String, sampleFps: Double, windows: List<LongRange>?, promise: Promise) {
        Thread {
            val cleanPath = videoPath.removePrefix("file://")
            if (!File(cleanPath).exists()) {
                promise.reject("VIDEO_NOT_FOUND", "Video file does not exist: $cleanPath")
                return@Thread
            }
            val safeSampleFps = max(1.0, sampleFps)
            try {
                val result = try {
                    extractWithMediaCodec(cleanPath, safeSampleFps, windows)
                } catch (codecError: Exception) {
                    extractWithRetriever(cleanPath, safeSampleFps)
                }
                promise.resolve(result)
            } catch (error: Exception) {
                promise.reject("VIDEO_POSE_ERROR", error.message, error)
            }
        }.start()
    }

    /**
     * Extrahera handleds-ankrade racket-crops (64x64 RGB) vid givna
     * tidsstämplar, för FH-/BH-sidoklassificeringen i Video studs FH/BH.
     * Sekventiell avkodning; vid varje ankare tas första framen >= ts,
     * pose körs på den upprätta halvupplösta bilden, och en kvadrat
     * centrerad bortom handleden längs underarmen beskärs (samma geometri
     * som träningens wrist_anchored_roi i classify_bounce_side.py).
     * Returnerar [{timestamp_ms, frame_ms, rgb_b64 (64*64*3), roi_source}].
     */
    @ReactMethod
    fun extractBounceSideCrops(videoPath: String, timestampsMs: com.facebook.react.bridge.ReadableArray, promise: Promise) {
        Thread {
            val cleanPath = videoPath.removePrefix("file://")
            if (!File(cleanPath).exists()) {
                promise.reject("VIDEO_NOT_FOUND", "Video file does not exist: $cleanPath")
                return@Thread
            }
            try {
                val anchors = ArrayList<Long>(timestampsMs.size())
                for (i in 0 until timestampsMs.size()) anchors.add(timestampsMs.getDouble(i).toLong())
                anchors.sort()
                promise.resolve(extractCropsWithRetriever(cleanPath, anchors))
            } catch (error: Exception) {
                promise.reject("BOUNCE_SIDE_CROP_ERROR", error.message, error)
            }
        }.start()
    }

    /**
     * Crops via Androids egen bitmap-avkodning (MediaMetadataRetriever):
     * korrekt färgrymd (BT.709 vs BT.601 - modellens features ÄR
     * färgstatistik, min handrullade YUV->RGB gav färgskiftade crops och
     * fel FH/BH-förslag), full upplösning och bilinjär skalning - närmast
     * träningens cv2-pipeline. Långsamma per-frame-seeks är OK här: bara
     * ~40 ankarframes per video, inte 900.
     */
    private fun extractCropsWithRetriever(path: String, anchors: List<Long>): WritableArray {
        val retriever = MediaMetadataRetriever()
        val detector = PoseDetection.getClient(
            PoseDetectorOptions.Builder()
                .setDetectorMode(PoseDetectorOptions.SINGLE_IMAGE_MODE)
                .build()
        )
        val results = Arguments.createArray()
        try {
            retriever.setDataSource(path)
            for (anchorTs in anchors) {
                val bitmap = retriever.getFrameAtTime(anchorTs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST)
                    ?: continue
                val input = InputImage.fromBitmap(bitmap, 0)
                val pose = try { Tasks.await(detector.process(input)) } catch (_: Exception) { null }

                val w = bitmap.width
                val h = bitmap.height
                var x0: Int; var y0: Int; var x1: Int; var y1: Int; var source: String
                val rWrist = pose?.getPoseLandmark(com.google.mlkit.vision.pose.PoseLandmark.RIGHT_WRIST)
                val lWrist = pose?.getPoseLandmark(com.google.mlkit.vision.pose.PoseLandmark.LEFT_WRIST)
                val rElbow = pose?.getPoseLandmark(com.google.mlkit.vision.pose.PoseLandmark.RIGHT_ELBOW)
                val lElbow = pose?.getPoseLandmark(com.google.mlkit.vision.pose.PoseLandmark.LEFT_ELBOW)
                val useRight = (rWrist?.inFrameLikelihood ?: 0f) >= (lWrist?.inFrameLikelihood ?: 0f)
                val wrist = if (useRight) rWrist else lWrist
                val elbow = if (useRight) rElbow else lElbow
                if (wrist != null && elbow != null) {
                    val wx = wrist.position.x
                    val wy = wrist.position.y
                    val fx = wx - elbow.position.x
                    val fy = wy - elbow.position.y
                    val flen = maxOf(Math.hypot(fx.toDouble(), fy.toDouble()).toFloat(), 1f)
                    val cx = wx + 0.8f * fx
                    val cy = wy + 0.8f * fy
                    val half = 1.3f * flen
                    x0 = maxOf(0, (cx - half).toInt()); y0 = maxOf(0, (cy - half).toInt())
                    x1 = minOf(w, (cx + half).toInt()); y1 = minOf(h, (cy + half).toInt())
                    source = "wrist_anchor"
                    if (x1 - x0 < 24 || y1 - y0 < 24) {
                        x0 = w / 3; y0 = h / 3; x1 = 2 * w / 3; y1 = 2 * h / 3; source = "center_fallback"
                    }
                } else {
                    x0 = w / 3; y0 = h / 3; x1 = 2 * w / 3; y1 = 2 * h / 3; source = "center_fallback"
                }

                val cropped = android.graphics.Bitmap.createBitmap(bitmap, x0, y0, x1 - x0, y1 - y0)
                val scaled = android.graphics.Bitmap.createScaledBitmap(cropped, 64, 64, true)
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

                results.pushMap(Arguments.createMap().apply {
                    putDouble("timestamp_ms", anchorTs.toDouble())
                    putDouble("frame_ms", anchorTs.toDouble())
                    putString("rgb_b64", android.util.Base64.encodeToString(rgb, android.util.Base64.NO_WRAP))
                    putString("roi_source", source)
                })
            }
        } finally {
            retriever.release()
            detector.close()
        }
        return results
    }

    @Suppress("unused")
    private fun extractCropsWithMediaCodec(path: String, anchors: List<Long>): WritableArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(path)
        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val trackFormat = extractor.getTrackFormat(i)
            val mime = trackFormat.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("video/")) { trackIndex = i; format = trackFormat; break }
        }
        if (trackIndex < 0 || format == null) {
            extractor.release()
            throw IllegalStateException("No video track")
        }
        extractor.selectTrack(trackIndex)
        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val rotation = try { format.getInteger(MediaFormat.KEY_ROTATION) } catch (_: Exception) { 0 }

        val detector = PoseDetection.getClient(
            PoseDetectorOptions.Builder()
                .setDetectorMode(PoseDetectorOptions.SINGLE_IMAGE_MODE)
                .build()
        )
        val codec = MediaCodec.createDecoderByType(mime)
        val results = Arguments.createArray()

        try {
            codec.configure(format, null, null, 0)
            codec.start()
            val bufferInfo = MediaCodec.BufferInfo()
            var anchorIdx = 0
            var inputDone = false
            var outputDone = false

            while (!outputDone && anchorIdx < anchors.size) {
                if (!inputDone) {
                    val inIndex = codec.dequeueInputBuffer(10_000)
                    if (inIndex >= 0) {
                        val inputBuffer = codec.getInputBuffer(inIndex)!!
                        val sampleSize = extractor.readSampleData(inputBuffer, 0)
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIndex, 0, sampleSize, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val outIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000)
                if (outIndex >= 0) {
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                    val ptsMs = bufferInfo.presentationTimeUs / 1000L
                    if (bufferInfo.size > 0 && anchorIdx < anchors.size && ptsMs >= anchors[anchorIdx]) {
                        val image = codec.getOutputImage(outIndex)
                        if (image != null) {
                            val converted = yuv420ToUprightNv21(image, rotation)
                            val anchorTs = anchors[anchorIdx]
                            val crop = buildWristCrop(converted, detector)
                            results.pushMap(Arguments.createMap().apply {
                                putDouble("timestamp_ms", anchorTs.toDouble())
                                putDouble("frame_ms", ptsMs.toDouble())
                                putString("rgb_b64", android.util.Base64.encodeToString(crop.first, android.util.Base64.NO_WRAP))
                                putString("roi_source", crop.second)
                            })
                            // hoppa över alla ankare som denna frame täckte
                            while (anchorIdx < anchors.size && ptsMs >= anchors[anchorIdx]) anchorIdx += 1
                        }
                    }
                    codec.releaseOutputBuffer(outIndex, false)
                }
            }
        } finally {
            try { codec.stop() } catch (_: Exception) {}
            codec.release()
            extractor.release()
            detector.close()
        }
        return results
    }

    /** Pose -> handleds-ankrad kvadrat -> 64x64 RGB-bytes (BT.601 limited range). */
    private fun buildWristCrop(frame: ConvertedFrame, detector: com.google.mlkit.vision.pose.PoseDetector): Pair<ByteArray, String> {
        val w = frame.width
        val h = frame.height
        val input = InputImage.fromByteArray(frame.data, w, h, 0, InputImage.IMAGE_FORMAT_NV21)
        val pose = try { Tasks.await(detector.process(input)) } catch (_: Exception) { null }

        var x0: Int; var y0: Int; var x1: Int; var y1: Int; var source: String
        val rWrist = pose?.getPoseLandmark(com.google.mlkit.vision.pose.PoseLandmark.RIGHT_WRIST)
        val lWrist = pose?.getPoseLandmark(com.google.mlkit.vision.pose.PoseLandmark.LEFT_WRIST)
        val rElbow = pose?.getPoseLandmark(com.google.mlkit.vision.pose.PoseLandmark.RIGHT_ELBOW)
        val lElbow = pose?.getPoseLandmark(com.google.mlkit.vision.pose.PoseLandmark.LEFT_ELBOW)
        val useRight = (rWrist?.inFrameLikelihood ?: 0f) >= (lWrist?.inFrameLikelihood ?: 0f)
        val wrist = if (useRight) rWrist else lWrist
        val elbow = if (useRight) rElbow else lElbow
        if (wrist != null && elbow != null) {
            val wx = wrist.position.x
            val wy = wrist.position.y
            val fx = wx - elbow.position.x
            val fy = wy - elbow.position.y
            val flen = maxOf(Math.hypot(fx.toDouble(), fy.toDouble()).toFloat(), 1f)
            val cx = wx + 0.8f * fx
            val cy = wy + 0.8f * fy
            val half = 1.3f * flen
            x0 = maxOf(0, (cx - half).toInt()); y0 = maxOf(0, (cy - half).toInt())
            x1 = minOf(w, (cx + half).toInt()); y1 = minOf(h, (cy + half).toInt())
            source = "wrist_anchor"
            if (x1 - x0 < 24 || y1 - y0 < 24) {
                x0 = w / 3; y0 = h / 3; x1 = 2 * w / 3; y1 = 2 * h / 3; source = "center_fallback"
            }
        } else {
            x0 = w / 3; y0 = h / 3; x1 = 2 * w / 3; y1 = 2 * h / 3; source = "center_fallback"
        }

        val out = ByteArray(64 * 64 * 3)
        val nv = frame.data
        val cw = x1 - x0
        val ch = y1 - y0
        var offset = 0
        for (dy in 0 until 64) {
            val sy = y0 + (dy * ch) / 64
            for (dx in 0 until 64) {
                val sx = x0 + (dx * cw) / 64
                val yVal = (nv[sy * w + sx].toInt() and 0xFF)
                val uvBase = w * h + (sy / 2) * w + (sx / 2) * 2
                val vVal = (nv[uvBase].toInt() and 0xFF) - 128
                val uVal = (nv[uvBase + 1].toInt() and 0xFF) - 128
                val c = 1.164383 * (yVal - 16)
                var r = (c + 1.596027 * vVal).toInt()
                var g = (c - 0.391762 * uVal - 0.812968 * vVal).toInt()
                var b = (c + 2.017232 * uVal).toInt()
                if (r < 0) r = 0; if (r > 255) r = 255
                if (g < 0) g = 0; if (g > 255) g = 255
                if (b < 0) b = 0; if (b > 255) b = 255
                out[offset++] = r.toByte()
                out[offset++] = g.toByte()
                out[offset++] = b.toByte()
            }
        }
        return Pair(out, source)
    }

    private fun insideWindows(windows: List<LongRange>, ptsMs: Long): Boolean {
        if (windows.isEmpty()) return false
        for (window in windows) {
            if (ptsMs < window.first) return false // sorterade: inget senare fönster kan träffa
            if (ptsMs <= window.last) return true
        }
        return false
    }

    // ── Snabb väg: sekventiell MediaCodec-avkodning ───────────────────────────

    private fun extractWithMediaCodec(
        path: String,
        sampleFps: Double,
        windows: List<LongRange>? = null,
    ): com.facebook.react.bridge.WritableMap {
        val extractor = MediaExtractor()
        extractor.setDataSource(path)
        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val trackFormat = extractor.getTrackFormat(i)
            val mime = trackFormat.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("video/")) {
                trackIndex = i
                format = trackFormat
                break
            }
        }
        if (trackIndex < 0 || format == null) {
            extractor.release()
            throw IllegalStateException("No video track")
        }
        extractor.selectTrack(trackIndex)

        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val rotation = try {
            format.getInteger(MediaFormat.KEY_ROTATION)
        } catch (_: Exception) {
            0
        }
        val durationMs = try {
            format.getLong(MediaFormat.KEY_DURATION) / 1000L
        } catch (_: Exception) {
            0L
        }
        val stepMs = max(1L, (1000.0 / sampleFps).toLong())

        val detector = PoseDetection.getClient(
            PoseDetectorOptions.Builder()
                .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
                .build()
        )
        val codec = MediaCodec.createDecoderByType(mime)
        val frames = Arguments.createArray()

        try {
            codec.configure(format, null, null, 0)
            codec.start()
            val bufferInfo = MediaCodec.BufferInfo()
            var nextTargetMs = 0L
            var inputDone = false
            var outputDone = false

            while (!outputDone) {
                if (!inputDone) {
                    val inIndex = codec.dequeueInputBuffer(10_000)
                    if (inIndex >= 0) {
                        val inputBuffer = codec.getInputBuffer(inIndex)!!
                        val sampleSize = extractor.readSampleData(inputBuffer, 0)
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIndex, 0, sampleSize, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val outIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000)
                if (outIndex >= 0) {
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        outputDone = true
                    }
                    val ptsMs = bufferInfo.presentationTimeUs / 1000L
                    val inWindow = windows == null || insideWindows(windows, ptsMs)
                    val shouldSample = bufferInfo.size > 0 && ptsMs >= nextTargetMs && inWindow
                    if (shouldSample) {
                        val image = codec.getOutputImage(outIndex)
                        if (image != null) {
                            // Konvertera med 2x nedskalning OCH fysisk rotation till
                            // stående läge (rotation=0 till ML Kit). Att rotera datan
                            // själv ger exakt samma koordinatrum som gamla bitmap-
                            // vägen, oberoende av ML Kits rotationssemantik; halverad
                            // upplösning ger ~4x mindre arbete och påverkar inte
                            // features som normaliseras med axelbredden.
                            val converted = yuv420ToUprightNv21(image, rotation)
                            val input = InputImage.fromByteArray(
                                converted.data, converted.width, converted.height,
                                0, InputImage.IMAGE_FORMAT_NV21
                            )
                            val pose = Tasks.await(detector.process(input))
                            pushPoseFrame(frames, ptsMs, pose)
                            while (nextTargetMs <= ptsMs) nextTargetMs += stepMs
                        }
                    }
                    codec.releaseOutputBuffer(outIndex, false)
                }
            }
        } finally {
            try { codec.stop() } catch (_: Exception) {}
            codec.release()
            extractor.release()
            detector.close()
        }

        return Arguments.createMap().apply {
            putString("video_path", path)
            putDouble("sample_fps", sampleFps)
            putDouble("duration_ms", durationMs.toDouble())
            putDouble("frame_count", frames.size().toDouble())
            putArray("frames", frames)
        }
    }

    private class ConvertedFrame(val data: ByteArray, val width: Int, val height: Int)

    private var convertBuffer: ByteArray? = null

    /**
     * YUV_420_888 -> NV21 i STÅENDE läge med 2x nedskalning.
     * Rotationen bakas in i datan (ML Kit får rotation=0) så att
     * koordinatrummet blir identiskt med gamla bitmap-vägen oavsett
     * ML Kit-semantik; nedskalningen ger ~4x mindre konverterings- och
     * inferensarbete utan att påverka axelbredd-normaliserade features.
     */
    private fun yuv420ToUprightNv21(image: Image, rotationDegrees: Int): ConvertedFrame {
        val crop = image.cropRect
        val srcW = crop.width()
        val srcH = crop.height()
        val rot = ((rotationDegrees % 360) + 360) % 360
        val decW = srcW / 2
        val decH = srcH / 2
        val dstW = if (rot == 90 || rot == 270) decH else decW
        val dstH = if (rot == 90 || rot == 270) decW else decH
        // NV21 kräver jämna dimensioner
        val outW = dstW and 0x7FFFFFFE
        val outH = dstH and 0x7FFFFFFE
        val needed = outW * outH * 3 / 2
        var out = convertBuffer
        if (out == null || out.size != needed) {
            out = ByteArray(needed)
            convertBuffer = out
        }

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuf = yPlane.buffer
        val yRow = yPlane.rowStride
        val yPix = yPlane.pixelStride
        val uBuf = uPlane.buffer
        val uRow = uPlane.rowStride
        val uPix = uPlane.pixelStride
        val vBuf = vPlane.buffer
        val vRow = vPlane.rowStride
        val vPix = vPlane.pixelStride

        // Y-plan: för varje dst-pixel, hitta källpixel via invers rotation
        // + 2x decimering (i beskuret källkoordinatrum).
        var offset = 0
        for (dy in 0 until outH) {
            for (dx in 0 until outW) {
                val sxDec: Int
                val syDec: Int
                when (rot) {
                    90 -> { sxDec = dy; syDec = decH - 1 - dx }
                    180 -> { sxDec = decW - 1 - dx; syDec = decH - 1 - dy }
                    270 -> { sxDec = decW - 1 - dy; syDec = dx }
                    else -> { sxDec = dx; syDec = dy }
                }
                val sx = crop.left + sxDec * 2
                val sy = crop.top + syDec * 2
                out[offset++] = yBuf.get(sy * yRow + sx * yPix)
            }
        }

        // Kroma (NV21: VU interleaved, halva upplösningen av dst).
        val chromaW = outW / 2
        val chromaH = outH / 2
        for (dy in 0 until chromaH) {
            for (dx in 0 until chromaW) {
                val fullDx = dx * 2
                val fullDy = dy * 2
                val sxDec: Int
                val syDec: Int
                when (rot) {
                    90 -> { sxDec = fullDy; syDec = decH - 1 - fullDx }
                    180 -> { sxDec = decW - 1 - fullDx; syDec = decH - 1 - fullDy }
                    270 -> { sxDec = decW - 1 - fullDy; syDec = fullDx }
                    else -> { sxDec = fullDx; syDec = fullDy }
                }
                val scx = (crop.left + sxDec * 2) / 2
                val scy = (crop.top + syDec * 2) / 2
                out[offset++] = vBuf.get(scy * vRow + scx * vPix)
                out[offset++] = uBuf.get(scy * uRow + scx * uPix)
            }
        }
        return ConvertedFrame(out, outW, outH)
    }

    /** YUV_420_888 (godtyckliga strides) -> NV21 (Y + interleaved VU),
     *  begränsat till cropRect (avkodare paddar ofta bredd/höjd). */
    private fun yuv420ToNv21(image: Image, out: ByteArray) {
        val crop = image.cropRect
        val width = crop.width()
        val height = crop.height()
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        var offset = 0
        val yBuffer = yPlane.buffer
        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        if (yPixelStride == 1 && yRowStride == width && crop.left == 0 && crop.top == 0) {
            yBuffer.position(0)
            yBuffer.get(out, 0, width * height)
            offset = width * height
        } else {
            for (row in 0 until height) {
                var pos = (crop.top + row) * yRowStride + crop.left * yPixelStride
                for (col in 0 until width) {
                    out[offset++] = yBuffer.get(pos)
                    pos += yPixelStride
                }
            }
        }

        val chromaHeight = height / 2
        val chromaWidth = width / 2
        val chromaTop = crop.top / 2
        val chromaLeft = crop.left / 2
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride
        for (row in 0 until chromaHeight) {
            val vRow = (chromaTop + row) * vRowStride
            val uRow = (chromaTop + row) * uRowStride
            for (col in 0 until chromaWidth) {
                out[offset++] = vBuffer.get(vRow + (chromaLeft + col) * vPixelStride)
                out[offset++] = uBuffer.get(uRow + (chromaLeft + col) * uPixelStride)
            }
        }
    }

    private fun pushPoseFrame(frames: WritableArray, timestampMs: Long, pose: Pose) {
        val landmarks = Arguments.createArray()
        for (landmark in pose.allPoseLandmarks) {
            landmarks.pushMap(Arguments.createMap().apply {
                putInt("type", landmark.landmarkType)
                putDouble("x", landmark.position.x.toDouble())
                putDouble("y", landmark.position.y.toDouble())
                putDouble("z", landmark.position3D.z.toDouble())
                putDouble("visibility", landmark.inFrameLikelihood.toDouble())
            })
        }
        frames.pushMap(Arguments.createMap().apply {
            putDouble("timestamp_ms", timestampMs.toDouble())
            putBoolean("pose_detected", pose.allPoseLandmarks.isNotEmpty())
            putArray("landmarks", landmarks)
        })
    }

    // ── Fallback: gamla retriever-vägen (långsam men robust) ──────────────────

    private fun extractWithRetriever(path: String, sampleFps: Double): com.facebook.react.bridge.WritableMap {
        val retriever = MediaMetadataRetriever()
        val detector = PoseDetection.getClient(
            PoseDetectorOptions.Builder()
                .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
                .build()
        )
        try {
            retriever.setDataSource(path)
            val durationMs = retriever
                .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()
                ?: 0L
            val stepMs = max(1L, (1000.0 / sampleFps).toLong())
            val frames = Arguments.createArray()
            var timestampMs = 0L
            while (timestampMs <= durationMs) {
                val bitmap = retriever.getFrameAtTime(
                    timestampMs * 1000L,
                    MediaMetadataRetriever.OPTION_CLOSEST
                )
                if (bitmap != null) {
                    val image = InputImage.fromBitmap(bitmap, 0)
                    val pose = Tasks.await(detector.process(image))
                    pushPoseFrame(frames, timestampMs, pose)
                    bitmap.recycle()
                }
                timestampMs += stepMs
            }
            return Arguments.createMap().apply {
                putString("video_path", path)
                putDouble("sample_fps", sampleFps)
                putDouble("duration_ms", durationMs.toDouble())
                putDouble("frame_count", frames.size().toDouble())
                putArray("frames", frames)
            }
        } finally {
            detector.close()
            retriever.release()
        }
    }
}
