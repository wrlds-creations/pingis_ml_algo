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
        Thread {
            val cleanPath = videoPath.removePrefix("file://")
            if (!File(cleanPath).exists()) {
                promise.reject("VIDEO_NOT_FOUND", "Video file does not exist: $cleanPath")
                return@Thread
            }
            val safeSampleFps = max(1.0, sampleFps)
            try {
                val result = try {
                    extractWithMediaCodec(cleanPath, safeSampleFps)
                } catch (codecError: Exception) {
                    extractWithRetriever(cleanPath, safeSampleFps)
                }
                promise.resolve(result)
            } catch (error: Exception) {
                promise.reject("VIDEO_POSE_ERROR", error.message, error)
            }
        }.start()
    }

    // ── Snabb väg: sekventiell MediaCodec-avkodning ───────────────────────────

    private fun extractWithMediaCodec(path: String, sampleFps: Double): com.facebook.react.bridge.WritableMap {
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
            var nv21: ByteArray? = null

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
                    val shouldSample = bufferInfo.size > 0 && ptsMs >= nextTargetMs
                    if (shouldSample) {
                        val image = codec.getOutputImage(outIndex)
                        if (image != null) {
                            // Avkodare paddar ofta till 16-alignment (1080 -> 1088);
                            // cropRect anger den giltiga regionen.
                            val crop = image.cropRect
                            val width = crop.width()
                            val height = crop.height()
                            if (nv21 == null || nv21!!.size != width * height * 3 / 2) {
                                nv21 = ByteArray(width * height * 3 / 2)
                            }
                            yuv420ToNv21(image, nv21!!)
                            val input = InputImage.fromByteArray(
                                nv21!!, width, height, rotation, InputImage.IMAGE_FORMAT_NV21
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
