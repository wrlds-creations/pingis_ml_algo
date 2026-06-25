package com.collectorapp

import android.app.Activity
import android.content.Intent
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.net.Uri
import android.provider.OpenableColumns
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.BaseActivityEventListener
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

class VideoSegmentModule(private val ctx: ReactApplicationContext)
    : ReactContextBaseJavaModule(ctx) {

    override fun getName() = "VideoSegment"

    companion object {
        private const val IMPORT_VIDEO_REQUEST = 7043
    }

    @Volatile private var pendingImportPromise: Promise? = null
    @Volatile private var pendingImportOutputPath: String? = null

    private val activityEventListener = object : BaseActivityEventListener() {
        override fun onActivityResult(activity: Activity, requestCode: Int, resultCode: Int, data: Intent?) {
            if (requestCode != IMPORT_VIDEO_REQUEST) return

            val promise = pendingImportPromise
            val outputPath = pendingImportOutputPath
            pendingImportPromise = null
            pendingImportOutputPath = null

            if (promise == null || outputPath == null) return
            if (resultCode != Activity.RESULT_OK) {
                promise.reject("IMPORT_CANCELLED", "Video import cancelled")
                return
            }

            val uri = data?.data
            if (uri == null) {
                promise.reject("IMPORT_NO_URI", "No video file selected")
                return
            }

            try {
                val flags = data.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION
                ctx.contentResolver.takePersistableUriPermission(uri, flags)
            } catch (_: Exception) {
            }

            Thread {
                try {
                    val result = copyVideoUri(uri, outputPath)
                    promise.resolve(result)
                } catch (error: Exception) {
                    promise.reject("IMPORT_VIDEO_ERROR", error.message, error)
                }
            }.start()
        }
    }

    init {
        ctx.addActivityEventListener(activityEventListener)
    }

    @ReactMethod
    fun importVideoFile(outputPath: String, promise: Promise) {
        val activity = ctx.currentActivity
        if (activity == null) {
            promise.reject("NO_ACTIVITY", "No active Android activity")
            return
        }
        if (pendingImportPromise != null) {
            promise.reject("IMPORT_ACTIVE", "A video import is already active")
            return
        }

        pendingImportPromise = promise
        pendingImportOutputPath = outputPath.removePrefix("file://")

        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "video/*"
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        }

        try {
            activity.startActivityForResult(intent, IMPORT_VIDEO_REQUEST)
        } catch (error: Exception) {
            pendingImportPromise = null
            pendingImportOutputPath = null
            promise.reject("IMPORT_PICKER_ERROR", error.message, error)
        }
    }

    @ReactMethod
    fun splitVideo(
        videoPath: String,
        outputDir: String,
        filenamePrefix: String,
        segmentDurationMs: Double,
        startIndex: Double,
        promise: Promise
    ) {
        Thread {
            val cleanPath = videoPath.removePrefix("file://")
            val inputFile = File(cleanPath)
            if (!inputFile.exists()) {
                promise.reject("VIDEO_NOT_FOUND", "Video file does not exist: $cleanPath")
                return@Thread
            }

            val cleanOutputDir = outputDir.removePrefix("file://")
            val outputDirectory = File(cleanOutputDir)
            if (!outputDirectory.exists() && !outputDirectory.mkdirs()) {
                promise.reject("OUTPUT_DIR_ERROR", "Could not create output directory: $cleanOutputDir")
                return@Thread
            }

            val safeSegmentMs = max(1L, segmentDurationMs.toLong())
            val firstTakeIndex = max(1, startIndex.toInt())
            val retriever = MediaMetadataRetriever()

            try {
                retriever.setDataSource(cleanPath)
                val durationMs = retriever
                    .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                    ?.toLongOrNull()
                    ?: 0L
                val rotation = retriever
                    .extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                    ?.toIntOrNull()
                    ?: 0
                if (durationMs <= 0L) {
                    promise.reject("VIDEO_DURATION_ERROR", "Could not read video duration: $cleanPath")
                    return@Thread
                }

                val segmentCount = max(1, ceil(durationMs.toDouble() / safeSegmentMs.toDouble()).toInt())
                val segments = Arguments.createArray()

                for (segmentIndex in 0 until segmentCount) {
                    val startMs = segmentIndex * safeSegmentMs
                    val endMs = min(durationMs, startMs + safeSegmentMs)
                    val takeIndex = firstTakeIndex + segmentIndex
                    val filename = "${filenamePrefix}_${takeIndex.toString().padStart(3, '0')}.mp4"
                    val outputFile = File(outputDirectory, filename)
                    if (outputFile.exists()) outputFile.delete()

                    val writtenDurationMs = writeSegment(
                        inputPath = cleanPath,
                        outputPath = outputFile.absolutePath,
                        startUs = startMs * 1000L,
                        endUs = endMs * 1000L,
                        rotation = rotation
                    )

                    segments.pushMap(Arguments.createMap().apply {
                        putString("video_filename", filename)
                        putString("video_path", outputFile.absolutePath)
                        putInt("take_index", takeIndex)
                        putDouble("start_ms", startMs.toDouble())
                        putDouble("end_ms", endMs.toDouble())
                        putDouble("duration_ms", max(0L, writtenDurationMs).toDouble())
                    })
                }

                promise.resolve(segments)
            } catch (error: Exception) {
                promise.reject("VIDEO_SEGMENT_ERROR", error.message, error)
            } finally {
                retriever.release()
            }
        }.start()
    }

    private fun writeSegment(
        inputPath: String,
        outputPath: String,
        startUs: Long,
        endUs: Long,
        rotation: Int
    ): Long {
        val extractor = MediaExtractor()
        var muxer: MediaMuxer? = null
        var muxerStarted = false

        try {
            extractor.setDataSource(inputPath)
            val trackIndexMap = mutableMapOf<Int, Int>()
            var maxInputSize = 1024 * 1024

            muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            if (rotation != 0) {
                muxer.setOrientationHint(rotation)
            }

            for (trackIndex in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(trackIndex)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
                if (!mime.startsWith("video/") && !mime.startsWith("audio/")) continue
                val muxerTrackIndex = muxer.addTrack(format)
                trackIndexMap[trackIndex] = muxerTrackIndex
                extractor.selectTrack(trackIndex)
                if (format.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)) {
                    maxInputSize = max(maxInputSize, format.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE))
                }
            }

            if (trackIndexMap.isEmpty()) {
                throw IllegalStateException("No audio/video tracks found in $inputPath")
            }

            muxer.start()
            muxerStarted = true
            extractor.seekTo(startUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)

            val buffer = ByteBuffer.allocate(maxInputSize)
            val bufferInfo = MediaCodec.BufferInfo()
            var firstWrittenUs = -1L
            var lastWrittenUs = -1L
            var samplesWritten = 0

            while (true) {
                val sampleTimeUs = extractor.sampleTime
                if (sampleTimeUs < 0 || sampleTimeUs > endUs) break
                val sourceTrackIndex = extractor.sampleTrackIndex
                if (sourceTrackIndex < 0) break
                val muxerTrackIndex = trackIndexMap[sourceTrackIndex]
                if (muxerTrackIndex == null) {
                    extractor.advance()
                    continue
                }

                buffer.clear()
                val sampleSize = extractor.readSampleData(buffer, 0)
                if (sampleSize < 0) break
                if (firstWrittenUs < 0) firstWrittenUs = sampleTimeUs
                bufferInfo.set(
                    0,
                    sampleSize,
                    max(0L, sampleTimeUs - firstWrittenUs),
                    extractor.sampleFlags
                )
                muxer.writeSampleData(muxerTrackIndex, buffer, bufferInfo)
                lastWrittenUs = sampleTimeUs
                samplesWritten += 1
                extractor.advance()
            }

            if (samplesWritten == 0) {
                throw IllegalStateException("No samples written for segment $outputPath")
            }

            return max(0L, (lastWrittenUs - firstWrittenUs) / 1000L)
        } finally {
            try {
                if (muxerStarted) muxer?.stop()
            } catch (_: Exception) {
            }
            muxer?.release()
            extractor.release()
        }
    }

    private fun copyVideoUri(uri: Uri, outputPath: String) = Arguments.createMap().apply {
        val outputFile = File(outputPath)
        outputFile.parentFile?.mkdirs()
        ctx.contentResolver.openInputStream(uri).use { input ->
            if (input == null) {
                throw IllegalStateException("Could not open selected video")
            }
            FileOutputStream(outputFile).use { output ->
                input.copyTo(output, 1024 * 1024)
            }
        }

        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(ctx, uri)
            val durationMs = retriever
                .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()
                ?: 0L
            val rotation = retriever
                .extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                ?.toIntOrNull()
                ?: 0
            putString("outputPath", outputFile.absolutePath)
            putString("displayName", displayNameForUri(uri))
            putString("sourceUri", uri.toString())
            putDouble("durationMs", durationMs.toDouble())
            putDouble("rotation", rotation.toDouble())
            putDouble("sizeBytes", outputFile.length().toDouble())
        } finally {
            retriever.release()
        }
    }

    private fun displayNameForUri(uri: Uri): String? {
        ctx.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null).use { cursor ->
            if (cursor != null && cursor.moveToFirst()) {
                val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (index >= 0) return cursor.getString(index)
            }
        }
        return null
    }
}
