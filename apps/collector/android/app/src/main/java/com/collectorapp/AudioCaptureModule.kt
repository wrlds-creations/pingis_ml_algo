package com.collectorapp

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaRecorder
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Base64
import androidx.core.app.ActivityCompat
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.BaseActivityEventListener
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.modules.core.DeviceEventManagerModule
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.floor
import kotlin.math.roundToInt

/**
 * Två funktioner:
 *
 * 1. capture(durationMs) — spelar in ett kort klipp och returnerar base64 PCM.
 * 2. startSession(outputPath, targetDurationMs) / stopSession() — spelar in en lång WAV-session
 *    (22 050 Hz, mono, PCM-16) för guided audio collection.
 *
 * Sessioninspelningen stoppas nu native när mål-längden nås, och stopSession väntar
 * in worker-tråden innan WAV-headern patchas och duration returneras.
 */
class AudioCaptureModule(private val ctx: ReactApplicationContext)
    : ReactContextBaseJavaModule(ctx) {

    override fun getName() = "AudioCapture"

    companion object {
        private const val TARGET_SAMPLE_RATE = 22_050
        private const val IMPORT_AUDIO_REQUEST = 7042
    }

    @Volatile private var sessionRecord: AudioRecord? = null
    @Volatile private var sessionRunning = false
    @Volatile private var sessionStopRequested = false
    @Volatile private var sessionStart = 0L
    @Volatile private var sessionTargetDurationMs = 0L
    @Volatile private var sessionCompletedDurationMs: Long? = null
    @Volatile private var sessionFos: FileOutputStream? = null
    @Volatile private var sessionPath: String? = null
    @Volatile private var sessionThread: Thread? = null
    @Volatile private var sessionWrittenSamples = 0L
    @Volatile private var sessionSampleRate = TARGET_SAMPLE_RATE
    @Volatile private var sessionAutoStopped = false
    @Volatile private var pendingImportPromise: Promise? = null
    @Volatile private var pendingImportOutputPath: String? = null

    private data class ImportedAudioResult(
        val displayName: String?,
        val sourceUri: String,
        val durationMs: Long,
        val sourceSampleRate: Int,
        val sourceChannels: Int,
        val writtenSamples: Int,
    )

    private val activityEventListener = object : BaseActivityEventListener() {
        override fun onActivityResult(activity: Activity, requestCode: Int, resultCode: Int, data: Intent?) {
            if (requestCode != IMPORT_AUDIO_REQUEST) return

            val promise = pendingImportPromise
            val outputPath = pendingImportOutputPath
            pendingImportPromise = null
            pendingImportOutputPath = null

            if (promise == null || outputPath == null) return
            if (resultCode != Activity.RESULT_OK) {
                promise.reject("IMPORT_CANCELLED", "Audio import cancelled")
                return
            }

            val uri = data?.data
            if (uri == null) {
                promise.reject("IMPORT_NO_URI", "No audio file selected")
                return
            }

            try {
                val flags = data.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION
                ctx.contentResolver.takePersistableUriPermission(uri, flags)
            } catch (_: Exception) {
            }

            Thread {
                try {
                    val result = decodeAudioUriToWav(uri, outputPath)
                    val payload = Arguments.createMap().apply {
                        putString("outputPath", outputPath)
                        putString("displayName", result.displayName)
                        putString("sourceUri", result.sourceUri)
                        putDouble("durationMs", result.durationMs.toDouble())
                        putDouble("sampleRate", TARGET_SAMPLE_RATE.toDouble())
                        putDouble("sourceSampleRate", result.sourceSampleRate.toDouble())
                        putDouble("channels", result.sourceChannels.toDouble())
                        putDouble("writtenSamples", result.writtenSamples.toDouble())
                    }
                    promise.resolve(payload)
                } catch (e: Exception) {
                    promise.reject("IMPORT_AUDIO_ERROR", e.message, e)
                }
            }.start()
        }
    }

    init {
        ctx.addActivityEventListener(activityEventListener)
    }

    @ReactMethod
    fun capture(durationMs: Int, promise: Promise) {
        if (ActivityCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            promise.reject("NO_PERMISSION", "RECORD_AUDIO permission not granted")
            return
        }

        val sampleRate = TARGET_SAMPLE_RATE
        val nSamples = sampleRate * durationMs / 1_000
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val bufSize = maxOf(minBuf, nSamples * 2)

        Thread {
            val rec = AudioRecord(
                MediaRecorder.AudioSource.MIC, sampleRate,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufSize)
            try {
                val samples = ShortArray(nSamples)
                rec.startRecording()
                var offset = 0
                while (offset < nSamples) {
                    val read = rec.read(samples, offset, nSamples - offset)
                    if (read <= 0) break
                    offset += read
                }
                rec.stop()
                val bytes = ByteBuffer.allocate(nSamples * 2)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .apply { samples.forEach { putShort(it) } }
                    .array()
                promise.resolve(Base64.encodeToString(bytes, Base64.NO_WRAP))
            } catch (e: Exception) {
                promise.reject("CAPTURE_ERROR", e.message)
            } finally {
                rec.release()
            }
        }.start()
    }

    @ReactMethod
    fun addListener(eventName: String) {
        // Required for NativeEventEmitter on Android.
    }

    @ReactMethod
    fun removeListeners(count: Int) {
        // Required for NativeEventEmitter on Android.
    }

    @ReactMethod
    fun importAudioFile(outputPath: String, promise: Promise) {
        val activity = ctx.currentActivity
        if (activity == null) {
            promise.reject("NO_ACTIVITY", "No active Android activity")
            return
        }
        if (pendingImportPromise != null) {
            promise.reject("IMPORT_ACTIVE", "An audio import is already active")
            return
        }

        pendingImportPromise = promise
        pendingImportOutputPath = outputPath

        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "audio/*"
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        }

        try {
            activity.startActivityForResult(intent, IMPORT_AUDIO_REQUEST)
        } catch (e: Exception) {
            pendingImportPromise = null
            pendingImportOutputPath = null
            promise.reject("IMPORT_PICKER_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun extractAudioFromVideoFile(inputPath: String, outputPath: String, promise: Promise) {
        Thread {
            try {
                val result = decodeAudioFileToWav(inputPath, outputPath)
                val payload = Arguments.createMap().apply {
                    putString("outputPath", outputPath)
                    putString("displayName", result.displayName)
                    putString("sourceUri", result.sourceUri)
                    putDouble("durationMs", result.durationMs.toDouble())
                    putDouble("sampleRate", TARGET_SAMPLE_RATE.toDouble())
                    putDouble("sourceSampleRate", result.sourceSampleRate.toDouble())
                    putDouble("channels", result.sourceChannels.toDouble())
                    putDouble("writtenSamples", result.writtenSamples.toDouble())
                }
                promise.resolve(payload)
            } catch (e: Exception) {
                promise.reject("EXTRACT_AUDIO_ERROR", e.message, e)
            }
        }.start()
    }

    @ReactMethod
    fun startSession(outputPath: String, targetDurationMs: Int, promise: Promise) {
        if (sessionRunning) {
            promise.reject("SESSION_ACTIVE", "A session is already running")
            return
        }
        if (ActivityCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            promise.reject("NO_PERMISSION", "RECORD_AUDIO permission not granted")
            return
        }

        val sr = TARGET_SAMPLE_RATE
        val minBuf = AudioRecord.getMinBufferSize(
            sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val bufSize = maxOf(minBuf, sr / 5 * 2)

        try {
            val file = File(outputPath)
            file.parentFile?.mkdirs()
            val fos = FileOutputStream(file)
            writeWavHeader(fos, sr, 0)

            val rec = AudioRecord(
                MediaRecorder.AudioSource.MIC, sr,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufSize)

            sessionRecord = rec
            sessionFos = fos
            sessionRunning = true
            sessionStopRequested = false
            sessionStart = System.currentTimeMillis()
            sessionTargetDurationMs = targetDurationMs.toLong()
            sessionCompletedDurationMs = null
            sessionPath = outputPath
            sessionWrittenSamples = 0L
            sessionSampleRate = sr
            sessionAutoStopped = false

            val worker = Thread {
                runSessionLoop(rec, fos, sr)
            }
            sessionThread = worker
            worker.start()

            promise.resolve("started")
        } catch (e: Exception) {
            sessionRunning = false
            sessionStopRequested = false
            sessionRecord = null
            sessionFos = null
            sessionPath = null
            promise.reject("SESSION_ERROR", e.message)
        }
    }

    @ReactMethod
    fun stopSession(promise: Promise) {
        val completed = sessionCompletedDurationMs
        if (!sessionRunning && completed != null) {
            promise.resolve(completed.toDouble())
            return
        }
        if (!sessionRunning) {
            promise.reject("NO_SESSION", "No session is running")
            return
        }

        sessionStopRequested = true

        Thread {
            try {
                sessionThread?.join(2_000)
                val duration = sessionCompletedDurationMs ?: (System.currentTimeMillis() - sessionStart)
                promise.resolve(duration.toDouble())
            } catch (e: Exception) {
                promise.reject("STOP_ERROR", e.message)
            }
        }.start()
    }

    private fun runSessionLoop(rec: AudioRecord, fos: FileOutputStream, sr: Int) {
        val buf = ShortArray(sr / 10)
        val bytesBuf = ByteArray(buf.size * 2)
        val targetSamples = if (sessionTargetDurationMs > 0L) {
            sessionTargetDurationMs * sr / 1000L
        } else {
            Long.MAX_VALUE
        }

        try {
            rec.startRecording()
            while (!sessionStopRequested) {
                val read = rec.read(buf, 0, buf.size)
                if (read <= 0) continue

                val bb = ByteBuffer.wrap(bytesBuf).order(ByteOrder.LITTLE_ENDIAN)
                for (i in 0 until read) bb.putShort(buf[i])
                fos.write(bytesBuf, 0, read * 2)
                sessionWrittenSamples += read.toLong()

                if (sessionWrittenSamples >= targetSamples) {
                    sessionAutoStopped = sessionTargetDurationMs > 0L
                    sessionStopRequested = true
                }
            }
        } catch (_: Exception) {
            // finaliseringen hanterar redan delvis inspelade filer
        } finally {
            finalizeSession()
        }
    }

    private fun finalizeSession() {
        val finishedPath = sessionPath
        try {
            sessionRecord?.stop()
        } catch (_: Exception) {
        }
        try {
            sessionRecord?.release()
        } catch (_: Exception) {
        }
        sessionRecord = null

        try {
            sessionFos?.flush()
            sessionFos?.close()
        } catch (_: Exception) {
        }
        sessionFos = null

        finishedPath?.let { path ->
            val file = File(path)
            if (file.exists()) {
                try {
                    val dataSize = (file.length() - 44).toInt().coerceAtLeast(0)
                    RandomAccessFile(file, "rw").use { raf ->
                        raf.seek(4)
                        raf.write(intToLEBytes(dataSize + 36))
                        raf.seek(40)
                        raf.write(intToLEBytes(dataSize))
                    }
                } catch (_: Exception) {
                }
            }
        }

        val durationMs = if (sessionSampleRate > 0) {
            sessionWrittenSamples * 1000L / sessionSampleRate
        } else {
            System.currentTimeMillis() - sessionStart
        }
        sessionCompletedDurationMs = durationMs
        sessionPath = null
        sessionThread = null
        sessionRunning = false
        sessionStopRequested = false
        val writtenSamples = sessionWrittenSamples
        val autoStopped = sessionAutoStopped
        sessionWrittenSamples = 0L
        sessionAutoStopped = false

        if (autoStopped && finishedPath != null && ctx.hasActiveCatalystInstance()) {
            val payload = Arguments.createMap().apply {
                putString("outputPath", finishedPath)
                putDouble("durationMs", durationMs.toDouble())
                putDouble("writtenSamples", writtenSamples.toDouble())
            }
            ctx
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit("onAudioSessionStopped", payload)
        }
    }

    private fun writeWavHeader(fos: FileOutputStream, sr: Int, dataBytes: Int) {
        val byteRate = sr * 2
        val blockAlign = 2
        val bitsPerSample = 16

        val bb = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray())
        bb.putInt(dataBytes + 36)
        bb.put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray())
        bb.putInt(16)
        bb.putShort(1)
        bb.putShort(1)
        bb.putInt(sr)
        bb.putInt(byteRate)
        bb.putShort(blockAlign.toShort())
        bb.putShort(bitsPerSample.toShort())
        bb.put("data".toByteArray())
        bb.putInt(dataBytes)

        fos.write(bb.array())
    }

    private fun intToLEBytes(v: Int): ByteArray =
        ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array()

    private fun decodeAudioUriToWav(uri: Uri, outputPath: String): ImportedAudioResult =
        decodeAudioSourceToWav(
            displayName = displayNameForUri(uri),
            sourceUri = uri.toString(),
            outputPath = outputPath,
        ) { extractor ->
            extractor.setDataSource(ctx, uri, null)
        }

    private fun decodeAudioFileToWav(inputPath: String, outputPath: String): ImportedAudioResult {
        val cleanPath = inputPath.removePrefix("file://")
        val inputFile = File(cleanPath)
        if (!inputFile.exists()) {
            throw IllegalArgumentException("Video file does not exist: $cleanPath")
        }
        return decodeAudioSourceToWav(
            displayName = inputFile.name,
            sourceUri = inputFile.absolutePath,
            outputPath = outputPath,
        ) { extractor ->
            extractor.setDataSource(inputFile.absolutePath)
        }
    }

    private fun decodeAudioSourceToWav(
        displayName: String?,
        sourceUri: String,
        outputPath: String,
        configureExtractor: (MediaExtractor) -> Unit,
    ): ImportedAudioResult {
        val extractor = MediaExtractor()
        var decoder: MediaCodec? = null
        val decodedChunks = mutableListOf<ShortArray>()
        var decodedSampleCount = 0L
        var outputSampleRate = 0
        var outputChannels = 1
        var outputEncoding = AudioFormat.ENCODING_PCM_16BIT

        try {
            configureExtractor(extractor)
            var audioTrackIndex = -1
            var inputFormat: MediaFormat? = null

            for (index in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(index)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("audio/")) {
                    audioTrackIndex = index
                    inputFormat = format
                    break
                }
            }

            val format = inputFormat ?: throw IllegalArgumentException("No audio track found")
            val mime = format.getString(MediaFormat.KEY_MIME)
                ?: throw IllegalArgumentException("Audio track is missing MIME type")
            outputSampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            outputChannels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT).coerceAtLeast(1)

            extractor.selectTrack(audioTrackIndex)
            val activeDecoder = MediaCodec.createDecoderByType(mime)
            decoder = activeDecoder
            activeDecoder.configure(format, null, null, 0)
            activeDecoder.start()

            val bufferInfo = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false

            while (!outputDone) {
                if (!inputDone) {
                    val inputIndex = activeDecoder.dequeueInputBuffer(10_000)
                    if (inputIndex >= 0) {
                        val inputBuffer = activeDecoder.getInputBuffer(inputIndex)
                        inputBuffer?.clear()
                        val sampleSize = if (inputBuffer != null) {
                            extractor.readSampleData(inputBuffer, 0)
                        } else {
                            -1
                        }

                        if (sampleSize < 0) {
                            activeDecoder.queueInputBuffer(
                                inputIndex,
                                0,
                                0,
                                0L,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                            )
                            inputDone = true
                        } else {
                            activeDecoder.queueInputBuffer(
                                inputIndex,
                                0,
                                sampleSize,
                                extractor.sampleTime.coerceAtLeast(0L),
                                0,
                            )
                            extractor.advance()
                        }
                    }
                }

                when (val outputIndex = activeDecoder.dequeueOutputBuffer(bufferInfo, 10_000)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    }
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val outputFormat = activeDecoder.outputFormat
                        outputSampleRate = outputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        outputChannels = outputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT).coerceAtLeast(1)
                        outputEncoding = if (outputFormat.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                            outputFormat.getInteger(MediaFormat.KEY_PCM_ENCODING)
                        } else {
                            AudioFormat.ENCODING_PCM_16BIT
                        }
                    }
                    else -> {
                        if (outputIndex >= 0) {
                            val outputBuffer = activeDecoder.getOutputBuffer(outputIndex)
                            if (outputBuffer != null && bufferInfo.size > 0) {
                                outputBuffer.position(bufferInfo.offset)
                                outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                                val monoSamples = pcmBufferToMonoShorts(
                                    outputBuffer.slice(),
                                    outputChannels,
                                    outputEncoding,
                                )
                                if (monoSamples.isNotEmpty()) {
                                    decodedChunks.add(monoSamples)
                                    decodedSampleCount += monoSamples.size.toLong()
                                }
                            }
                            if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                                outputDone = true
                            }
                            activeDecoder.releaseOutputBuffer(outputIndex, false)
                        }
                    }
                }
            }
        } finally {
            try { decoder?.stop() } catch (_: Exception) {}
            try { decoder?.release() } catch (_: Exception) {}
            try { extractor.release() } catch (_: Exception) {}
        }

        if (decodedSampleCount <= 0 || outputSampleRate <= 0) {
            throw IllegalArgumentException("Selected audio file decoded to no samples")
        }

        val resampled = resampleMono(decodedChunks, decodedSampleCount, outputSampleRate, TARGET_SAMPLE_RATE)
        writeWavFile(outputPath, TARGET_SAMPLE_RATE, resampled)

        return ImportedAudioResult(
            displayName = displayName,
            sourceUri = sourceUri,
            durationMs = resampled.size * 1000L / TARGET_SAMPLE_RATE,
            sourceSampleRate = outputSampleRate,
            sourceChannels = outputChannels,
            writtenSamples = resampled.size,
        )
    }

    private fun pcmBufferToMonoShorts(buffer: ByteBuffer, channels: Int, encoding: Int): ShortArray {
        val safeChannels = channels.coerceAtLeast(1)
        val ordered = buffer.order(ByteOrder.LITTLE_ENDIAN)

        if (encoding == AudioFormat.ENCODING_PCM_FLOAT) {
            val frameCount = ordered.remaining() / (4 * safeChannels)
            val output = ShortArray(frameCount)
            for (frame in 0 until frameCount) {
                var sum = 0.0
                for (channel in 0 until safeChannels) {
                    sum += ordered.float.toDouble()
                }
                val mono = (sum / safeChannels).coerceIn(-1.0, 1.0)
                output[frame] = (mono * 32767.0).roundToInt().toShort()
            }
            return output
        }

        val frameCount = ordered.remaining() / (2 * safeChannels)
        val output = ShortArray(frameCount)
        for (frame in 0 until frameCount) {
            var sum = 0
            for (channel in 0 until safeChannels) {
                sum += ordered.short.toInt()
            }
            output[frame] = (sum / safeChannels)
                .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                .toShort()
        }
        return output
    }

    private fun resampleMono(
        chunks: List<ShortArray>,
        sampleCount: Long,
        sourceRate: Int,
        targetRate: Int,
    ): ShortArray {
        if (sampleCount > Int.MAX_VALUE) {
            throw IllegalArgumentException("Audio file is too long to import in one pass")
        }

        val source = ShortArray(sampleCount.toInt())
        var offset = 0
        for (chunk in chunks) {
            chunk.copyInto(source, offset)
            offset += chunk.size
        }

        if (sourceRate == targetRate) return source

        val outputLength = ((sampleCount.toDouble() * targetRate.toDouble()) / sourceRate.toDouble())
            .roundToInt()
            .coerceAtLeast(1)
        val output = ShortArray(outputLength)
        val step = sourceRate.toDouble() / targetRate.toDouble()

        for (index in output.indices) {
            val sourcePos = index * step
            val leftIndex = floor(sourcePos).toInt().coerceIn(0, source.lastIndex)
            val rightIndex = (leftIndex + 1).coerceAtMost(source.lastIndex)
            val fraction = sourcePos - leftIndex
            val sample = source[leftIndex] * (1.0 - fraction) + source[rightIndex] * fraction
            output[index] = sample.roundToInt()
                .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                .toShort()
        }

        return output
    }

    private fun writeWavFile(outputPath: String, sampleRate: Int, samples: ShortArray) {
        val file = File(outputPath)
        file.parentFile?.mkdirs()
        FileOutputStream(file).use { fos ->
            writeWavHeader(fos, sampleRate, samples.size * 2)
            val byteBuffer = ByteBuffer.allocate(8192 * 2).order(ByteOrder.LITTLE_ENDIAN)
            var index = 0
            while (index < samples.size) {
                byteBuffer.clear()
                val end = minOf(samples.size, index + 8192)
                for (sampleIndex in index until end) {
                    byteBuffer.putShort(samples[sampleIndex])
                }
                fos.write(byteBuffer.array(), 0, (end - index) * 2)
                index = end
            }
        }
    }

    private fun displayNameForUri(uri: Uri): String? {
        try {
            ctx.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (index >= 0) return cursor.getString(index)
                }
            }
        } catch (_: Exception) {
        }
        return uri.lastPathSegment
    }
}
