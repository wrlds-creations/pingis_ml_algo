package com.collectorapp

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Base64
import androidx.core.app.ActivityCompat
import com.facebook.react.bridge.Arguments
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
    @Volatile private var sessionSampleRate = 22_050
    @Volatile private var sessionAutoStopped = false

    @ReactMethod
    fun capture(durationMs: Int, promise: Promise) {
        if (ActivityCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            promise.reject("NO_PERMISSION", "RECORD_AUDIO permission not granted")
            return
        }

        val sampleRate = 22_050
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

        val sr = 22_050
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
}
