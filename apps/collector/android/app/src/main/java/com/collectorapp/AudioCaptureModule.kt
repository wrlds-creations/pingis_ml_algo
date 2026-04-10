package com.collectorapp

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Base64
import androidx.core.app.ActivityCompat
import com.facebook.react.bridge.*
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Två funktioner:
 *
 * 1. capture(durationMs) — spelar in ett kort klipp och returnerar base64 PCM.
 *    Används av live-klassificeringen (nu ersatt av AudioStreamModule).
 *
 * 2. startSession(outputPath) / stopSession() — spelar in en lång WAV-session
 *    (22 050 Hz, mono, PCM-16). Används av datainsamlingsskärmen.
 *    WAV-headern skrivs korrekt med rätt data-chunk-storlek vid stop.
 */
class AudioCaptureModule(private val ctx: ReactApplicationContext)
    : ReactContextBaseJavaModule(ctx) {

    override fun getName() = "AudioCapture"

    // ── Session-state ──────────────────────────────────────────────────────────

    @Volatile private var sessionRecord: AudioRecord? = null
    @Volatile private var sessionRunning = false
    @Volatile private var sessionStart   = 0L
    @Volatile private var sessionFos: FileOutputStream? = null
    @Volatile private var sessionPath: String? = null

    // ── capture (befintlig funktion) ───────────────────────────────────────────

    @ReactMethod
    fun capture(durationMs: Int, promise: Promise) {
        if (ActivityCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            promise.reject("NO_PERMISSION", "RECORD_AUDIO permission not granted")
            return
        }

        val sampleRate = 22_050
        val nSamples   = sampleRate * durationMs / 1_000
        val minBuf     = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val bufSize    = maxOf(minBuf, nSamples * 2)

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

    // ── startSession ───────────────────────────────────────────────────────────

    @ReactMethod
    fun startSession(outputPath: String, promise: Promise) {
        if (sessionRunning) {
            promise.reject("SESSION_ACTIVE", "A session is already running")
            return
        }
        if (ActivityCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            promise.reject("NO_PERMISSION", "RECORD_AUDIO permission not granted")
            return
        }

        val sr      = 22_050
        val minBuf  = AudioRecord.getMinBufferSize(
            sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val bufSize = maxOf(minBuf, sr / 5 * 2) // ≥ 200ms buffer

        try {
            val file = File(outputPath)
            file.parentFile?.mkdirs()
            val fos = FileOutputStream(file)
            writeWavHeader(fos, sr, 0) // placeholder — size patched in stopSession

            val rec = AudioRecord(
                MediaRecorder.AudioSource.MIC, sr,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufSize)

            sessionRecord  = rec
            sessionFos     = fos
            sessionRunning = true
            sessionStart   = System.currentTimeMillis()
            sessionPath    = outputPath

            Thread {
                val buf = ShortArray(sr / 10) // 100ms chunks
                rec.startRecording()
                val bytesBuf = ByteArray(buf.size * 2)
                while (sessionRunning) {
                    val read = rec.read(buf, 0, buf.size)
                    if (read > 0) {
                        // interleave shorts → bytes LE
                        val bb = ByteBuffer.wrap(bytesBuf).order(ByteOrder.LITTLE_ENDIAN)
                        for (i in 0 until read) bb.putShort(buf[i])
                        fos.write(bytesBuf, 0, read * 2)
                    }
                }
            }.start()

            promise.resolve("started")
        } catch (e: Exception) {
            promise.reject("SESSION_ERROR", e.message)
        }
    }

    // ── stopSession ────────────────────────────────────────────────────────────

    @ReactMethod
    fun stopSession(promise: Promise) {
        if (!sessionRunning) {
            promise.reject("NO_SESSION", "No session is running")
            return
        }
        sessionRunning = false
        val durationMs = System.currentTimeMillis() - sessionStart

        Thread {
            try {
                sessionRecord?.stop()
                sessionRecord?.release()
                sessionRecord = null

                sessionFos?.flush()
                sessionFos?.close()
                sessionFos = null

                // Patch WAV header with correct data chunk size
                sessionPath?.let { path ->
                    val file = File(path)
                    if (file.exists()) {
                        val dataSize = (file.length() - 44).toInt()
                        RandomAccessFile(file, "rw").use { raf ->
                            // RIFF chunk size = dataSize + 36
                            raf.seek(4)
                            raf.write(intToLEBytes(dataSize + 36))
                            // data chunk size
                            raf.seek(40)
                            raf.write(intToLEBytes(dataSize))
                        }
                    }
                }
                sessionPath = null
                promise.resolve(durationMs.toDouble())
            } catch (e: Exception) {
                promise.reject("STOP_ERROR", e.message)
            }
        }.start()
    }

    // ── WAV header helpers ─────────────────────────────────────────────────────

    private fun writeWavHeader(fos: FileOutputStream, sr: Int, dataBytes: Int) {
        val byteRate    = sr * 2       // mono 16-bit
        val blockAlign  = 2
        val bitsPerSample = 16

        val bb = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        // RIFF
        bb.put("RIFF".toByteArray())
        bb.putInt(dataBytes + 36)
        bb.put("WAVE".toByteArray())
        // fmt
        bb.put("fmt ".toByteArray())
        bb.putInt(16)              // sub-chunk size
        bb.putShort(1)             // PCM
        bb.putShort(1)             // mono
        bb.putInt(sr)
        bb.putInt(byteRate)
        bb.putShort(blockAlign.toShort())
        bb.putShort(bitsPerSample.toShort())
        // data
        bb.put("data".toByteArray())
        bb.putInt(dataBytes)

        fos.write(bb.array())
    }

    private fun intToLEBytes(v: Int): ByteArray =
        ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array()
}
