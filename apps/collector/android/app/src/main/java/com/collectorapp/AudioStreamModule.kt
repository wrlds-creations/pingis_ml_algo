package com.collectorapp

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Base64
import androidx.core.app.ActivityCompat
import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Kontinuerlig ljud-streaming med ADAPTIV onset-detektion.
 *
 * Istället för en fast energitröskel jämförs varje frames RMS mot ett
 * rullande medelvärde av de senaste 500ms (bakgrundsnivån). En studs
 * triggar när current_RMS > background_avg * multiplier.
 *
 * Det innebär att detektionen fungerar lika bra i ett tyst rum som i
 * ett rum med prat/musik — studsen är alltid relativt starkare än bakgrunden.
 */
class AudioStreamModule(private val ctx: ReactApplicationContext)
    : ReactContextBaseJavaModule(ctx) {

    override fun getName() = "AudioStream"

    companion object {
        const val SR            = 22_050
        const val FRAME_SIZE    = 220           // 10 ms
        const val PRE_SAMPLES   = 6_615         // 300 ms
        const val POST_SAMPLES  = 15_435        // 700 ms
        const val CLIP_SAMPLES  = SR            // 22 050 = 1 s
        const val RING_SIZE     = SR * 3        // 3 s cirkulär buffer
        const val RETRIGGER_MS  = 150L          // min ms mellan onsets

        // Adaptiv tröskel: hur många gånger starkare än bakgrunden en studs måste vara
        const val ONSET_RATIO   = 2.5

        // Bakgrundsestimat: rullande medel över de senaste N frames (500ms = 50 frames)
        const val BG_FRAMES     = 50

        // Absolut minimumtröskel — ignorerar extremt tysta rum (undviker noise floor)
        const val ABS_MIN_RMS   = 0.005

        const val EVENT_NAME    = "onBounceDetected"
    }

    @Volatile private var isRunning     = false
    @Volatile private var lastOnsetTime = 0L

    // threshold-variabeln används nu som ONSET_RATIO-multiplier från slidern
    // (slider-värdet 0.005–0.15 mappas om till ratio 1.5–4.0 i startStreaming)
    @Volatile private var onsetRatio    = ONSET_RATIO

    private val ring = ShortArray(RING_SIZE)
    @Volatile private var writePos = 0

    // Cirkulär buffer för bakgrundsestimat
    private val bgBuffer = DoubleArray(BG_FRAMES)
    private var bgIdx    = 0
    private var bgFilled = false

    // ── ReactMethods ───────────────────────────────────────────────────────────

    @ReactMethod
    fun startStreaming(thresh: Double, promise: Promise) {
        if (isRunning) { promise.resolve("already running"); return }
        if (ActivityCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            promise.reject("NO_PERMISSION", "RECORD_AUDIO permission not granted")
            return
        }
        // Mappa slider-värdet (0.005–0.15) till onset-ratio (1.5–5.0)
        // Lågt slider-värde = känslig = låg ratio
        // Högt slider-värde = strikt = hög ratio
        onsetRatio    = 1.5 + (thresh - 0.005) / (0.15 - 0.005) * (5.0 - 1.5)
        isRunning     = true
        lastOnsetTime = 0L
        bgIdx         = 0
        bgFilled      = false
        bgBuffer.fill(0.0)
        Thread(::streamLoop, "AudioStreamThread").start()
        promise.resolve("started")
    }

    @ReactMethod
    fun stopStreaming(promise: Promise) {
        isRunning = false
        promise.resolve("stopped")
    }

    @ReactMethod
    fun setThreshold(thresh: Double, promise: Promise) {
        onsetRatio = 1.5 + (thresh - 0.005) / (0.15 - 0.005) * (5.0 - 1.5)
        promise.resolve("ok")
    }

    // ── Huvud-loop ─────────────────────────────────────────────────────────────

    private fun streamLoop() {
        val minBuf = AudioRecord.getMinBufferSize(
            SR, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = AudioRecord(
            MediaRecorder.AudioSource.MIC, SR,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, FRAME_SIZE * 8))
        try {
            rec.startRecording()
            val frame = ShortArray(FRAME_SIZE)

            while (isRunning) {
                val read = rec.read(frame, 0, FRAME_SIZE)
                if (read <= 0) continue

                // Skriv frame till ring-buffer
                for (i in 0 until read) {
                    ring[writePos % RING_SIZE] = frame[i]
                    writePos++
                }

                val rms = computeRMS(frame, read)

                // Uppdatera bakgrundsestimat med current RMS
                // (vi lägger bara till i bakgrundsbuffern, inte vid onset)
                val now = System.currentTimeMillis()
                val inCooldown = now - lastOnsetTime < RETRIGGER_MS

                if (!inCooldown) {
                    // Beräkna bakgrundsnivå
                    val bgAvg = if (bgFilled)
                        bgBuffer.average()
                    else
                        bgBuffer.take(bgIdx).average().takeIf { bgIdx > 0 } ?: rms

                    // Adaptiv onset: studs måste vara onsetRatio gånger starkare än bakgrunden
                    // och alltid över absolut minimum
                    val adaptiveThreshold = maxOf(bgAvg * onsetRatio, ABS_MIN_RMS)

                    if (rms >= adaptiveThreshold) {
                        lastOnsetTime = now
                        val capturedOnsetPos = writePos - read
                        scheduleExtraction(capturedOnsetPos)
                        // Fyll bakgrundsbuffern med onset-RMS så att efterklangen
                        // inte triggar en ny onset (eko ser "normal" ut)
                        bgBuffer.fill(rms)
                        bgIdx = 0
                        bgFilled = true
                    } else {
                        // Uppdatera bakgrundsbuffer med denna tysta frame
                        bgBuffer[bgIdx % BG_FRAMES] = rms
                        bgIdx++
                        if (bgIdx >= BG_FRAMES) bgFilled = true
                    }
                }
            }
        } finally {
            rec.stop()
            rec.release()
        }
    }

    // ── Extrahera klipp i bakgrunden ──────────────────────────────────────────

    private fun scheduleExtraction(onsetPos: Int) {
        Thread {
            val targetWritePos = onsetPos + POST_SAMPLES
            while (isRunning && writePos < targetWritePos) {
                Thread.sleep(5)
            }
            if (!isRunning) return@Thread

            val clip  = ShortArray(CLIP_SAMPLES)
            val start = onsetPos - PRE_SAMPLES
            for (i in 0 until CLIP_SAMPLES) {
                val pos = (start + i).let { ((it % RING_SIZE) + RING_SIZE) % RING_SIZE }
                clip[i] = ring[pos]
            }

            val bytes = ByteBuffer.allocate(CLIP_SAMPLES * 2)
                .order(ByteOrder.LITTLE_ENDIAN)
                .apply { clip.forEach { putShort(it) } }
                .array()

            ctx.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit(EVENT_NAME, Base64.encodeToString(bytes, Base64.NO_WRAP))
        }.start()
    }

    // ── RMS ────────────────────────────────────────────────────────────────────

    private fun computeRMS(frame: ShortArray, n: Int): Double {
        var sum = 0.0
        for (i in 0 until n) sum += (frame[i].toDouble() / 32768.0).let { it * it }
        return sqrt(sum / n)
    }
}
