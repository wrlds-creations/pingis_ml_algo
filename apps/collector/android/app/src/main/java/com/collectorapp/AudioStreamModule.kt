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
import kotlin.math.cos
import kotlin.math.log10
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
        const val PRE_SAMPLES   = 2_205         // 100 ms
        const val POST_SAMPLES  = 4_410         // 200 ms
        const val CLIP_SAMPLES  = PRE_SAMPLES + POST_SAMPLES
        const val RING_SIZE     = SR * 3        // 3 s cirkulär buffer
        const val DEFAULT_RETRIGGER_MS = 220L   // min ms mellan onsets (förhindrar dubbeldetektering)

        // Adaptiv tröskel: hur många gånger starkare än bakgrunden en studs måste vara
        const val ONSET_RATIO   = 2.5

        // Bakgrundsestimat: rullande medel över de senaste N frames (300ms = 30 frames)
        const val BG_FRAMES     = 30

        // Absolut minimumtröskel — ignorerar extremt tysta rum (undviker noise floor)
        const val ABS_MIN_RMS   = 0.003

        const val EVENT_NAME    = "onBounceDetected"

        // Spektral gate: avvisa icke-bollljud (prat, klapp, steg)
        const val SPECTRAL_FFT  = 256
        const val BALL_LO_HZ    = 200.0
        const val BALL_HI_HZ    = 6000.0
        const val MIN_BALL_RATIO = 0.55  // minst 55% av energin i boll-bandet
        const val MAX_FLATNESS   = 0.6   // spectral flatness > detta = broadband brus

        // Duration gate: avvisa sustained ljud (prat, musik)
        const val SUSTAIN_FRAMES    = 8   // 80ms (8 × 10ms)
        const val SUSTAIN_MAX_ABOVE = 5   // om > 5 av 8 frames fortfarande höga → skippa
    }

    @Volatile private var isRunning     = false
    @Volatile private var lastOnsetTime = 0L
    @Volatile private var retriggerMs   = DEFAULT_RETRIGGER_MS

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

    @ReactMethod
    fun setRetriggerMs(ms: Double, promise: Promise) {
        retriggerMs = ms.toLong().coerceIn(0L, 800L)
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
                val inCooldown = now - lastOnsetTime < retriggerMs

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
                        if (!spectralGate(frame, read)) {
                            bgBuffer[bgIdx % BG_FRAMES] = rms
                            bgIdx++
                            if (bgIdx >= BG_FRAMES) bgFilled = true
                            continue
                        }

                        lastOnsetTime = now
                        val capturedOnsetPos = writePos - read
                        scheduleExtraction(capturedOnsetPos)
                        // Höj bakgrunden måttligt — inte till onset-nivå (som
                        // blockerar nästa slag i 500ms) utan till 2× bakgrunden.
                        // retriggerMs hanterar eko/decay och kan justeras från JS.
                        val elevatedBg = bgAvg * 2.0
                        bgBuffer.fill(elevatedBg)
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

    // ── Spektral gate: avvisar icke-boll-ljud ────────────────────────────────

    /**
     * Kör en snabb 256-punkts FFT på onset-framen och kontrollerar:
     * 1. Att minst MIN_BALL_RATIO av energin ligger i 200-6000 Hz (boll-bandet)
     * 2. Att spectral flatness < MAX_FLATNESS (inte broadband brus/klapp)
     *
     * Returnerar true om framen passerar (sannolikt boll-studs).
     */
    private fun spectralGate(frame: ShortArray, n: Int): Boolean {
        val nFft = SPECTRAL_FFT
        val re = DoubleArray(nFft)
        val im = DoubleArray(nFft)

        // Hann-fönster + ladda samples
        val len = minOf(n, nFft)
        for (i in 0 until len) {
            val w = 0.5 - 0.5 * cos(2.0 * Math.PI * i / (len - 1))
            re[i] = frame[i].toDouble() / 32768.0 * w
        }

        // Iterativ FFT (radix-2, Cooley-Tukey)
        var j = 0
        for (i in 1 until nFft) {
            var bit = nFft shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }
        var half = 1
        while (half < nFft) {
            val ang = -Math.PI / half
            val wr0 = cos(ang)
            val wi0 = kotlin.math.sin(ang)
            for (i in 0 until nFft step half * 2) {
                var wr = 1.0; var wi = 0.0
                for (k in 0 until half) {
                    val ur = re[i + k];          val ui = im[i + k]
                    val vr = re[i+k+half]*wr - im[i+k+half]*wi
                    val vi = re[i+k+half]*wi + im[i+k+half]*wr
                    re[i + k]       = ur + vr
                    im[i + k]       = ui + vi
                    re[i + k + half] = ur - vr
                    im[i + k + half] = ui - vi
                    val nwr = wr * wr0 - wi * wi0
                    wi = wr * wi0 + wi * wr0
                    wr = nwr
                }
            }
            half *= 2
        }

        // Power-spektrum (bara positiva frekvenser)
        val nBins = nFft / 2 + 1
        var totalEnergy = 0.0
        var ballEnergy = 0.0
        var logSum = 0.0
        var arithSum = 0.0
        val binHz = SR.toDouble() / nFft

        for (k in 1 until nBins) {  // skip DC
            val power = re[k] * re[k] + im[k] * im[k]
            val freq = k * binHz
            totalEnergy += power
            if (freq in BALL_LO_HZ..BALL_HI_HZ) ballEnergy += power

            // Spectral flatness
            val p = power + 1e-10
            logSum += kotlin.math.ln(p)
            arithSum += p
        }

        // Check 1: tillräckligt med energi i boll-bandet?
        if (totalEnergy > 0 && ballEnergy / totalEnergy < MIN_BALL_RATIO) {
            return false
        }

        // Check 2: spectral flatness (broadband brus = hög flatness)
        val validBins = nBins - 1
        if (validBins > 0 && arithSum > 0) {
            val geoMean = kotlin.math.exp(logSum / validBins)
            val ariMean = arithSum / validBins
            val flatness = geoMean / ariMean
            if (flatness > MAX_FLATNESS) return false
        }

        return true
    }
}
