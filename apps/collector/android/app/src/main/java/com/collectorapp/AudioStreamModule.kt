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
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
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

        // T0076 peak-candidate gate mirrors the offline "Peak fast balanced"
        // candidate source. The 80 ms lookahead is the local-maximum margin used
        // offline before the model/veto stage.
        const val PEAK_LOOKAHEAD_MS = 80.0
        const val PEAK_ROUGH_HEIGHT_FACTOR = 0.35
    }

    @Volatile private var isRunning     = false
    @Volatile private var lastOnsetTime = 0L
    @Volatile private var retriggerMs   = DEFAULT_RETRIGGER_MS
    @Volatile private var streamThread: Thread? = null
    @Volatile private var debugRecordingPath: String? = null

    // threshold-variabeln används nu som ONSET_RATIO-multiplier från slidern
    // (slider-värdet 0.005–0.15 mappas om till ratio 1.5–4.0 i startStreaming)
    @Volatile private var onsetRatio    = ONSET_RATIO

    // ── Gate-konfiguration (Fable-läget) ──────────────────────────────────────
    // Default = exakt gamla beteendet: broadband-RMS + spektral gate på.
    // "bandpass": frame-RMS beräknas på 1.5–7 kHz-bandpassat ljud (musik ligger
    // lågfrekvent; bollträffen behåller sin energi i bandet) — kausal variant
    // av offline-replayens nollfas-filter, skillnaden är några ms gruppfördröjning.
    @Volatile private var gateMode            = "broadband"
    @Volatile private var spectralGateEnabled = true
    @Volatile private var absMinRms           = ABS_MIN_RMS

    // T0076 guarded peak candidate mode. Disabled on every startStreaming and
    // enabled only by the Bounce audio test screen after the stream starts.
    @Volatile private var peakGateEnabled = false
    @Volatile private var peakSmoothMs = 3.0
    @Volatile private var peakMinGapMs = 220.0
    @Volatile private var peakBackgroundWindowMs = 500.0
    @Volatile private var peakBackgroundExcludeMs = 60.0
    @Volatile private var peakAbsMin = 0.08
    @Volatile private var peakRatioMin = 2.0
    @Volatile private var peakZMin = 0.0

    private val peakEnvRing = DoubleArray(RING_SIZE)
    private var peakSmoothWindow = DoubleArray(1)
    private var peakSmoothIdx = 0
    private var peakSmoothCount = 0
    private var peakSmoothSum = 0.0
    private var peakCandidateSample = -1
    private var peakCandidateValue = 0.0
    private var peakLastAcceptedSample = -RING_SIZE

    // Butterworth ordning 4 bandpass 1500–7000 Hz @ 22050 Hz som biquad-kaskad
    // (scipy.signal.butter(4, [1500, 7000], 'bandpass', fs=22050, output='sos')).
    // Rad: b0, b1, b2, a1, a2 (a0 = 1).
    private val bpSos = arrayOf(
        doubleArrayOf(0.09331299315653971, 0.18662598631307942, 0.09331299315653971, 0.19676635870899223, 0.09988498828008321),
        doubleArrayOf(1.0, -2.0, 1.0, -1.1925622397360882, 0.39614858444118883),
        doubleArrayOf(1.0, 2.0, 1.0, 0.6137123805223851, 0.5737584318188238),
        doubleArrayOf(1.0, -2.0, 1.0, -1.6102285405185548, 0.7781414737694868),
    )
    // Direct form II transposed: 2 tillstånd per sektion, persistent över frames.
    private val bpState = Array(4) { DoubleArray(2) }

    private fun resetBandpassState() {
        for (s in bpState) { s[0] = 0.0; s[1] = 0.0 }
    }

    private fun resetPeakGateState() {
        val samples = maxOf(1, (SR * peakSmoothMs / 1000.0).toInt())
        peakSmoothWindow = DoubleArray(samples)
        peakSmoothIdx = 0
        peakSmoothCount = 0
        peakSmoothSum = 0.0
        peakCandidateSample = -1
        peakCandidateValue = 0.0
        peakLastAcceptedSample = -RING_SIZE
        peakEnvRing.fill(0.0)
    }

    /** Filtrerar en frame kausalt genom biquad-kaskaden och returnerar RMS. */
    private fun bandpassFrameRms(frame: ShortArray, n: Int): Double {
        var sumSq = 0.0
        for (i in 0 until n) {
            var x = frame[i].toDouble() / 32768.0
            for (s in 0 until 4) {
                val c = bpSos[s]
                val st = bpState[s]
                val y = c[0] * x + st[0]
                st[0] = c[1] * x - c[3] * y + st[1]
                st[1] = c[2] * x - c[4] * y
                x = y
            }
            sumSq += x * x
        }
        return sqrt(sumSq / n)
    }

    private val ring = ShortArray(RING_SIZE)
    @Volatile private var writePos = 0

    // Cirkulär buffer för bakgrundsestimat
    private val bgBuffer = DoubleArray(BG_FRAMES)
    private var bgIdx    = 0
    private var bgFilled = false

    private data class SpectralGateResult(
        val passed: Boolean,
        val ballRatio: Double,
        val flatness: Double,
    )

    private data class OnsetDebug(
        val onsetTimeMs: Long,
        val onsetPos: Int,
        val rms: Double,
        val backgroundRms: Double,
        val adaptiveThreshold: Double,
        val onsetRatio: Double,
        val retriggerMs: Long,
        val spectralPassed: Boolean,
        val ballRatio: Double,
        val flatness: Double,
        val gateId: String = "adaptive_rms",
        val peakValue: Double? = null,
        val peakRatio: Double? = null,
        val peakZ: Double? = null,
        val peakLocalMad: Double? = null,
        val peakSmoothMs: Double? = null,
        val peakMinGapMs: Double? = null,
        val peakBackgroundWindowMs: Double? = null,
        val peakBackgroundExcludeMs: Double? = null,
        val peakAbsMin: Double? = null,
        val peakRatioMin: Double? = null,
        val peakZMin: Double? = null,
    )

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
        // Gate-konfig återställs till gamla beteendet vid varje start så att
        // de befintliga lägena aldrig påverkas av ett tidigare Fable-pass.
        // Fable-skärmen anropar setGateConfig direkt efter startStreaming.
        gateMode            = "broadband"
        spectralGateEnabled = true
        absMinRms           = ABS_MIN_RMS
        peakGateEnabled     = false
        resetBandpassState()
        resetPeakGateState()
        val thread = Thread(::streamLoop, "AudioStreamThread")
        streamThread = thread
        thread.start()
        promise.resolve("started")
    }

    @ReactMethod
    fun stopStreaming(promise: Promise) {
        isRunning = false
        streamThread?.let { thread ->
            if (thread != Thread.currentThread()) {
                try { thread.join(1000) } catch (_: InterruptedException) {}
            }
        }
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

    @ReactMethod
    fun setDebugRecordingPath(path: String?, promise: Promise) {
        debugRecordingPath = path?.takeIf { it.isNotBlank() }
        promise.resolve("ok")
    }

    /**
     * Konfigurera gate-läget (Fable-läget). mode: "broadband" | "bandpass".
     * spectralGate: kör 256-pt spektralgaten som hård avvisning eller inte
     * (ball_ratio/flatness rapporteras alltid i debug). absMin: absolut
     * RMS-golv för triggern (bandpass bör använda ~0.0015).
     * Anropa innan startStreaming; återställer filtertillstånd.
     */
    @ReactMethod
    fun setGateConfig(mode: String, spectralGate: Boolean, absMin: Double, promise: Promise) {
        gateMode = if (mode == "bandpass") "bandpass" else "broadband"
        spectralGateEnabled = spectralGate
        absMinRms = absMin.coerceIn(0.0001, 0.05)
        resetBandpassState()
        promise.resolve("ok")
    }

    @ReactMethod
    fun setPeakGateConfig(
        enabled: Boolean,
        smoothMs: Double,
        minGapMs: Double,
        backgroundWindowMs: Double,
        backgroundExcludeMs: Double,
        absMin: Double,
        ratioMin: Double,
        zMin: Double,
        promise: Promise
    ) {
        peakSmoothMs = smoothMs.coerceIn(1.0, 20.0)
        peakMinGapMs = minGapMs.coerceIn(80.0, 800.0)
        peakBackgroundWindowMs = backgroundWindowMs.coerceIn(100.0, 2000.0)
        peakBackgroundExcludeMs = backgroundExcludeMs.coerceIn(0.0, 300.0)
        peakAbsMin = absMin.coerceIn(0.001, 1.0)
        peakRatioMin = ratioMin.coerceIn(0.0, 100.0)
        peakZMin = zMin.coerceIn(-20.0, 1000.0)
        peakGateEnabled = enabled
        resetPeakGateState()
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
        val debugPath = debugRecordingPath
        var debugFile: File? = null
        var debugOut: FileOutputStream? = null
        var debugDataBytes = 0L
        try {
            rec.startRecording()
            if (!debugPath.isNullOrBlank()) {
                debugFile = File(debugPath)
                debugFile.parentFile?.mkdirs()
                debugOut = FileOutputStream(debugFile)
                debugOut.write(wavHeader(0L))
            }
            val frame = ShortArray(FRAME_SIZE)

            while (isRunning) {
                val read = rec.read(frame, 0, FRAME_SIZE)
                if (read <= 0) continue

                debugOut?.let { out ->
                    try {
                        val bytes = shortsToLittleEndianBytes(frame, read)
                        out.write(bytes)
                        debugDataBytes += bytes.size.toLong()
                    } catch (_: Exception) {
                        try { out.close() } catch (_: Exception) {}
                        debugOut = null
                    }
                }

                // Skriv frame till ring-buffer. Peak-läget läser samma samples
                // sample-för-sample för att kunna hitta en 3 ms rå amplitudenvelop.
                for (i in 0 until read) {
                    ring[writePos % RING_SIZE] = frame[i]
                    if (peakGateEnabled) processPeakSample(frame[i], writePos)
                    writePos++
                }

                if (peakGateEnabled) continue

                val rms = computeRMS(frame, read)
                // Gate-RMS: bandpassad om Fable-läget begärt det, annars rå.
                // Filtret måste mata sitt tillstånd varje frame, även i cooldown.
                val gateRms = if (gateMode == "bandpass") bandpassFrameRms(frame, read) else rms

                // Uppdatera bakgrundsestimat med current RMS
                // (vi lägger bara till i bakgrundsbuffern, inte vid onset)
                val now = System.currentTimeMillis()
                val inCooldown = now - lastOnsetTime < retriggerMs

                if (!inCooldown) {
                    // Beräkna bakgrundsnivå
                    val bgAvg = if (bgFilled)
                        bgBuffer.average()
                    else
                        bgBuffer.take(bgIdx).average().takeIf { bgIdx > 0 } ?: gateRms

                    // Adaptiv onset: studs måste vara onsetRatio gånger starkare än bakgrunden
                    // och alltid över absolut minimum
                    val adaptiveThreshold = maxOf(bgAvg * onsetRatio, absMinRms)

                    if (gateRms >= adaptiveThreshold) {
                        val capturedOnsetPos = writePos - read
                        val spectral = spectralGate(frame, read)
                        // Spektralgaten är hård avvisning bara när den är på;
                        // i Fable-läget gör modellen avvisningsjobbet i stället.
                        val spectralRejects = spectralGateEnabled && !spectral.passed
                        val debug = OnsetDebug(
                            onsetTimeMs = now,
                            onsetPos = capturedOnsetPos,
                            rms = gateRms,
                            backgroundRms = bgAvg,
                            adaptiveThreshold = adaptiveThreshold,
                            onsetRatio = onsetRatio,
                            retriggerMs = retriggerMs,
                            spectralPassed = spectral.passed,
                            ballRatio = spectral.ballRatio,
                            flatness = spectral.flatness,
                        )

                        if (spectralRejects) {
                            emitCandidate(null, debug, "spectral_gate")
                            bgBuffer[bgIdx % BG_FRAMES] = gateRms
                            bgIdx++
                            if (bgIdx >= BG_FRAMES) bgFilled = true
                            continue
                        }

                        lastOnsetTime = now
                        scheduleExtraction(capturedOnsetPos, debug)
                        // Höj bakgrunden måttligt — inte till onset-nivå (som
                        // blockerar nästa slag i 500ms) utan till 2× bakgrunden.
                        // retriggerMs hanterar eko/decay och kan justeras från JS.
                        val elevatedBg = bgAvg * 2.0
                        bgBuffer.fill(elevatedBg)
                        bgIdx = 0
                        bgFilled = true
                    } else {
                        // Uppdatera bakgrundsbuffer med denna tysta frame
                        bgBuffer[bgIdx % BG_FRAMES] = gateRms
                        bgIdx++
                        if (bgIdx >= BG_FRAMES) bgFilled = true
                    }
                }
            }
        } finally {
            try { rec.stop() } catch (_: Exception) {}
            rec.release()
            debugOut?.let { out ->
                try { out.flush() } catch (_: Exception) {}
                try { out.close() } catch (_: Exception) {}
            }
            debugFile?.let { file ->
                try { patchWavHeader(file, debugDataBytes) } catch (_: Exception) {}
            }
            debugRecordingPath = null
            streamThread = null
        }
    }

    private data class PeakLocalStats(val background: Double, val mad: Double)

    private fun processPeakSample(sample: Short, sampleIndex: Int) {
        val value = abs(sample.toDouble() / 32768.0)
        if (peakSmoothWindow.isEmpty()) resetPeakGateState()
        if (peakSmoothCount < peakSmoothWindow.size) {
            peakSmoothWindow[peakSmoothIdx] = value
            peakSmoothSum += value
            peakSmoothCount++
        } else {
            peakSmoothSum += value - peakSmoothWindow[peakSmoothIdx]
            peakSmoothWindow[peakSmoothIdx] = value
        }
        peakSmoothIdx = (peakSmoothIdx + 1) % peakSmoothWindow.size

        val envelopeValue = peakSmoothSum / maxOf(1, peakSmoothCount)
        peakEnvRing[sampleIndex % RING_SIZE] = envelopeValue

        val roughHeight = maxOf(1e-6, peakAbsMin * PEAK_ROUGH_HEIGHT_FACTOR)
        if (envelopeValue >= roughHeight) {
            if (peakCandidateSample < 0 || envelopeValue >= peakCandidateValue) {
                peakCandidateSample = sampleIndex
                peakCandidateValue = envelopeValue
            }
        }

        val lookaheadSamples = maxOf(1, (SR * PEAK_LOOKAHEAD_MS / 1000.0).toInt())
        if (peakCandidateSample >= 0 && sampleIndex - peakCandidateSample >= lookaheadSamples) {
            evaluatePeakCandidate(sampleIndex)
            peakCandidateSample = -1
            peakCandidateValue = 0.0
        }
    }

    private fun evaluatePeakCandidate(currentSampleIndex: Int) {
        val peakSample = peakCandidateSample
        if (peakSample < 0) return

        val minGapSamples = maxOf(1, (SR * peakMinGapMs / 1000.0).toInt())
        if (peakSample - peakLastAcceptedSample < minGapSamples) return

        val stats = peakLocalStats(peakSample)
        val peakValue = peakCandidateValue
        val peakRatio = peakValue / maxOf(stats.background, 1e-8)
        val peakZ = (peakValue - stats.background) / maxOf(stats.mad, 1e-8)

        if (peakValue < peakAbsMin) return
        if (peakRatio < peakRatioMin) return
        if (peakZ < peakZMin) return

        val frameRms = frameRmsAt(peakSample)
        val sampleDelayMs = ((currentSampleIndex - peakSample).toDouble() * 1000.0 / SR).toLong()
        val onsetTimeMs = System.currentTimeMillis() - sampleDelayMs
        peakLastAcceptedSample = peakSample
        lastOnsetTime = onsetTimeMs

        val debug = OnsetDebug(
            onsetTimeMs = onsetTimeMs,
            onsetPos = peakSample,
            rms = frameRms,
            backgroundRms = stats.background,
            adaptiveThreshold = peakAbsMin,
            onsetRatio = peakRatio,
            retriggerMs = peakMinGapMs.toLong(),
            spectralPassed = true,
            ballRatio = 0.0,
            flatness = 0.0,
            gateId = "peak_fast_balanced",
            peakValue = peakValue,
            peakRatio = peakRatio,
            peakZ = peakZ,
            peakLocalMad = stats.mad,
            peakSmoothMs = peakSmoothMs,
            peakMinGapMs = peakMinGapMs,
            peakBackgroundWindowMs = peakBackgroundWindowMs,
            peakBackgroundExcludeMs = peakBackgroundExcludeMs,
            peakAbsMin = peakAbsMin,
            peakRatioMin = peakRatioMin,
            peakZMin = peakZMin,
        )
        scheduleExtraction(peakSample, debug)
    }

    private fun peakLocalStats(sampleIndex: Int): PeakLocalStats {
        val excludeSamples = maxOf(0, (SR * peakBackgroundExcludeMs / 1000.0).toInt())
        val windowSamples = maxOf(1, (SR * peakBackgroundWindowMs / 1000.0).toInt())
        val end = sampleIndex - excludeSamples
        val start = end - windowSamples
        val minSamples = maxOf(16, (0.02 * SR).toInt())
        var values = readPeakEnvWindow(start, end)
        if (values.size < minSamples) {
            values = readPeakEnvWindow(maxOf(0, sampleIndex - windowSamples), sampleIndex)
        }
        if (values.isEmpty()) return PeakLocalStats(1e-6, 1e-6)
        values.sort()
        val median = medianOfSorted(values)
        val deviations = DoubleArray(values.size)
        for (i in values.indices) deviations[i] = abs(values[i] - median)
        deviations.sort()
        val mad = medianOfSorted(deviations)
        return PeakLocalStats(maxOf(median, 1e-8), maxOf(mad, 1e-8))
    }

    private fun readPeakEnvWindow(startInclusive: Int, endExclusive: Int): DoubleArray {
        val start = maxOf(0, startInclusive)
        val end = minOf(writePos, endExclusive)
        val count = end - start
        if (count <= 0 || count >= RING_SIZE) return DoubleArray(0)
        val out = DoubleArray(count)
        for (i in 0 until count) {
            out[i] = peakEnvRing[(start + i) % RING_SIZE]
        }
        return out
    }

    private fun medianOfSorted(values: DoubleArray): Double {
        if (values.isEmpty()) return 0.0
        val mid = values.size / 2
        return if (values.size % 2 == 0) (values[mid - 1] + values[mid]) / 2.0 else values[mid]
    }

    private fun frameRmsAt(startSample: Int): Double {
        var sum = 0.0
        for (i in 0 until FRAME_SIZE) {
            val sample = ring[(startSample + i) % RING_SIZE].toDouble() / 32768.0
            sum += sample * sample
        }
        return sqrt(sum / FRAME_SIZE)
    }

    private fun shortsToLittleEndianBytes(samples: ShortArray, n: Int): ByteArray {
        return ByteBuffer.allocate(n * 2)
            .order(ByteOrder.LITTLE_ENDIAN)
            .apply {
                for (i in 0 until n) putShort(samples[i])
            }
            .array()
    }

    private fun wavHeader(dataBytes: Long): ByteArray {
        val byteRate = SR * 2
        val riffSize = 36L + dataBytes
        return ByteBuffer.allocate(44)
            .order(ByteOrder.LITTLE_ENDIAN)
            .apply {
                put("RIFF".toByteArray(Charsets.US_ASCII))
                putInt(riffSize.coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
                put("WAVE".toByteArray(Charsets.US_ASCII))
                put("fmt ".toByteArray(Charsets.US_ASCII))
                putInt(16)
                putShort(1.toShort())
                putShort(1.toShort())
                putInt(SR)
                putInt(byteRate)
                putShort(2.toShort())
                putShort(16.toShort())
                put("data".toByteArray(Charsets.US_ASCII))
                putInt(dataBytes.coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
            }
            .array()
    }

    private fun patchWavHeader(file: File, dataBytes: Long) {
        RandomAccessFile(file, "rw").use { raf ->
            val riffSize = 36L + dataBytes
            raf.seek(4)
            raf.write(intToLittleEndian(riffSize.coerceAtMost(Int.MAX_VALUE.toLong()).toInt()))
            raf.seek(40)
            raf.write(intToLittleEndian(dataBytes.coerceAtMost(Int.MAX_VALUE.toLong()).toInt()))
        }
    }

    private fun intToLittleEndian(value: Int): ByteArray {
        return ByteBuffer.allocate(4)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putInt(value)
            .array()
    }

    // ── Extrahera klipp i bakgrunden ──────────────────────────────────────────

    private fun scheduleExtraction(onsetPos: Int, debug: OnsetDebug) {
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

            emitCandidate(Base64.encodeToString(bytes, Base64.NO_WRAP), debug, null)
        }.start()
    }

    private fun emitCandidate(audioB64: String?, debug: OnsetDebug, rejectedReason: String?) {
        val debugMap = Arguments.createMap().apply {
            putString("gate_id", debug.gateId)
            putDouble("onset_time_ms", debug.onsetTimeMs.toDouble())
            putDouble("onset_pos", debug.onsetPos.toDouble())
            putDouble("rms", debug.rms)
            putDouble("background_rms", debug.backgroundRms)
            putDouble("adaptive_threshold", debug.adaptiveThreshold)
            putDouble("onset_ratio", debug.onsetRatio)
            putDouble("retrigger_ms", debug.retriggerMs.toDouble())
            putBoolean("spectral_passed", debug.spectralPassed)
            putDouble("ball_ratio", debug.ballRatio)
            putDouble("flatness", debug.flatness)
            debug.peakValue?.let { putDouble("peak_value", it) }
            debug.peakRatio?.let { putDouble("peak_ratio", it) }
            debug.peakZ?.let { putDouble("peak_z", it) }
            debug.peakLocalMad?.let { putDouble("peak_local_mad", it) }
            debug.peakSmoothMs?.let { putDouble("peak_smooth_ms", it) }
            debug.peakMinGapMs?.let { putDouble("peak_min_gap_ms", it) }
            debug.peakBackgroundWindowMs?.let { putDouble("peak_background_window_ms", it) }
            debug.peakBackgroundExcludeMs?.let { putDouble("peak_background_exclude_ms", it) }
            debug.peakAbsMin?.let { putDouble("peak_abs_min", it) }
            debug.peakRatioMin?.let { putDouble("peak_ratio_min", it) }
            debug.peakZMin?.let { putDouble("peak_z_min", it) }
            if (rejectedReason == null) putNull("native_reject_reason") else putString("native_reject_reason", rejectedReason)
        }
        val payload = Arguments.createMap().apply {
            if (audioB64 == null) putNull("audio_b64") else putString("audio_b64", audioB64)
            putMap("native_debug", debugMap)
        }
        ctx.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(EVENT_NAME, payload)
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
    private fun spectralGate(frame: ShortArray, n: Int): SpectralGateResult {
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
        val ballRatio = if (totalEnergy > 0) ballEnergy / totalEnergy else 0.0
        if (totalEnergy > 0 && ballRatio < MIN_BALL_RATIO) {
            return SpectralGateResult(false, ballRatio, 0.0)
        }

        // Check 2: spectral flatness (broadband brus = hög flatness)
        val validBins = nBins - 1
        var flatness = 0.0
        if (validBins > 0 && arithSum > 0) {
            val geoMean = kotlin.math.exp(logSum / validBins)
            val ariMean = arithSum / validBins
            flatness = geoMean / ariMean
            if (flatness > MAX_FLATNESS) return SpectralGateResult(false, ballRatio, flatness)
        }

        return SpectralGateResult(true, ballRatio, flatness)
    }
}
