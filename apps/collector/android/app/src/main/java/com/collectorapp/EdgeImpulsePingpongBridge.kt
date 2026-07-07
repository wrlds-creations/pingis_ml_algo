package com.collectorapp

data class EdgeImpulsePingpongResult(
    val available: Boolean,
    val ok: Boolean,
    val errorCode: Double,
    val bounceProbability: Double,
    val noiseProbability: Double,
    val dspMs: Double,
    val classificationMs: Double,
    val anomalyMs: Double,
    val expectedSampleRateHz: Int,
    val expectedSampleCount: Int,
)

object EdgeImpulsePingpongBridge {
    private val loadError: Throwable?

    init {
        loadError = try {
            System.loadLibrary("edge_impulse_pingpong")
            null
        } catch (err: Throwable) {
            err
        }
    }

    @JvmStatic private external fun isAvailableNative(): Boolean
    @JvmStatic private external fun expectedSampleCountNative(): Int
    @JvmStatic private external fun expectedSampleRateNative(): Int
    @JvmStatic private external fun classifyPcm16kNative(samples: ShortArray): DoubleArray

    fun isAvailable(): Boolean {
        if (loadError != null) return false
        return try { isAvailableNative() } catch (_: Throwable) { false }
    }

    fun expectedSampleCount(): Int {
        if (loadError != null) return 0
        return try { expectedSampleCountNative() } catch (_: Throwable) { 0 }
    }

    fun expectedSampleRateHz(): Int {
        if (loadError != null) return 0
        return try { expectedSampleRateNative() } catch (_: Throwable) { 0 }
    }

    fun classify(samples: ShortArray): EdgeImpulsePingpongResult {
        val expectedRate = expectedSampleRateHz()
        val expectedCount = expectedSampleCount()
        if (!isAvailable()) {
            return EdgeImpulsePingpongResult(
                available = false,
                ok = false,
                errorCode = -404.0,
                bounceProbability = 0.0,
                noiseProbability = 0.0,
                dspMs = 0.0,
                classificationMs = 0.0,
                anomalyMs = 0.0,
                expectedSampleRateHz = expectedRate,
                expectedSampleCount = expectedCount,
            )
        }
        val values = try { classifyPcm16kNative(samples) } catch (_: Throwable) {
            doubleArrayOf(0.0, -500.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        }
        return EdgeImpulsePingpongResult(
            available = true,
            ok = values.getOrNull(0) == 1.0,
            errorCode = values.getOrNull(1) ?: -501.0,
            bounceProbability = values.getOrNull(2) ?: 0.0,
            noiseProbability = values.getOrNull(3) ?: 0.0,
            dspMs = values.getOrNull(4) ?: 0.0,
            classificationMs = values.getOrNull(5) ?: 0.0,
            anomalyMs = values.getOrNull(6) ?: 0.0,
            expectedSampleRateHz = expectedRate,
            expectedSampleCount = expectedCount,
        )
    }
}
