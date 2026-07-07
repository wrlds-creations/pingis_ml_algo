#include <jni.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#if HAVE_EDGE_IMPULSE_PINGPONG
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "model-parameters/model_metadata.h"
#endif

namespace {

#if HAVE_EDGE_IMPULSE_PINGPONG
std::mutex g_classifier_mutex;
std::vector<float> g_signal_buffer;

int get_signal_data(size_t offset, size_t length, float *out_ptr) {
    if (offset + length > g_signal_buffer.size()) {
        return EIDSP_SIGNAL_SIZE_MISMATCH;
    }
    std::memcpy(out_ptr, g_signal_buffer.data() + offset, length * sizeof(float));
    return EIDSP_OK;
}

float probability_for_label(const ei_impulse_result_t &result, const char *label) {
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; ++i) {
        if (std::strcmp(result.classification[i].label, label) == 0) {
            return result.classification[i].value;
        }
    }
    return 0.0f;
}
#endif

} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_collectorapp_EdgeImpulsePingpongBridge_isAvailableNative(JNIEnv *, jclass) {
#if HAVE_EDGE_IMPULSE_PINGPONG
    return JNI_TRUE;
#else
    return JNI_FALSE;
#endif
}

extern "C" JNIEXPORT jint JNICALL
Java_com_collectorapp_EdgeImpulsePingpongBridge_expectedSampleCountNative(JNIEnv *, jclass) {
#if HAVE_EDGE_IMPULSE_PINGPONG
    return EI_CLASSIFIER_RAW_SAMPLE_COUNT;
#else
    return 0;
#endif
}

extern "C" JNIEXPORT jint JNICALL
Java_com_collectorapp_EdgeImpulsePingpongBridge_expectedSampleRateNative(JNIEnv *, jclass) {
#if HAVE_EDGE_IMPULSE_PINGPONG
    return EI_CLASSIFIER_FREQUENCY;
#else
    return 0;
#endif
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_collectorapp_EdgeImpulsePingpongBridge_classifyPcm16kNative(
    JNIEnv *env,
    jclass,
    jshortArray samples
) {
    // Result layout:
    // [ok, errorCode, bounceProbability, noiseProbability, dspMs, classificationMs, anomalyMs]
    constexpr int kResultLength = 7;
    double values[kResultLength] = {0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0};

#if HAVE_EDGE_IMPULSE_PINGPONG
    if (samples == nullptr) {
        values[1] = -2.0;
    } else {
        const jsize length = env->GetArrayLength(samples);
        if (length != EI_CLASSIFIER_RAW_SAMPLE_COUNT) {
            values[1] = -3.0;
        } else {
            std::vector<jshort> pcm(static_cast<size_t>(length));
            env->GetShortArrayRegion(samples, 0, length, pcm.data());

            std::lock_guard<std::mutex> lock(g_classifier_mutex);
            g_signal_buffer.resize(static_cast<size_t>(length));
            for (jsize i = 0; i < length; ++i) {
                g_signal_buffer[static_cast<size_t>(i)] =
                    std::clamp(static_cast<float>(pcm[static_cast<size_t>(i)]) / 32768.0f, -1.0f, 1.0f);
            }

            signal_t signal;
            signal.total_length = g_signal_buffer.size();
            signal.get_data = &get_signal_data;

            ei_impulse_result_t result = {};
            EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
            if (err == EI_IMPULSE_OK) {
                values[0] = 1.0;
                values[1] = 0.0;
                values[2] = probability_for_label(result, "Bounce");
                values[3] = probability_for_label(result, "noise");
                values[4] = result.timing.dsp;
                values[5] = result.timing.classification;
                values[6] = result.timing.anomaly;
            } else {
                values[1] = static_cast<double>(err);
            }
        }
    }
#else
    (void)samples;
    values[1] = -404.0;
#endif

    jdoubleArray out = env->NewDoubleArray(kResultLength);
    env->SetDoubleArrayRegion(out, 0, kResultLength, values);
    return out;
}
