package com.collectorapp

import android.content.pm.ActivityInfo
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod

class ReviewOrientationModule(private val ctx: ReactApplicationContext) :
    ReactContextBaseJavaModule(ctx) {

    override fun getName() = "ReviewOrientation"

    @ReactMethod
    fun lockLandscape(promise: Promise) {
        val activity = ctx.currentActivity
        if (activity == null) {
            promise.reject("NO_ACTIVITY", "No active activity to lock orientation")
            return
        }

        activity.runOnUiThread {
            activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
            promise.resolve(null)
        }
    }

    @ReactMethod
    fun lockPortrait(promise: Promise) {
        val activity = ctx.currentActivity
        if (activity == null) {
            promise.reject("NO_ACTIVITY", "No active activity to lock orientation")
            return
        }

        activity.runOnUiThread {
            activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_PORTRAIT
            promise.resolve(null)
        }
    }

    @ReactMethod
    fun unlock(promise: Promise) {
        val activity = ctx.currentActivity
        if (activity == null) {
            promise.resolve(null)
            return
        }

        activity.runOnUiThread {
            activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
            promise.resolve(null)
        }
    }
}
