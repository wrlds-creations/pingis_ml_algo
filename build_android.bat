@echo off
set ANDROID_HOME=C:\Users\lovea\AppData\Local\Android\Sdk
set PATH=%PATH%;%ANDROID_HOME%\platform-tools
cd /d C:\Users\lovea\Desktop\dev\pingis_ml_algo\apps\collector
npx react-native run-android
