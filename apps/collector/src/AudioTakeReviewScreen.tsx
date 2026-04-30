import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  PanResponder,
  Pressable,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  useWindowDimensions,
} from 'react-native';
import AudioRecorderPlayer from 'react-native-audio-recorder-player';
import RNFS from 'react-native-fs';
import Video, { type VideoRef } from 'react-native-video';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import {
  REVIEW_PRE_MS,
  buildMarkerZoomWaveformWindow,
  buildSuggestedReviewMarkers,
  buildWaveformBins,
  createManualMarker,
  decodeWavFile,
  snapMarkerToAttack,
  writePreviewClip,
  writeTakePlaybackClip,
} from './audioReview';
import { ReviewOrientation } from './ReviewOrientation';
import type { AudioEvent, AudioReviewLabel, AudioReviewMarker } from './types';

interface Props {
  event: AudioEvent;
  filePath: string;
  videoFilePath?: string;
  onSave: (markers: AudioReviewMarker[]) => Promise<void> | void;
  onDiscard: () => Promise<void> | void;
  onBack: () => void;
}

type PlaybackMode = 'idle' | 'playing_full_take' | 'playing_preview' | 'paused_full_take';
type PlaybackRate = 1 | 0.5 | 0.25;

const REVIEW_UI_REVISION = 'Simple Review UI | attack_start | r11c-overview-fit';
const NUDGE_STEP_MS = 10;
const PLAYBACK_SUBSCRIPTION_SEC = 0.05;
const OVERVIEW_BAR_COUNT = 260;
const DETAIL_PRE_MS = 120;
const DETAIL_POST_MS = 120;
const OVERVIEW_PRE_MS = 1800;
const OVERVIEW_POST_MS = 1800;

function labelText(label: AudioReviewLabel) {
  switch (label) {
    case 'racket_contact':
      return 'Racket';
    case 'not_racket_contact':
      return 'Not racket';
    case 'ignore':
      return 'Ignore';
  }
}

function labelColor(label: AudioReviewLabel) {
  switch (label) {
    case 'racket_contact':
      return '#2ecc71';
    case 'not_racket_contact':
      return '#ff9f43';
    case 'ignore':
      return '#888';
  }
}

function sortMarkers(markers: AudioReviewMarker[]) {
  return [...markers].sort((a, b) => a.timestamp_ms - b.timestamp_ms);
}

function clampTimestamp(timestampMs: number, durationMs: number) {
  return Math.max(0, Math.min(durationMs, Math.round(timestampMs)));
}

function formatMs(ms: number) {
  const safe = Math.max(0, Math.round(ms));
  const seconds = Math.floor(safe / 1000);
  const centiseconds = Math.floor((safe % 1000) / 10);
  return `${seconds}.${String(centiseconds).padStart(2, '0')}s`;
}

function ratioToLeft(timestampMs: number, startMs: number, endMs: number, width: number) {
  if (endMs <= startMs) return 0;
  const ratio = (timestampMs - startMs) / (endMs - startMs);
  return Math.max(0, Math.min(width, ratio * width));
}

function mapAudioPlayheadToVideoMs(event: AudioEvent, audioMs: number, audioDurationMs: number) {
  const video = event.video_recording;
  if (!video) return 0;
  if (audioDurationMs <= 0 || video.duration_ms <= 0) return 0;
  const normalized = Math.max(0, Math.min(1, audioMs / audioDurationMs));
  return Math.round(normalized * video.duration_ms);
}

export function AudioTakeReviewScreen({ event, filePath, videoFilePath, onSave, onDiscard, onBack }: Props) {
  const { width: windowWidth, height: windowHeight } = useWindowDimensions();
  const insets = useSafeAreaInsets();
  const playerRef = useRef(new AudioRecorderPlayer());
  const videoRef = useRef<VideoRef>(null);
  const decodedRef = useRef<{ sampleRate: number; samples: Float32Array } | null>(null);
  const tempPlaybackPathRef = useRef<string | null>(null);
  const playbackModeRef = useRef<PlaybackMode>('idle');
  const playbackRateRef = useRef<PlaybackRate>(1);
  const playbackSourceStartMsRef = useRef(0);
  const overviewDragStartMsRef = useRef(0);
  const overviewMarkerDragStartMsRef = useRef(0);
  const detailMarkerDragStartMsRef = useRef(0);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [durationMs, setDurationMs] = useState(event.duration_ms);
  const [overviewBins, setOverviewBins] = useState<number[]>([]);
  const [markers, setMarkers] = useState<AudioReviewMarker[]>(event.review?.markers ?? []);
  const [selectedMarkerId, setSelectedMarkerId] = useState<string | null>(event.review?.markers?.[0]?.id ?? null);
  const [playbackPositionMs, setPlaybackPositionMs] = useState(0);
  const [playbackMode, setPlaybackMode] = useState<PlaybackMode>('idle');
  const [playbackRate, setPlaybackRate] = useState<PlaybackRate>(1);
  const [overviewWidth, setOverviewWidth] = useState(0);
  const [detailWidth, setDetailWidth] = useState(0);
  const overviewBinCount = useMemo(
    () => Math.max(64, Math.floor((overviewWidth || 240) / 2)),
    [overviewWidth],
  );

  const orderedMarkers = useMemo(() => sortMarkers(markers), [markers]);
  const selectedMarker = useMemo(
    () => orderedMarkers.find(marker => marker.id === selectedMarkerId) ?? null,
    [orderedMarkers, selectedMarkerId],
  );
  const selectedMarkerIndex = useMemo(
    () => selectedMarker ? orderedMarkers.findIndex(marker => marker.id === selectedMarker.id) : -1,
    [orderedMarkers, selectedMarker],
  );

  const detailFocusMs = playbackMode === 'playing_full_take'
    ? playbackPositionMs
    : selectedMarker?.timestamp_ms ?? playbackPositionMs;

  const detailWindow = useMemo(() => {
    if (!decodedRef.current) {
      return { bins: [], start_ms: 0, end_ms: 0, focus_ms: 0, peak_ms: 0 };
    }
    return buildMarkerZoomWaveformWindow(
      decodedRef.current.samples,
      decodedRef.current.sampleRate,
      detailFocusMs,
      Math.max(280, Math.round((detailWidth || 720) / 2)),
      DETAIL_PRE_MS,
      DETAIL_POST_MS,
    );
  }, [detailFocusMs, detailWidth]);
  const overviewFocusMs = playbackPositionMs;
  const overviewWindow = useMemo(() => {
    if (!decodedRef.current) {
      return { bins: overviewBins, start_ms: 0, end_ms: durationMs };
    }
    const totalDurationMs = durationMs;
    if (totalDurationMs <= 0) {
      return { bins: [], start_ms: 0, end_ms: 0 };
    }

    const desiredSpanMs = OVERVIEW_PRE_MS + OVERVIEW_POST_MS;
    const safeSpanMs = Math.min(totalDurationMs, desiredSpanMs);
    let startMs = Math.max(0, overviewFocusMs - OVERVIEW_PRE_MS);
    let endMs = Math.min(totalDurationMs, startMs + safeSpanMs);
    startMs = Math.max(0, endMs - safeSpanMs);

    const startSample = Math.max(0, Math.floor((startMs / 1000) * decodedRef.current.sampleRate));
    const endSample = Math.min(
      decodedRef.current.samples.length,
      Math.ceil((endMs / 1000) * decodedRef.current.sampleRate),
    );

    return {
      bins: buildWaveformBins(decodedRef.current.samples.slice(startSample, endSample), overviewBinCount),
      start_ms: Math.round(startMs),
      end_ms: Math.round(endMs),
    };
  }, [durationMs, overviewBinCount, overviewBins, overviewFocusMs]);
  const hasVideo = Boolean(videoFilePath && event.video_recording);
  const videoPlaybackMs = useMemo(
    () => mapAudioPlayheadToVideoMs(event, playbackPositionMs, durationMs),
    [durationMs, event, playbackPositionMs],
  );

  const stopCurrentPlayback = useCallback(async (nextMode: PlaybackMode = 'idle') => {
    playbackModeRef.current = nextMode;
    setPlaybackMode(nextMode);

    const tempPath = tempPlaybackPathRef.current;
    tempPlaybackPathRef.current = null;

    await playerRef.current.stopPlayer().catch(() => {});

    if (tempPath) {
      await RNFS.unlink(tempPath).catch(() => {});
    }
  }, []);

  const setPlayheadMs = useCallback((nextTimestampMs: number) => {
    setPlaybackPositionMs(clampTimestamp(nextTimestampMs, durationMs));
  }, [durationMs]);

  const updateSelectedMarkerTimestamp = useCallback((nextTimestampMs: number) => {
    if (!selectedMarkerId) return;
    setMarkers(prev => sortMarkers(
      prev.map(marker => (
        marker.id === selectedMarkerId
          ? { ...marker, timestamp_ms: clampTimestamp(nextTimestampMs, durationMs) }
          : marker
      )),
    ));
  }, [durationMs, selectedMarkerId]);

  const stopForScrubIfNeeded = useCallback(() => {
    if (playbackModeRef.current === 'playing_full_take') {
      void stopCurrentPlayback('paused_full_take');
    } else if (playbackModeRef.current === 'playing_preview') {
      void stopCurrentPlayback('idle');
    }
  }, [stopCurrentPlayback]);

  const handleSelectMarker = useCallback((marker: AudioReviewMarker) => {
    setSelectedMarkerId(marker.id);
    setPlayheadMs(Math.max(0, marker.timestamp_ms - REVIEW_PRE_MS));
  }, [setPlayheadMs]);

  const handleJumpToMarker = useCallback((direction: -1 | 1) => {
    if (selectedMarkerIndex < 0) return;
    const nextIndex = selectedMarkerIndex + direction;
    if (nextIndex < 0 || nextIndex >= orderedMarkers.length) return;
    handleSelectMarker(orderedMarkers[nextIndex]);
  }, [handleSelectMarker, orderedMarkers, selectedMarkerIndex]);

  const overviewPlayheadResponder = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: (_, gestureState) => Math.abs(gestureState.dx) > 2,
    onPanResponderGrant: () => {
      stopForScrubIfNeeded();
      overviewDragStartMsRef.current = playbackPositionMs;
    },
    onPanResponderMove: (_, gestureState) => {
      if (overviewWidth <= 0) return;
      const msPerPx = Math.max(1, overviewWindow.end_ms - overviewWindow.start_ms) / overviewWidth;
      setPlayheadMs(overviewDragStartMsRef.current + gestureState.dx * msPerPx);
    },
  }), [overviewWidth, overviewWindow.end_ms, overviewWindow.start_ms, playbackPositionMs, setPlayheadMs, stopForScrubIfNeeded]);

  const overviewMarkerResponder = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => Boolean(selectedMarker),
    onMoveShouldSetPanResponder: (_, gestureState) => Boolean(selectedMarker) && Math.abs(gestureState.dx) > 2,
    onPanResponderGrant: () => {
      if (!selectedMarker) return;
      stopForScrubIfNeeded();
      overviewMarkerDragStartMsRef.current = selectedMarker.timestamp_ms;
    },
    onPanResponderMove: (_, gestureState) => {
      if (!selectedMarker || overviewWidth <= 0) return;
      const msPerPx = Math.max(1, overviewWindow.end_ms - overviewWindow.start_ms) / overviewWidth;
      const nextTimestampMs = overviewMarkerDragStartMsRef.current + gestureState.dx * msPerPx;
      updateSelectedMarkerTimestamp(nextTimestampMs);
      setPlayheadMs(Math.max(0, nextTimestampMs - REVIEW_PRE_MS));
    },
  }), [overviewWidth, overviewWindow.end_ms, overviewWindow.start_ms, selectedMarker, setPlayheadMs, stopForScrubIfNeeded, updateSelectedMarkerTimestamp]);

  const detailMarkerResponder = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => Boolean(selectedMarker),
    onMoveShouldSetPanResponder: (_, gestureState) => Boolean(selectedMarker) && Math.abs(gestureState.dx) > 1,
    onPanResponderGrant: () => {
      if (!selectedMarker) return;
      stopForScrubIfNeeded();
      detailMarkerDragStartMsRef.current = selectedMarker.timestamp_ms;
    },
    onPanResponderMove: (_, gestureState) => {
      if (!selectedMarker || detailWidth <= 0) return;
      const detailDurationMs = Math.max(1, detailWindow.end_ms - detailWindow.start_ms);
      const msPerPx = detailDurationMs / detailWidth;
      const nextTimestampMs = detailMarkerDragStartMsRef.current + gestureState.dx * msPerPx;
      updateSelectedMarkerTimestamp(nextTimestampMs);
      setPlayheadMs(Math.max(0, nextTimestampMs - REVIEW_PRE_MS));
    },
  }), [detailWidth, detailWindow.end_ms, detailWindow.start_ms, selectedMarker, setPlayheadMs, stopForScrubIfNeeded, updateSelectedMarkerTimestamp]);

  useEffect(() => {
    playbackModeRef.current = playbackMode;
  }, [playbackMode]);

  useEffect(() => {
    playbackRateRef.current = playbackRate;
  }, [playbackRate]);

  useEffect(() => {
    if (!hasVideo || !videoRef.current) return;
    if (playbackMode === 'playing_full_take' || playbackMode === 'playing_preview') return;
    videoRef.current.seek(videoPlaybackMs / 1000);
  }, [hasVideo, playbackMode, videoPlaybackMs]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    decodeWavFile(filePath)
      .then(decoded => {
        if (cancelled) return;
        decodedRef.current = { sampleRate: decoded.sampleRate, samples: decoded.samples };
        setDurationMs(decoded.durationMs);
        setOverviewBins(buildWaveformBins(decoded.samples, overviewBinCount));

        const nextMarkers = event.review?.markers?.length
          ? sortMarkers(event.review.markers)
          : buildSuggestedReviewMarkers(decoded.samples, decoded.sampleRate, event.scenario_id);
        setMarkers(nextMarkers);
        setSelectedMarkerId(nextMarkers[0]?.id ?? null);
        setPlaybackPositionMs(0);
      })
      .catch(error => {
        Alert.alert('Review error', `Could not load take: ${String(error)}`);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [event.review?.markers, event.scenario_id, filePath, overviewBinCount]);

  useEffect(() => {
    void ReviewOrientation.lockPortrait();
    return () => {
      void stopCurrentPlayback('idle');
      void ReviewOrientation.unlock();
    };
  }, [stopCurrentPlayback]);

  useEffect(() => {
    playerRef.current.setSubscriptionDuration(PLAYBACK_SUBSCRIPTION_SEC).catch(() => {});
    playerRef.current.addPlayBackListener(eventData => {
      const currentPosition = Math.round(eventData.currentPosition);
      const duration = Math.round(eventData.duration);

      if (playbackModeRef.current === 'playing_full_take' || playbackModeRef.current === 'playing_preview') {
        const mappedMs = playbackSourceStartMsRef.current + currentPosition * playbackRateRef.current;
        setPlaybackPositionMs(clampTimestamp(mappedMs, durationMs));
      }

      if (duration > 0 && currentPosition >= duration - 40) {
        void stopCurrentPlayback('idle');
      }
    });

    return () => {
      playerRef.current.removePlayBackListener();
      void stopCurrentPlayback('idle');
    };
  }, [durationMs, stopCurrentPlayback]);

  const handleOverviewPress = useCallback((locationX: number) => {
    stopForScrubIfNeeded();
    if (overviewWidth <= 0) return;
    const windowDurationMs = Math.max(1, overviewWindow.end_ms - overviewWindow.start_ms);
    setPlayheadMs(overviewWindow.start_ms + (locationX / overviewWidth) * windowDurationMs);
  }, [overviewWidth, overviewWindow.end_ms, overviewWindow.start_ms, setPlayheadMs, stopForScrubIfNeeded]);

  const overviewScrubResponder = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => false,
    onMoveShouldSetPanResponder: (_, gestureState) => Math.abs(gestureState.dx) > 2,
    onPanResponderGrant: eventData => {
      handleOverviewPress(eventData.nativeEvent.locationX);
    },
    onPanResponderMove: eventData => {
      handleOverviewPress(eventData.nativeEvent.locationX);
    },
  }), [handleOverviewPress]);

  const handleDetailPress = useCallback((locationX: number) => {
    stopForScrubIfNeeded();
    if (detailWidth <= 0) return;
    const ratio = locationX / detailWidth;
    const nextTimestampMs = detailWindow.start_ms + ratio * (detailWindow.end_ms - detailWindow.start_ms);
    setPlayheadMs(nextTimestampMs);
  }, [detailWidth, detailWindow.end_ms, detailWindow.start_ms, setPlayheadMs, stopForScrubIfNeeded]);

  const handlePlayFromHere = useCallback(async () => {
    if (!decodedRef.current) return;

    try {
      await stopCurrentPlayback('idle');
      if (hasVideo) {
        videoRef.current?.seek(mapAudioPlayheadToVideoMs(event, playbackPositionMs, durationMs) / 1000);
      }
      const clipPath = await writeTakePlaybackClip(
        decodedRef.current.samples,
        decodedRef.current.sampleRate,
        playbackPositionMs,
        playbackRate,
      );
      tempPlaybackPathRef.current = clipPath;
      playbackSourceStartMsRef.current = playbackPositionMs;
      playbackModeRef.current = 'playing_full_take';
      setPlaybackMode('playing_full_take');
      await playerRef.current.startPlayer(clipPath);
    } catch (error) {
      await stopCurrentPlayback('idle');
      Alert.alert('Playback error', `Could not play from the selected position: ${String(error)}`);
    }
  }, [event, hasVideo, playbackPositionMs, playbackRate, stopCurrentPlayback]);

  const handlePause = useCallback(async () => {
    const nextMode = playbackModeRef.current === 'playing_full_take' ? 'paused_full_take' : 'idle';
    await stopCurrentPlayback(nextMode);
  }, [stopCurrentPlayback]);

  const handlePlaySelectedMarker = useCallback(async () => {
    if (!selectedMarker || !decodedRef.current) return;

    try {
      await stopCurrentPlayback('idle');
      const previewStartMs = Math.max(0, selectedMarker.timestamp_ms - REVIEW_PRE_MS);
      if (hasVideo) {
        videoRef.current?.seek(mapAudioPlayheadToVideoMs(event, previewStartMs, durationMs) / 1000);
      }
      const previewPath = await writePreviewClip(
        decodedRef.current.samples,
        decodedRef.current.sampleRate,
        selectedMarker.timestamp_ms,
        playbackRate,
      );
      tempPlaybackPathRef.current = previewPath;
      playbackSourceStartMsRef.current = previewStartMs;
      playbackModeRef.current = 'playing_preview';
      setPlaybackMode('playing_preview');
      setPlayheadMs(previewStartMs);
      await playerRef.current.startPlayer(previewPath);
    } catch (error) {
      await stopCurrentPlayback('idle');
      Alert.alert('Preview error', `Could not play marker preview: ${String(error)}`);
    }
  }, [event, hasVideo, playbackRate, selectedMarker, setPlayheadMs, stopCurrentPlayback]);

  const handleSetPlaybackRate = useCallback((nextRate: PlaybackRate) => {
    setPlaybackRate(nextRate);
    playbackRateRef.current = nextRate;
    if (playbackModeRef.current === 'playing_full_take' || playbackModeRef.current === 'playing_preview') {
      void handlePause();
    }
  }, [handlePause]);

  const updateSelectedMarker = useCallback((updater: (marker: AudioReviewMarker) => AudioReviewMarker) => {
    if (!selectedMarkerId) return;
    setMarkers(prev => sortMarkers(prev.map(marker => (
      marker.id === selectedMarkerId ? updater(marker) : marker
    ))));
  }, [selectedMarkerId]);

  const handleAddMarkerHere = useCallback(() => {
    const marker = createManualMarker(playbackPositionMs, event.scenario_id);
    setMarkers(prev => sortMarkers([...prev, marker]));
    setSelectedMarkerId(marker.id);
  }, [event.scenario_id, playbackPositionMs]);

  const handleDeleteSelectedMarker = useCallback(() => {
    if (!selectedMarker) return;
    const nextMarkers = orderedMarkers.filter(marker => marker.id !== selectedMarker.id);
    setMarkers(nextMarkers);
    const nextSelected = nextMarkers[Math.min(selectedMarkerIndex, nextMarkers.length - 1)] ?? null;
    setSelectedMarkerId(nextSelected?.id ?? null);
    if (nextSelected) {
      setPlayheadMs(Math.max(0, nextSelected.timestamp_ms - REVIEW_PRE_MS));
    }
  }, [orderedMarkers, selectedMarker, selectedMarkerIndex, setPlayheadMs]);

  const handleNudgeMarker = useCallback((deltaMs: number) => {
    if (!selectedMarker) return;
    const nextTimestampMs = clampTimestamp(selectedMarker.timestamp_ms + deltaMs, durationMs);
    updateSelectedMarker(marker => ({
      ...marker,
      timestamp_ms: nextTimestampMs,
    }));
    setPlayheadMs(Math.max(0, nextTimestampMs - REVIEW_PRE_MS));
  }, [durationMs, selectedMarker, setPlayheadMs, updateSelectedMarker]);

  const handleSnapSelectedMarker = useCallback(() => {
    if (!selectedMarker || !decodedRef.current) return;
    const snappedTimestampMs = snapMarkerToAttack(
      decodedRef.current.samples,
      decodedRef.current.sampleRate,
      selectedMarker.timestamp_ms,
    );
    updateSelectedMarker(marker => ({
      ...marker,
      timestamp_ms: snappedTimestampMs,
    }));
    setPlayheadMs(Math.max(0, snappedTimestampMs - REVIEW_PRE_MS));
  }, [selectedMarker, setPlayheadMs, updateSelectedMarker]);

  const handleSave = useCallback(async () => {
    if (orderedMarkers.length === 0) {
      Alert.alert('No markers', 'Add at least one marker or discard the take.');
      return;
    }

    setSaving(true);
    try {
      await onSave(orderedMarkers.map(marker => ({
        ...marker,
        timestamp_ms: clampTimestamp(marker.timestamp_ms, durationMs),
      })));
    } finally {
      setSaving(false);
    }
  }, [durationMs, onSave, orderedMarkers]);

  const handleDiscard = useCallback(() => {
    Alert.alert(
      'Discard take',
      'This removes the take from the session and deletes the WAV file and review video.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Discard',
          style: 'destructive',
          onPress: () => {
            onDiscard();
          },
        },
      ],
    );
  }, [onDiscard]);

  const markerCounts = useMemo(() => ({
    racket: orderedMarkers.filter(marker => marker.final_label === 'racket_contact').length,
    notRacket: orderedMarkers.filter(marker => marker.final_label === 'not_racket_contact').length,
    ignore: orderedMarkers.filter(marker => marker.final_label === 'ignore').length,
  }), [orderedMarkers]);

  const overviewPlayheadLeft = ratioToLeft(
    playbackPositionMs,
    overviewWindow.start_ms,
    overviewWindow.end_ms,
    overviewWidth,
  );
  const detailPlayheadLeft = ratioToLeft(playbackPositionMs, detailWindow.start_ms, detailWindow.end_ms, detailWidth);
  const detailPeakLeft = ratioToLeft(detailWindow.peak_ms, detailWindow.start_ms, detailWindow.end_ms, detailWidth);
  const detailMarkerLeft = selectedMarker
    ? ratioToLeft(selectedMarker.timestamp_ms, detailWindow.start_ms, detailWindow.end_ms, detailWidth)
    : 0;
  const detailMarkerVisible = Boolean(
    selectedMarker &&
    selectedMarker.timestamp_ms >= detailWindow.start_ms &&
    selectedMarker.timestamp_ms <= detailWindow.end_ms,
  );

  const detailBins = detailWindow.bins;
  const detailBarWidth = detailBins.length > 0 && detailWidth > 0
    ? Math.max(1, Math.floor(detailWidth / detailBins.length))
    : 2;
  const compactVideoHeight = Math.min(246, Math.max(196, Math.round(windowHeight * 0.3)));
  const compactVideoWidth = Math.round(compactVideoHeight * (9 / 16));
  const topSectionMinHeight = Math.max(compactVideoHeight + 56, 250);

  return (
    <View style={styles.root}>
      <StatusBar hidden barStyle="light-content" backgroundColor="#0d0d0d" />
      {loading ? (
        <View style={styles.loadingBox}>
          <ActivityIndicator color="#f5c76d" />
          <Text style={styles.loadingTxt}>Preparing waveform and markers...</Text>
        </View>
      ) : (
        <View style={styles.screen}>
          <View style={styles.header}>
            <View style={styles.headerMain}>
              <TouchableOpacity onPress={onBack} style={styles.backBtn}>
                <Text style={styles.backTxt}>Back</Text>
              </TouchableOpacity>
              <Text style={styles.headerTitle}>Review take</Text>
              <Text style={styles.headerSub}>
                {event.scenario_id} | take {event.take_index} | {Math.round(durationMs / 1000)}s
              </Text>
            </View>
            <View style={styles.headerInfo}>
              <Text style={styles.revisionTxt}>{REVIEW_UI_REVISION}</Text>
              <Text style={styles.headerHint}>White = playhead | Green/orange = marker | Gray = peak guide</Text>
            </View>
          </View>

          <View style={[styles.topSection, { minHeight: topSectionMinHeight }]}>
            {hasVideo && videoFilePath ? (
              <View style={styles.topRow}>
                <View style={styles.videoColumn}>
                  <Text style={styles.sectionLabel}>REVIEW VIDEO</Text>
                  <View style={[styles.videoFramePortrait, { width: compactVideoWidth, height: compactVideoHeight }]}>
                    <Video
                      ref={videoRef}
                      source={{ uri: `file://${videoFilePath}` }}
                      style={styles.videoPlayer}
                      paused={!(playbackMode === 'playing_full_take' || playbackMode === 'playing_preview')}
                      rate={playbackRate}
                      muted
                      repeat={false}
                      controls={false}
                      resizeMode="contain"
                      onError={error => {
                        Alert.alert('Video error', `Could not load review video: ${String(error)}`);
                      }}
                    />
                  </View>
                  <Text style={styles.videoMeta}>
                    Video {formatMs(videoPlaybackMs)} | Audio {formatMs(playbackPositionMs)}
                  </Text>
                </View>

                <View style={styles.controlColumn}>
                  <Text style={styles.sectionLabel}>PLAYBACK</Text>
                  <View style={styles.rateRow}>
                    {([1, 0.5, 0.25] as PlaybackRate[]).map(rate => {
                      const active = playbackRate === rate;
                      return (
                        <TouchableOpacity
                          key={`rate-${rate}`}
                          style={[styles.rateBtn, active && styles.rateBtnActive]}
                          onPress={() => handleSetPlaybackRate(rate)}
                        >
                          <Text style={[styles.rateTxt, active && styles.rateTxtActive]}>{rate}x</Text>
                        </TouchableOpacity>
                      );
                    })}
                  </View>
                  <View style={styles.buttonRow}>
                    <TouchableOpacity style={[styles.controlBtn, styles.primaryControlBtn]} onPress={handlePlayFromHere}>
                      <Text style={styles.primaryControlTxt}>Play</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.controlBtn} onPress={handlePause}>
                      <Text style={styles.controlTxt}>Pause</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.controlBtn} onPress={handlePlaySelectedMarker}>
                      <Text style={styles.controlTxt}>Play marker</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.controlBtn} onPress={handleAddMarkerHere}>
                      <Text style={styles.controlTxt}>Add marker</Text>
                    </TouchableOpacity>
                  </View>
                  <Text style={styles.statusTxt}>{playbackMode} | {formatMs(playbackPositionMs)}</Text>

                  <Text style={styles.sectionLabel}>SELECTED MARKER</Text>
                  {selectedMarker ? (
                    <>
                      <Text style={styles.markerTitle}>{Math.round(selectedMarker.timestamp_ms)} ms | {selectedMarker.source}</Text>
                      <Text style={styles.markerMeta}>
                        {selectedMarkerIndex + 1}/{orderedMarkers.length} | Suggested {labelText(selectedMarker.suggested_label)} | Current {labelText(selectedMarker.final_label)}
                      </Text>
                      <View style={styles.buttonRow}>
                        <TouchableOpacity
                          style={styles.controlBtn}
                          onPress={() => handleJumpToMarker(-1)}
                          disabled={selectedMarkerIndex <= 0}
                        >
                          <Text style={[styles.controlTxt, selectedMarkerIndex <= 0 && styles.disabledTxt]}>Prev</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                          style={styles.controlBtn}
                          onPress={() => handleJumpToMarker(1)}
                          disabled={selectedMarkerIndex >= orderedMarkers.length - 1}
                        >
                          <Text style={[styles.controlTxt, selectedMarkerIndex >= orderedMarkers.length - 1 && styles.disabledTxt]}>Next</Text>
                        </TouchableOpacity>
                      </View>
                      <View style={styles.labelRow}>
                        {(['racket_contact', 'not_racket_contact', 'ignore'] as AudioReviewLabel[]).map(label => {
                          const active = selectedMarker.final_label === label;
                          return (
                            <TouchableOpacity
                              key={label}
                              style={[
                                styles.labelBtn,
                                active && { borderColor: labelColor(label), backgroundColor: '#161616' },
                              ]}
                              onPress={() => updateSelectedMarker(marker => ({ ...marker, final_label: label }))}
                            >
                              <Text style={[styles.labelBtnTxt, active && { color: labelColor(label) }]}>
                                {labelText(label)}
                              </Text>
                            </TouchableOpacity>
                          );
                        })}
                      </View>
                      <View style={styles.buttonRow}>
                        <TouchableOpacity style={styles.controlBtn} onPress={() => handleNudgeMarker(-NUDGE_STEP_MS)}>
                          <Text style={styles.controlTxt}>-10</Text>
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.controlBtn} onPress={() => handleNudgeMarker(NUDGE_STEP_MS)}>
                          <Text style={styles.controlTxt}>+10</Text>
                        </TouchableOpacity>
                        <TouchableOpacity style={[styles.controlBtn, styles.snapBtn]} onPress={handleSnapSelectedMarker}>
                          <Text style={styles.snapTxt}>Snap</Text>
                        </TouchableOpacity>
                        <TouchableOpacity style={[styles.controlBtn, styles.deleteBtn]} onPress={handleDeleteSelectedMarker}>
                          <Text style={styles.deleteTxt}>Delete</Text>
                        </TouchableOpacity>
                      </View>
                      <Text style={styles.countMeta}>
                        R {markerCounts.racket} | N {markerCounts.notRacket} | I {markerCounts.ignore}
                      </Text>
                    </>
                  ) : (
                    <Text style={styles.emptyTxt}>Select a marker or add one at the playhead.</Text>
                  )}
                </View>
              </View>
            ) : (
              <View style={styles.controlColumnSolo}>
                <Text style={styles.sectionLabel}>PLAYBACK</Text>
                <View style={styles.rateRow}>
                  {([1, 0.5, 0.25] as PlaybackRate[]).map(rate => {
                    const active = playbackRate === rate;
                    return (
                      <TouchableOpacity
                        key={`rate-${rate}`}
                        style={[styles.rateBtn, active && styles.rateBtnActive]}
                        onPress={() => handleSetPlaybackRate(rate)}
                      >
                        <Text style={[styles.rateTxt, active && styles.rateTxtActive]}>{rate}x</Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>
                <View style={styles.buttonRow}>
                  <TouchableOpacity style={[styles.controlBtn, styles.primaryControlBtn]} onPress={handlePlayFromHere}>
                    <Text style={styles.primaryControlTxt}>Play</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.controlBtn} onPress={handlePause}>
                    <Text style={styles.controlTxt}>Pause</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.controlBtn} onPress={handlePlaySelectedMarker}>
                    <Text style={styles.controlTxt}>Play marker</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.controlBtn} onPress={handleAddMarkerHere}>
                    <Text style={styles.controlTxt}>Add marker</Text>
                  </TouchableOpacity>
                </View>
                <Text style={styles.statusTxt}>{playbackMode} | {formatMs(playbackPositionMs)}</Text>
              </View>
            )}
          </View>

          <View style={styles.editorCard}>
            <Text style={styles.sectionLabel}>EDITOR</Text>
            <View
              style={styles.overviewMini}
              onLayout={eventData => setOverviewWidth(eventData.nativeEvent.layout.width)}
              {...overviewScrubResponder.panHandlers}
            >
              <Pressable
                style={StyleSheet.absoluteFill}
                onPress={eventData => handleOverviewPress(eventData.nativeEvent.locationX)}
              />
              <View style={styles.overviewWaveformRow}>
                {overviewWindow.bins.map((bin, index) => (
                  <View
                    key={`overview-bin-${index}`}
                    style={[
                      styles.overviewBar,
                      {
                        height: Math.max(6, bin * 48),
                        opacity: 0.3 + bin * 0.7,
                      },
                    ]}
                  />
                ))}
              </View>

              {orderedMarkers.map(marker => {
                if (marker.timestamp_ms < overviewWindow.start_ms || marker.timestamp_ms > overviewWindow.end_ms) {
                  return null;
                }
                const left = ratioToLeft(
                  marker.timestamp_ms,
                  overviewWindow.start_ms,
                  overviewWindow.end_ms,
                  overviewWidth,
                );
                const accent = labelColor(marker.final_label);
                const isSelected = marker.id === selectedMarkerId;
                return (
                  <TouchableOpacity
                    key={marker.id}
                    activeOpacity={0.9}
                    style={[
                      styles.overviewMarkerHitbox,
                      {
                        left: Math.max(0, left - 14),
                        borderColor: isSelected ? accent : 'transparent',
                        backgroundColor: isSelected ? '#181818' : 'transparent',
                      },
                    ]}
                    onPress={() => handleSelectMarker(marker)}
                    {...(isSelected ? overviewMarkerResponder.panHandlers : {})}
                  >
                    <View style={[styles.markerStem, { backgroundColor: accent }]} />
                    <View style={[styles.markerDot, { backgroundColor: accent }]} />
                  </TouchableOpacity>
                );
              })}

              <View
                style={[styles.playheadHitbox, { left: Math.max(0, overviewPlayheadLeft - 18) }]}
                {...overviewPlayheadResponder.panHandlers}
              >
                  <View style={styles.playheadLine} />
                  <View style={styles.playheadKnob} />
                </View>
              </View>
              <Text style={styles.windowMeta}>
                Overview {formatMs(overviewWindow.start_ms)} to {formatMs(overviewWindow.end_ms)}
              </Text>

              <View
                style={styles.detailSurface}
              onLayout={eventData => setDetailWidth(eventData.nativeEvent.layout.width)}
            >
              <Pressable
                style={StyleSheet.absoluteFill}
                onPress={eventData => handleDetailPress(eventData.nativeEvent.locationX)}
              />
              <View style={styles.waveformRow}>
                {detailBins.map((bin, index) => (
                  <View
                    key={`detail-bin-${index}`}
                    style={[
                      styles.detailBar,
                      {
                        width: detailBarWidth,
                        height: Math.max(10, bin * 72),
                        opacity: 0.28 + bin * 0.72,
                      },
                    ]}
                  />
                ))}
              </View>

              <View style={[styles.peakGuide, { left: Math.max(0, detailPeakLeft) }]} />
              <View style={[styles.detailPlayheadLine, { left: Math.max(0, detailPlayheadLeft) }]} />

              {detailMarkerVisible && selectedMarker && (
                <View
                  style={[styles.detailMarkerHitbox, { left: Math.max(0, detailMarkerLeft - 22) }]}
                  {...detailMarkerResponder.panHandlers}
                >
                  <View
                    style={[
                      styles.detailMarkerLine,
                      { backgroundColor: labelColor(selectedMarker.final_label) },
                    ]}
                  />
                </View>
              )}
            </View>
            <Text style={styles.windowMeta}>
              Detail {formatMs(detailWindow.start_ms)} to {formatMs(detailWindow.end_ms)} | Peak {formatMs(detailWindow.peak_ms)}
            </Text>
          </View>

          <View style={styles.helpRow}>
            <Text style={styles.helpRowText}>Place marker on first clear rise, before the gray peak.</Text>
          </View>

          <View style={[styles.footerActions, { marginBottom: Math.max(insets.bottom, 12) }]}>
              <TouchableOpacity style={[styles.footerBtn, styles.saveBtn]} onPress={handleSave} disabled={saving}>
                <Text style={styles.saveBtnTxt}>{saving ? 'Saving...' : 'Save take'}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.footerBtn, styles.discardBtn]} onPress={handleDiscard}>
                <Text style={styles.discardBtnTxt}>Discard take</Text>
              </TouchableOpacity>
            </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#0d0d0d',
  },
  screen: {
    flex: 1,
    paddingHorizontal: 16,
    paddingTop: 10,
    paddingBottom: 12,
  },
  loadingBox: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  loadingTxt: { color: '#aaa', fontSize: 13 },
  header: {
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: 6,
    marginBottom: 8,
  },
  headerMain: { flex: 1 },
  backBtn: { paddingBottom: 2 },
  backTxt: { color: '#4a9eff', fontSize: 14, fontWeight: '700' },
  headerTitle: { color: '#fff', fontSize: 22, fontWeight: '800' },
  headerSub: { color: '#8a8a8a', fontSize: 11, marginTop: 2 },
  headerInfo: {
    width: '100%',
    alignItems: 'flex-start',
  },
  revisionTxt: { color: '#f5c76d', fontSize: 12, fontWeight: '800' },
  headerHint: { color: '#7c7c7c', fontSize: 10, marginTop: 2, textAlign: 'left', lineHeight: 14 },
  topSection: {
    borderRadius: 16,
    backgroundColor: '#111',
    padding: 12,
    marginBottom: 10,
  },
  topRow: {
    flexDirection: 'row',
    gap: 10,
    flex: 1,
  },
  videoColumn: {
    alignItems: 'center',
    gap: 6,
  },
  controlColumn: {
    flex: 1,
  },
  controlColumnSolo: {
    flex: 1,
  },
  card: {
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 12,
  },
  editorCard: {
    flex: 1,
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 12,
    minHeight: 0,
  },
  sectionLabel: { color: '#777', fontSize: 9, letterSpacing: 1.8, marginBottom: 6 },
  helperTxt: { color: '#8e8e8e', fontSize: 11, lineHeight: 15, marginBottom: 8 },
  videoFramePortrait: {
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#050505',
  },
  videoPlayer: {
    width: '100%',
    height: '100%',
    backgroundColor: '#050505',
  },
  videoMeta: {
    color: '#6f6f6f',
    fontSize: 10,
    fontFamily: 'monospace',
    textAlign: 'center',
  },
  controlsBlock: {
    gap: 6,
  },
  overviewMini: {
    backgroundColor: '#0d0d0d',
    borderRadius: 14,
    paddingHorizontal: 10,
    paddingVertical: 8,
    overflow: 'hidden',
    minHeight: 112,
    marginBottom: 10,
  },
  detailSurface: {
    flex: 1,
    backgroundColor: '#0d0d0d',
    borderRadius: 14,
    paddingHorizontal: 10,
    paddingVertical: 8,
    overflow: 'hidden',
    minHeight: 72,
  },
  waveformRow: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 1,
  },
  overviewWaveformRow: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 0,
  },
  overviewBar: {
    width: 2,
    borderRadius: 2,
    backgroundColor: '#4a9eff',
    alignSelf: 'center',
  },
  detailBar: {
    borderRadius: 2,
    backgroundColor: '#4a9eff',
    alignSelf: 'center',
  },
  overviewMarkerHitbox: {
    position: 'absolute',
    top: 6,
    bottom: 6,
    width: 32,
    alignItems: 'center',
    justifyContent: 'flex-start',
    borderWidth: 1,
    borderRadius: 10,
    zIndex: 4,
    elevation: 4,
  },
  markerStem: { width: 3, flex: 1, marginTop: 4 },
  markerDot: { width: 10, height: 10, borderRadius: 5, marginBottom: 6 },
  playheadHitbox: {
    position: 'absolute',
    top: 6,
    bottom: 6,
    width: 44,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 5,
    elevation: 5,
  },
  playheadLine: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
    backgroundColor: '#ffffff',
  },
  playheadKnob: {
    position: 'absolute',
    top: -2,
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#111',
  },
  detailPlayheadLine: {
    position: 'absolute',
    top: 6,
    bottom: 6,
    width: 2,
    backgroundColor: '#ffffff',
  },
  detailMarkerHitbox: {
    position: 'absolute',
    top: 6,
    bottom: 6,
    width: 56,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 4,
    elevation: 4,
  },
  detailMarkerLine: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 3,
    borderRadius: 2,
  },
  peakGuide: {
    position: 'absolute',
    top: 6,
    bottom: 6,
    width: 2,
    backgroundColor: '#545454',
    opacity: 0.8,
  },
  windowMeta: { color: '#7a7a7a', fontSize: 10, marginTop: 6, lineHeight: 14 },
  rateRow: { flexDirection: 'row', gap: 6, flexWrap: 'wrap', marginBottom: 4 },
  rateBtn: {
    borderRadius: 9,
    borderWidth: 1,
    borderColor: '#282828',
    backgroundColor: '#171717',
    paddingVertical: 6,
    paddingHorizontal: 10,
  },
  rateBtnActive: {
    borderColor: '#2ecc71',
    backgroundColor: '#0d2d1a',
  },
  rateTxt: { color: '#bdbdbd', fontSize: 11, fontWeight: '700' },
  rateTxtActive: { color: '#2ecc71' },
  buttonRow: { flexDirection: 'row', gap: 6, flexWrap: 'wrap', marginTop: 4 },
  controlBtn: {
    borderRadius: 10,
    backgroundColor: '#1a1a1a',
    paddingVertical: 7,
    paddingHorizontal: 9,
  },
  primaryControlBtn: { backgroundColor: '#0d2d1a' },
  controlTxt: { color: '#e0e0e0', fontSize: 11, fontWeight: '700' },
  primaryControlTxt: { color: '#2ecc71', fontSize: 11, fontWeight: '800' },
  statusTxt: { color: '#8a8a8a', fontSize: 10, marginTop: 4 },
  markerTitle: { color: '#fff', fontSize: 14, fontWeight: '800' },
  markerMeta: { color: '#8a8a8a', fontSize: 10, lineHeight: 13, marginTop: 2 },
  countMeta: { color: '#6f6f6f', fontSize: 10, lineHeight: 13, marginTop: 4 },
  labelRow: { flexDirection: 'row', gap: 6, flexWrap: 'wrap', marginTop: 4 },
  labelBtn: {
    borderRadius: 9,
    borderWidth: 1,
    borderColor: '#262626',
    paddingVertical: 7,
    paddingHorizontal: 9,
    backgroundColor: '#0d0d0d',
  },
  labelBtnTxt: { color: '#d0d0d0', fontWeight: '700', fontSize: 11 },
  snapBtn: { backgroundColor: '#17253c' },
  snapTxt: { color: '#a8c2ff', fontSize: 11, fontWeight: '700' },
  deleteBtn: { backgroundColor: '#2d0d0d' },
  deleteTxt: { color: '#ff9f9f', fontSize: 11, fontWeight: '700' },
  disabledTxt: { color: '#666' },
  emptyTxt: { color: '#777', fontSize: 11, lineHeight: 15 },
  helpRow: {
    paddingHorizontal: 4,
    paddingVertical: 6,
  },
  helpRowText: {
    color: '#757575',
    fontSize: 10,
    lineHeight: 13,
  },
  footerActions: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 4,
  },
  footerBtn: {
    flex: 1,
    borderRadius: 13,
    paddingVertical: 13,
    alignItems: 'center',
  },
  saveBtn: { backgroundColor: '#0d2d0d' },
  saveBtnTxt: { color: '#2ecc71', fontSize: 14, fontWeight: '800' },
  discardBtn: { backgroundColor: '#2d0d0d' },
  discardBtnTxt: { color: '#ff7f7f', fontSize: 14, fontWeight: '800' },
});
