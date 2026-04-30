import React, { useRef } from 'react';
import {
  PanResponder,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  valueFormatter?: (value: number) => string;
  leftHint?: string;
  rightHint?: string;
}

export function DebugSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  valueFormatter = current => current.toString(),
  leftHint,
  rightHint,
}: Props) {
  const trackRef = useRef<View>(null);
  const trackX = useRef(0);
  const trackW = useRef(1);
  const thumbSize = 22;

  const applyX = (pageX: number) => {
    const ratio = Math.max(0, Math.min(1, (pageX - trackX.current) / trackW.current));
    const raw = min + ratio * (max - min);
    const snapped = Math.round(raw / step) * step;
    const next = Math.max(min, Math.min(max, Number(snapped.toFixed(4))));
    onChange(next);
  };

  const pan = useRef(PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: (_event, gestureState) =>
      Math.abs(gestureState.dx) > 2 && Math.abs(gestureState.dx) >= Math.abs(gestureState.dy),
    onPanResponderTerminationRequest: () => false,
    onPanResponderGrant: e => applyX(e.nativeEvent.pageX),
    onPanResponderMove: e => applyX(e.nativeEvent.pageX),
    onPanResponderRelease: e => applyX(e.nativeEvent.pageX),
  })).current;

  const onLayout = () => {
    trackRef.current?.measureInWindow((x, _y, width) => {
      trackX.current = x;
      trackW.current = Math.max(width, 1);
    });
  };

  const fillPct = ((value - min) / (max - min)) * 100;

  return (
    <View style={s.wrap}>
      <View style={s.header}>
        <Text style={s.label}>{label}</Text>
        <Text style={s.value}>{valueFormatter(value)}</Text>
      </View>

      <View
        ref={trackRef}
        onLayout={onLayout}
        style={[s.touchArea, { marginHorizontal: thumbSize / 2 }]}
        {...pan.panHandlers}
      >
        <Pressable style={s.touchFill} hitSlop={8} onPress={e => applyX(e.nativeEvent.pageX)}>
          <View style={s.track}>
            <View style={[s.fill, { width: `${fillPct}%` }]} />
            <View
              style={[
                s.thumb,
                {
                  left: `${fillPct}%` as any,
                  width: thumbSize,
                  height: thumbSize,
                  borderRadius: thumbSize / 2,
                  top: -(thumbSize / 2 - 3),
                },
              ]}
            />
          </View>
        </Pressable>
      </View>

      {(leftHint || rightHint) && (
        <View style={s.hints}>
          <Text style={s.hint}>{leftHint ?? ''}</Text>
          <Text style={s.hint}>{rightHint ?? ''}</Text>
        </View>
      )}
    </View>
  );
}

const s = StyleSheet.create({
  wrap: { marginTop: 14 },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  label: {
    color: '#8a8a8a',
    fontSize: 10,
    letterSpacing: 1.6,
  },
  value: {
    color: '#f2f2f2',
    fontSize: 12,
    fontFamily: 'monospace',
  },
  track: {
    height: 6,
    backgroundColor: '#1a1a1a',
    borderRadius: 4,
  },
  touchArea: {
    height: 34,
    justifyContent: 'center',
  },
  touchFill: {
    justifyContent: 'center',
  },
  fill: {
    height: '100%',
    backgroundColor: '#f5c76d',
    borderRadius: 4,
  },
  thumb: {
    position: 'absolute',
    backgroundColor: '#f5c76d',
    borderWidth: 2,
    borderColor: '#1b1406',
    marginLeft: -11,
  },
  hints: {
    marginTop: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  hint: {
    color: '#666',
    fontSize: 10,
  },
});
