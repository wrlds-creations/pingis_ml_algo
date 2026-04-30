import React, { useState } from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import type { PlayerSetup } from './types';

interface Props {
  onCollectionMode: (setup: PlayerSetup) => void;
  onAudioMode?: (setup: PlayerSetup) => void;
  onBounceAudioImuMode?: (setup: PlayerSetup) => void;
  onLiveMode?: (setup: PlayerSetup) => void;
  onBounceFreeMode?: (setup: PlayerSetup) => void;
  onBounceAlternatingMode?: (setup: PlayerSetup) => void;
}

export function SetupScreen({
  onCollectionMode,
  onAudioMode,
  onBounceAudioImuMode,
  onLiveMode,
  onBounceFreeMode,
  onBounceAlternatingMode,
}: Props) {
  const [name, setName] = useState('');
  const [handedness, setHandedness] = useState<'right' | 'left'>('right');

  const canContinue = name.trim().length > 0;
  const setup = { name: name.trim(), handedness };

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <StatusBar barStyle="light-content" backgroundColor="#0d0d0d" />

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={styles.title}>Pingis{'\n'}Collector</Text>
        <Text style={styles.subtitle}>Profile, data collection, and test modes</Text>

        <Text style={styles.label}>NAME</Text>
        <TextInput
          style={styles.input}
          value={name}
          onChangeText={setName}
          placeholder="Your name"
          placeholderTextColor="#555"
          autoCapitalize="words"
          returnKeyType="done"
        />

        <Text style={styles.label}>PLAYING HAND</Text>
        <Text style={styles.helpText}>Which hand holds the racket?</Text>
        <View style={styles.handRow}>
          <TouchableOpacity
            style={[styles.handBtn, handedness === 'right' && styles.handBtnOn]}
            onPress={() => setHandedness('right')}
            activeOpacity={0.7}
          >
            <Text style={styles.handIcon}>R</Text>
            <Text style={[styles.handBtnTxt, handedness === 'right' && styles.handBtnTxtOn]}>
              RIGHT HAND
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.handBtn, handedness === 'left' && styles.handBtnOn]}
            onPress={() => setHandedness('left')}
            activeOpacity={0.7}
          >
            <Text style={styles.handIcon}>L</Text>
            <Text style={[styles.handBtnTxt, handedness === 'left' && styles.handBtnTxtOn]}>
              LEFT HAND
            </Text>
          </TouchableOpacity>
        </View>

        <View style={styles.infoBox}>
          <Text style={styles.infoTxt}>
            Choose a focused mode. Bounce contact is the current priority, and Stroke debug is
            paused from the main test flow until the bounce baseline is stable.
          </Text>
        </View>

        <Text style={styles.sectionLabel}>TEST MODES</Text>
        {onBounceFreeMode && (
          <ModeButton
            disabled={!canContinue}
            title="Studs fritt"
            subtitle="Count racket bounces and show FH / BH / Uncertain"
            colorStyle="gold"
            onPress={() => canContinue && onBounceFreeMode(setup)}
          />
        )}
        {onBounceAlternatingMode && (
          <ModeButton
            disabled={!canContinue}
            title="Studs vaxla sida"
            subtitle="Count only correct FH / BH alternation"
            colorStyle="orange"
            onPress={() => canContinue && onBounceAlternatingMode(setup)}
          />
        )}
        {onLiveMode && (
          <ModeButton
            disabled={!canContinue}
            title="Studsdetektor"
            subtitle="Audio-only baseline for binary racket contact"
            colorStyle="green"
            onPress={() => canContinue && onLiveMode(setup)}
          />
        )}

        <Text style={styles.sectionLabel}>DATA</Text>
        <ModeButton
          disabled={!canContinue}
          title="Datainsamling"
          subtitle="Collect labeled IMU sessions for model training"
          colorStyle="darkGreen"
          onPress={() => canContinue && onCollectionMode(setup)}
        />
        {onAudioMode && (
          <ModeButton
            disabled={!canContinue}
            title="Ljud-insamling"
            subtitle="Record labeled bounce sounds"
            colorStyle="purple"
            onPress={() => canContinue && onAudioMode(setup)}
          />
        )}
        {onBounceAudioImuMode && (
          <ModeButton
            disabled={!canContinue}
            title="Studs audio + IMU"
            subtitle="Review audio as usual, but save synchronized AirHive IMU for a future bounce model"
            colorStyle="blue"
            onPress={() => canContinue && onBounceAudioImuMode(setup)}
          />
        )}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

function ModeButton({
  title,
  subtitle,
  disabled,
  onPress,
  colorStyle,
}: {
  title: string;
  subtitle: string;
  disabled: boolean;
  onPress: () => void;
  colorStyle: 'gold' | 'orange' | 'blue' | 'green' | 'darkGreen' | 'purple';
}) {
  return (
    <TouchableOpacity
      style={[
        styles.modeBtn,
        styles[`${colorStyle}Mode`],
        disabled && styles.modeBtnOff,
      ]}
      onPress={onPress}
      activeOpacity={0.7}
      disabled={disabled}
    >
      <Text style={[styles.modeTitle, disabled && styles.modeTxtOff]}>{title}</Text>
      <Text style={[styles.modeSubtitle, disabled && styles.modeTxtOff]}>{subtitle}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0d0d0d' },
  scroll: { flex: 1 },
  content: { flexGrow: 1, padding: 24, paddingBottom: 40, justifyContent: 'center' },
  title: { color: '#fff', fontSize: 32, fontWeight: '800', marginBottom: 4 },
  subtitle: { color: '#777', fontSize: 15, marginBottom: 32 },
  label: { color: '#777', fontSize: 10, letterSpacing: 2, marginBottom: 8 },
  helpText: { color: '#888', fontSize: 12, marginBottom: 10 },
  input: {
    backgroundColor: '#141414',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 10,
    padding: 16,
    color: '#fff',
    fontSize: 18,
    marginBottom: 28,
  },
  handRow: { flexDirection: 'row', gap: 12, marginBottom: 20 },
  handBtn: {
    flex: 1,
    backgroundColor: '#111',
    borderWidth: 1,
    borderColor: '#222',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  handBtnOn: { borderColor: '#4a9eff', backgroundColor: '#0d1f33' },
  handIcon: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#1f1f1f',
    color: '#f5f5f5',
    textAlign: 'center',
    textAlignVertical: 'center',
    fontWeight: '700',
    marginBottom: 8,
  },
  handBtnTxt: { color: '#777', fontWeight: '700', fontSize: 12, letterSpacing: 1 },
  handBtnTxtOn: { color: '#4a9eff' },
  infoBox: { backgroundColor: '#111', borderRadius: 10, padding: 14, marginBottom: 24 },
  infoTxt: { color: '#777', fontSize: 12, lineHeight: 18 },
  sectionLabel: { color: '#666', fontSize: 10, letterSpacing: 2, marginBottom: 10, marginTop: 4 },
  modeBtn: {
    borderRadius: 14,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#222',
  },
  modeBtnOff: { backgroundColor: '#111', borderColor: '#171717' },
  modeTitle: { color: '#fff', fontWeight: '800', fontSize: 16, marginBottom: 4 },
  modeSubtitle: { color: '#b0b0b0', fontSize: 12, lineHeight: 18 },
  modeTxtOff: { color: '#3a3a3a' },
  goldMode: { backgroundColor: '#2b220d' },
  orangeMode: { backgroundColor: '#2c180c' },
  blueMode: { backgroundColor: '#0d1f33' },
  greenMode: { backgroundColor: '#0f2617' },
  darkGreenMode: { backgroundColor: '#0d2d0d' },
  purpleMode: { backgroundColor: '#21122f' },
});
