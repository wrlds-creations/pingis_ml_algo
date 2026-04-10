import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import type { PlayerSetup } from './types';

interface Props {
  onDone: (setup: PlayerSetup) => void;
  onAudioMode?: (setup: PlayerSetup) => void;
  onLiveMode?: (setup: PlayerSetup) => void;
}

export function SetupScreen({ onDone, onAudioMode, onLiveMode }: Props) {
  const [name, setName] = useState('');
  const [handedness, setHandedness] = useState<'right' | 'left'>('right');

  const canContinue = name.trim().length > 0;

  return (
    <KeyboardAvoidingView
      style={s.root}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <StatusBar barStyle="light-content" backgroundColor="#0d0d0d" />

      <View style={s.content}>
        <Text style={s.title}>Pingis{'\n'}Datainsamling</Text>
        <Text style={s.subtitle}>Ställ in din profil</Text>

        {/* Spelarnamn */}
        <Text style={s.label}>NAMN</Text>
        <TextInput
          style={s.input}
          value={name}
          onChangeText={setName}
          placeholder="Ditt namn"
          placeholderTextColor="#555"
          autoCapitalize="words"
          returnKeyType="done"
        />

        {/* Händhet */}
        <Text style={s.label}>SPELHAND</Text>
        <Text style={s.helpText}>
          Vilken hand håller du racket med?
        </Text>
        <View style={s.handRow}>
          <TouchableOpacity
            style={[s.handBtn, handedness === 'right' && s.handBtnOn]}
            onPress={() => setHandedness('right')}
            activeOpacity={0.7}
          >
            <Text style={s.handIcon}>🏓</Text>
            <Text style={[s.handBtnTxt, handedness === 'right' && s.handBtnTxtOn]}>
              HÖGER HAND
            </Text>
            <Text style={s.handSub}>Forehand = svänger åt vänster</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[s.handBtn, handedness === 'left' && s.handBtnOn]}
            onPress={() => setHandedness('left')}
            activeOpacity={0.7}
          >
            <Text style={s.handIcon}>🏓</Text>
            <Text style={[s.handBtnTxt, handedness === 'left' && s.handBtnTxtOn]}>
              VÄNSTER HAND
            </Text>
            <Text style={s.handSub}>Forehand = svänger åt höger</Text>
          </TouchableOpacity>
        </View>

        {/* Info-ruta */}
        <View style={s.infoBox}>
          <Text style={s.infoTxt}>
            Händhet sparas i varje session-fil och hjälper ML-modellen att
            förstå rörelsemönstret korrekt för din hand.
          </Text>
        </View>

        {/* Fortsätt-knappar */}
        <TouchableOpacity
          style={[s.continueBtn, !canContinue && s.continueBtnOff]}
          onPress={() => canContinue && onDone({ name: name.trim(), handedness })}
          activeOpacity={0.7}
        >
          <Text style={[s.continueTxt, !canContinue && s.continueTxtOff]}>
            Fortsätt → Kalibrering
          </Text>
        </TouchableOpacity>

        {onAudioMode && (
          <TouchableOpacity
            style={[s.audioBtn, !canContinue && s.audioBtnOff]}
            onPress={() => canContinue && onAudioMode({ name: name.trim(), handedness })}
            activeOpacity={0.7}
          >
            <Text style={[s.audioTxt, !canContinue && s.audioTxtOff]}>
              Ljud-insamling
            </Text>
          </TouchableOpacity>
        )}

        {onLiveMode && (
          <TouchableOpacity
            style={[s.liveBtn, !canContinue && s.liveBtnOff]}
            onPress={() => canContinue && onLiveMode({ name: name.trim(), handedness })}
            activeOpacity={0.7}
          >
            <Text style={[s.liveTxt, !canContinue && s.liveTxtOff]}>
              Live-klassificering
            </Text>
          </TouchableOpacity>
        )}
      </View>
    </KeyboardAvoidingView>
  );
}

const s = StyleSheet.create({
  root:          { flex: 1, backgroundColor: '#0d0d0d' },
  content:       { flex: 1, padding: 24, justifyContent: 'center' },
  title:         { color: '#fff', fontSize: 32, fontWeight: '800', marginBottom: 4 },
  subtitle:      { color: '#777', fontSize: 15, marginBottom: 36 },

  label:         { color: '#777', fontSize: 10, letterSpacing: 2, marginBottom: 8 },
  helpText:      { color: '#888', fontSize: 12, marginBottom: 10 },

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

  handRow:       { flexDirection: 'row', gap: 12, marginBottom: 20 },
  handBtn: {
    flex: 1,
    backgroundColor: '#111',
    borderWidth: 1,
    borderColor: '#222',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  handBtnOn:     { borderColor: '#4a9eff', backgroundColor: '#0d1f33' },
  handIcon:      { fontSize: 24, marginBottom: 8 },
  handBtnTxt:    { color: '#777', fontWeight: '700', fontSize: 12, letterSpacing: 1, marginBottom: 4 },
  handBtnTxtOn:  { color: '#4a9eff' },
  handSub:       { color: '#666', fontSize: 10, textAlign: 'center' },

  infoBox:       { backgroundColor: '#111', borderRadius: 10, padding: 14, marginBottom: 28 },
  infoTxt:       { color: '#777', fontSize: 12, lineHeight: 18 },

  continueBtn: {
    backgroundColor: '#0d2d0d',
    borderRadius: 12,
    padding: 18,
    alignItems: 'center',
  },
  continueBtnOff: { backgroundColor: '#111' },
  continueTxt:    { color: '#2ecc71', fontWeight: '700', fontSize: 16, letterSpacing: 1 },
  continueTxtOff: { color: '#2a2a2a' },

  audioBtn: {
    backgroundColor: '#0d1f33',
    borderRadius: 12,
    padding: 18,
    alignItems: 'center',
    marginTop: 10,
  },
  audioBtnOff: { backgroundColor: '#111' },
  audioTxt:    { color: '#4a9eff', fontWeight: '700', fontSize: 16, letterSpacing: 1 },
  audioTxtOff: { color: '#2a2a2a' },

  liveBtn: {
    backgroundColor: '#1a0d2d',
    borderRadius: 12,
    padding: 18,
    alignItems: 'center',
    marginTop: 10,
  },
  liveBtnOff: { backgroundColor: '#111' },
  liveTxt:    { color: '#a855f7', fontWeight: '700', fontSize: 16, letterSpacing: 1 },
  liveTxtOff: { color: '#2a2a2a' },
});
