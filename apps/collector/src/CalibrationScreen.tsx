import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Alert,
  PermissionsAndroid,
  Platform,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { BleManager } from 'react-native-ble-plx';
import type { BleError, Characteristic, Device } from 'react-native-ble-plx';
import {
  ACCEL_UUID,
  ACCEL_UUID_ALT,
  GYRO_UUID,
  MAG_UUID,
  SERVICE_UUID,
  averageVector,
  magnitude3,
  normalize,
  parsePacket,
} from './airhive';
import type {
  CalibrationData,
  CalibrationMode,
  ImuSample,
  PlayerSetup,
  Vector3,
} from './types';

const STABLE_SAMPLES_NEEDED = 150;
const GYRO_STABLE_THRESHOLD = 5.0;
const POSE_SAMPLES_NEEDED = 20;
const POSE_GYRO_THRESHOLD = 25.0;

type ConnState = 'idle' | 'scanning' | 'connecting' | 'connected';

interface PoseSnapshot {
  accel: Vector3;
  gyro_mag: number;
}

interface Props {
  setup: PlayerSetup;
  mode?: CalibrationMode;
  onCalibrated: (calibration: CalibrationData, device: Device) => void;
  onBack: () => void;
}

const bleManager = new BleManager();

function SensorOnTableDiagram() {
  return (
    <View style={diagram.wrap}>
      <View style={diagram.sensor}>
        <View style={diagram.sensorLed} />
        <Text style={diagram.sensorText}>AirHive</Text>
        <Text style={diagram.sensorSub}>display up</Text>
      </View>
      <View style={diagram.arrowCol}>
        <View style={diagram.arrowLine} />
        <View style={diagram.arrowHead} />
        <Text style={diagram.arrowLabel}>gravity</Text>
      </View>
      <View style={diagram.tableRow}>
        <View style={diagram.tableLine} />
        <Text style={diagram.tableLabel}>TABLE</Text>
        <View style={diagram.tableLine} />
      </View>
    </View>
  );
}

function VectorPreview({ title, vector }: { title: string; vector: Vector3 }) {
  return (
    <View style={preview.wrap}>
      <Text style={preview.title}>{title}</Text>
      <Text style={preview.value}>
        X {vector.x.toFixed(0)}  Y {vector.y.toFixed(0)}  Z {vector.z.toFixed(0)}
      </Text>
    </View>
  );
}

export function CalibrationScreen({ setup, mode = 'table_only', onCalibrated, onBack }: Props) {
  const [connState, setConnState] = useState<ConnState>('idle');
  const [sampleHz, setSampleHz] = useState(0);
  const [stableCount, setStableCount] = useState(0);
  const [tableCalibration, setTableCalibration] = useState<CalibrationData['table'] | null>(null);
  const [forehandPose, setForehandPose] = useState<Vector3 | null>(null);
  const [backhandPose, setBackhandPose] = useState<Vector3 | null>(null);
  const [statusText, setStatusText] = useState<string | null>(null);

  const deviceRef = useRef<Device | null>(null);
  const latestRef = useRef({
    accel: { x: 0, y: 0, z: 0 },
    gyro: { x: 0, y: 0, z: 0 },
    mag: { x: 0, y: 0, z: 0 },
  });

  const stableBufferRef = useRef<ImuSample[]>([]);
  const consecutiveStableRef = useRef(0);
  const poseBufferRef = useRef<PoseSnapshot[]>([]);
  const sampleCountRef = useRef(0);
  const hzTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const tableDoneRef = useRef(false);

  const requestPermissions = useCallback(async () => {
    if (Platform.OS !== 'android') return true;
    const api = Platform.Version as number;
    if (api >= 31) {
      const results = await PermissionsAndroid.requestMultiple([
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
      ]);
      return Object.values(results).every(value => value === PermissionsAndroid.RESULTS.GRANTED);
    }
    const result = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
    );
    return result === PermissionsAndroid.RESULTS.GRANTED;
  }, []);

  const finishTableCalibration = useCallback((samples: ImuSample[]) => {
    const count = samples.length;
    const gravity = {
      x: samples.reduce((sum, sample) => sum + sample.accel_x, 0) / count,
      y: samples.reduce((sum, sample) => sum + sample.accel_y, 0) / count,
      z: samples.reduce((sum, sample) => sum + sample.accel_z, 0) / count,
    };
    const gyroBias = {
      x: samples.reduce((sum, sample) => sum + sample.gyro_x, 0) / count,
      y: samples.reduce((sum, sample) => sum + sample.gyro_y, 0) / count,
      z: samples.reduce((sum, sample) => sum + sample.gyro_z, 0) / count,
    };

    setTableCalibration({
      gravity,
      gyro_bias: gyroBias,
      captured_at: new Date().toISOString(),
    });
    setStatusText(
      mode === 'bounce_sides'
        ? 'Table calibration ready. Capture bounce side poses next.'
        : 'Table calibration ready.',
    );
  }, [mode]);

  const processSampleForTableCalibration = useCallback((sample: ImuSample) => {
    if (tableDoneRef.current) return;

    const gyroMag = magnitude3(sample.gyro_x, sample.gyro_y, sample.gyro_z);
    if (gyroMag < GYRO_STABLE_THRESHOLD) {
      consecutiveStableRef.current += 1;
      stableBufferRef.current.push(sample);
      if (stableBufferRef.current.length > STABLE_SAMPLES_NEEDED) {
        stableBufferRef.current.shift();
      }
      setStableCount(consecutiveStableRef.current);

      if (consecutiveStableRef.current >= STABLE_SAMPLES_NEEDED) {
        tableDoneRef.current = true;
        finishTableCalibration(stableBufferRef.current);
      }
      return;
    }

    consecutiveStableRef.current = 0;
    stableBufferRef.current = [];
    setStableCount(0);
  }, [finishTableCalibration]);

  const handleNotification = useCallback(
    (_error: BleError | null, characteristic: Characteristic | null) => {
      if (!characteristic?.value || !characteristic.uuid) return;
      const parsed = parsePacket(characteristic.uuid, characteristic.value);
      if (!parsed) return;

      const latest = latestRef.current;
      if (parsed.type === 'accel') latest.accel = parsed;
      else if (parsed.type === 'gyro') latest.gyro = parsed;
      else latest.mag = parsed;

      sampleCountRef.current += 1;

      const sample: ImuSample = {
        accel_x: latest.accel.x,
        accel_y: latest.accel.y,
        accel_z: latest.accel.z,
        gyro_x: latest.gyro.x,
        gyro_y: latest.gyro.y,
        gyro_z: latest.gyro.z,
        mag_x: latest.mag.x,
        mag_y: latest.mag.y,
        mag_z: latest.mag.z,
        ts_ms: Date.now(),
      };

      const poseSnapshot: PoseSnapshot = {
        accel: {
          x: sample.accel_x,
          y: sample.accel_y,
          z: sample.accel_z,
        },
        gyro_mag: magnitude3(sample.gyro_x, sample.gyro_y, sample.gyro_z),
      };
      poseBufferRef.current.push(poseSnapshot);
      if (poseBufferRef.current.length > 60) {
        poseBufferRef.current.shift();
      }

      processSampleForTableCalibration(sample);
    },
    [processSampleForTableCalibration],
  );

  const connect = useCallback(async () => {
    if (!(await requestPermissions())) {
      Alert.alert('Permission denied', 'Bluetooth permission is required.');
      return;
    }

    setConnState('scanning');
    setStatusText('Scanning for AirHive...');

    bleManager.startDeviceScan(null, { allowDuplicates: false }, async (error, device) => {
      if (error) {
        setConnState('idle');
        setStatusText(null);
        Alert.alert('Scan error', error.message);
        return;
      }
      if (!device) return;

      const name = device.name ?? '';
      if (!name.toLowerCase().includes('airhive') && !name.toLowerCase().includes('berg')) return;

      bleManager.stopDeviceScan();
      setConnState('connecting');
      setStatusText(`Connecting to ${name}...`);

      try {
        const connected = await device.connect();
        await connected.discoverAllServicesAndCharacteristics();
        deviceRef.current = connected;
        setConnState('connected');
        setStatusText('Sensor connected. Start table calibration.');

        for (const uuid of [ACCEL_UUID, ACCEL_UUID_ALT, GYRO_UUID, MAG_UUID]) {
          try {
            connected.monitorCharacteristicForService(SERVICE_UUID, uuid, handleNotification);
          } catch (_) {}
        }

        let lastCount = 0;
        hzTimerRef.current = setInterval(() => {
          setSampleHz(sampleCountRef.current - lastCount);
          lastCount = sampleCountRef.current;
        }, 1000);

        connected.onDisconnected(() => {
          setConnState('idle');
          setStatusText('Sensor disconnected.');
          setSampleHz(0);
          if (hzTimerRef.current) clearInterval(hzTimerRef.current);
        });
      } catch (error: any) {
        setConnState('idle');
        setStatusText(null);
        Alert.alert('Connection failed', error?.message ?? 'Unknown error');
      }
    });
  }, [handleNotification, requestPermissions]);

  useEffect(() => {
    return () => {
      if (hzTimerRef.current) clearInterval(hzTimerRef.current);
    };
  }, []);

  const resetTableCalibration = useCallback(() => {
    tableDoneRef.current = false;
    stableBufferRef.current = [];
    consecutiveStableRef.current = 0;
    setStableCount(0);
    setTableCalibration(null);
    setForehandPose(null);
    setBackhandPose(null);
    setStatusText('Table calibration reset. Place the sensor flat again.');
  }, []);

  const capturePose = useCallback((side: 'forehand' | 'backhand') => {
    if (!tableCalibration) {
      Alert.alert('Table first', 'Complete table calibration before capturing bounce side poses.');
      return;
    }

    const snapshots = poseBufferRef.current.slice(-POSE_SAMPLES_NEEDED);
    if (snapshots.length < POSE_SAMPLES_NEEDED) {
      Alert.alert('Too few samples', 'Hold the racket still for another second and try again.');
      return;
    }

    const avgGyro = snapshots.reduce((sum, item) => sum + item.gyro_mag, 0) / snapshots.length;
    if (avgGyro > POSE_GYRO_THRESHOLD) {
      Alert.alert('Hold still', 'Keep your wrist still when capturing a bounce side pose.');
      return;
    }

    const averagePose = normalize(averageVector(snapshots.map(item => item.accel)));
    if (side === 'forehand') {
      setForehandPose(averagePose);
      setStatusText('Forehand-side pose captured.');
    } else {
      setBackhandPose(averagePose);
      setStatusText('Backhand-side pose captured.');
    }
  }, [tableCalibration]);

  const start = useCallback(() => {
    if (!deviceRef.current || !tableCalibration) return;

    const calibration: CalibrationData = {
      calibration_id: `cal-${Date.now()}`,
      captured_at: new Date().toISOString(),
      gravity: tableCalibration.gravity,
      gyro_bias: tableCalibration.gyro_bias,
      table: tableCalibration,
      ...(mode === 'bounce_sides' && forehandPose && backhandPose
        ? {
            bounce_sides: {
              forehand: {
                side: 'forehand',
                pose_accel: forehandPose,
                captured_at: new Date().toISOString(),
              },
              backhand: {
                side: 'backhand',
                pose_accel: backhandPose,
                captured_at: new Date().toISOString(),
              },
            },
          }
        : {}),
    };

    onCalibrated(calibration, deviceRef.current);
  }, [backhandPose, forehandPose, mode, onCalibrated, tableCalibration]);

  const tableReady = tableCalibration !== null;
  const bounceReady = mode === 'table_only' || tableReady;
  const bounceSidesCaptured = !!forehandPose && !!backhandPose;
  const progressPct = Math.min((stableCount / STABLE_SAMPLES_NEEDED) * 100, 100);

  return (
    <ScrollView style={styles.root} contentContainerStyle={styles.content}>
      <StatusBar barStyle="light-content" backgroundColor="#0d0d0d" />

      <View style={styles.header}>
        <TouchableOpacity onPress={onBack} style={styles.backBtn}>
          <Text style={styles.backTxt}>Back</Text>
        </TouchableOpacity>
        <Text style={styles.playerTxt}>
          {setup.name} · {setup.handedness === 'right' ? 'Right' : 'Left'} hand
        </Text>
      </View>

      <Text style={styles.title}>Calibration</Text>
      <Text style={styles.subtitle}>
        {mode === 'bounce_sides'
          ? 'AirHive-baseline för Audio plus IMU. FH/BH-poser är valfria hjälpdata och är inte slag-labels.'
          : 'Table calibration for data collection or internal tooling.'}
      </Text>

      <View style={styles.stepBox}>
        <Text style={styles.stepNum}>STEP 1</Text>
        <Text style={styles.stepTitle}>Connect sensor</Text>
        <Text style={styles.stepText}>
          Turn on the AirHive sensor and keep it close to the phone.
        </Text>
        <TouchableOpacity
          style={[styles.actionBtn, connState !== 'idle' && styles.actionBtnBusy]}
          onPress={connect}
          disabled={connState !== 'idle'}
        >
          <Text style={styles.actionBtnTxt}>
            {connState === 'idle'
              ? 'Connect AirHive'
              : connState === 'scanning'
                ? 'Scanning...'
                : connState === 'connecting'
                  ? 'Connecting...'
                  : 'Connected'}
          </Text>
        </TouchableOpacity>
        {statusText && <Text style={styles.infoTxt}>{statusText}</Text>}
      </View>

      {connState === 'connected' && (
        <View style={styles.stepBox}>
          <Text style={styles.stepNum}>STEP 2</Text>
          <Text style={styles.stepTitle}>Table calibration</Text>
          <Text style={styles.stepText}>
            Place the sensor flat on a table with the display facing up. This step finds gravity
            and gyro bias. Keep it completely still for about 3 seconds.
          </Text>
          <SensorOnTableDiagram />
          <Text style={styles.hzTxt}>{sampleHz} Hz</Text>
          <View style={styles.progressBg}>
            <View style={[styles.progressFill, { width: `${progressPct}%` as const }]} />
          </View>
          <Text style={styles.progressTxt}>
            {tableReady
              ? 'Table calibration ready.'
              : `Collecting stable samples... ${stableCount} / ${STABLE_SAMPLES_NEEDED}`}
          </Text>
          <TouchableOpacity style={styles.linkBtn} onPress={resetTableCalibration}>
            <Text style={styles.linkTxt}>Reset table calibration</Text>
          </TouchableOpacity>
        </View>
      )}

      {tableReady && tableCalibration && (
        <VectorPreview title="Table gravity" vector={tableCalibration.gravity} />
      )}

      {mode === 'bounce_sides' && tableReady && (
        <View style={styles.stepBox}>
          <Text style={styles.stepNum}>STEP 3</Text>
          <Text style={styles.stepTitle}>Bounce side poses</Text>
          <Text style={styles.stepText}>
            Håll racket stilla i en tydlig forehand-side och backhand-side för racketstuds om du vill
            spara extra orienteringshjälp. Det här beskriver studs-sida, inte forehand/backhand-slag i spel.
          </Text>

          <TouchableOpacity
            style={[styles.secondaryBtn, forehandPose && styles.secondaryBtnReady]}
            onPress={() => capturePose('forehand')}
          >
            <Text style={[styles.secondaryTxt, forehandPose && styles.secondaryTxtReady]}>
              {forehandPose ? 'Recapture forehand-side pose' : 'Capture forehand-side pose'}
            </Text>
          </TouchableOpacity>
          {forehandPose && <VectorPreview title="Forehand-side pose" vector={forehandPose} />}

          <TouchableOpacity
            style={[styles.secondaryBtn, backhandPose && styles.secondaryBtnReady]}
            onPress={() => capturePose('backhand')}
          >
            <Text style={[styles.secondaryTxt, backhandPose && styles.secondaryTxtReady]}>
              {backhandPose ? 'Recapture backhand-side pose' : 'Capture backhand-side pose'}
            </Text>
          </TouchableOpacity>
          {backhandPose && <VectorPreview title="Backhand-side pose" vector={backhandPose} />}
        </View>
      )}

      {tableReady && bounceReady && (
        <TouchableOpacity style={styles.startBtn} onPress={start} activeOpacity={0.7}>
          <Text style={styles.startBtnTxt}>
            {mode === 'bounce_sides'
              ? (bounceSidesCaptured ? 'Starta Audio plus IMU' : 'Fortsätt utan posekalibrering')
              : 'Continue to next screen'}
          </Text>
        </TouchableOpacity>
      )}
    </ScrollView>
  );
}

const diagram = StyleSheet.create({
  wrap: { alignItems: 'center', marginVertical: 18 },
  sensor: {
    width: 140,
    height: 80,
    backgroundColor: '#1a1a2e',
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#4a9eff',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  sensorLed: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#2ecc71',
    marginBottom: 6,
  },
  sensorText: { color: '#4a9eff', fontWeight: '700', fontSize: 13 },
  sensorSub: { color: '#778', fontSize: 10, marginTop: 2 },
  arrowCol: { alignItems: 'center', marginBottom: 4 },
  arrowLine: { width: 2, height: 20, backgroundColor: '#555' },
  arrowHead: {
    width: 0,
    height: 0,
    borderLeftWidth: 6,
    borderRightWidth: 6,
    borderTopWidth: 10,
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
    borderTopColor: '#555',
    marginBottom: 2,
  },
  arrowLabel: { color: '#778', fontSize: 10 },
  tableRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  tableLine: { flex: 1, height: 2, backgroundColor: '#444' },
  tableLabel: { color: '#aaa', fontSize: 11, fontWeight: '600', letterSpacing: 2 },
});

const preview = StyleSheet.create({
  wrap: {
    backgroundColor: '#111',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  title: {
    color: '#777',
    fontSize: 11,
    letterSpacing: 1.5,
    marginBottom: 8,
  },
  value: {
    color: '#cfd7ff',
    fontSize: 13,
    fontFamily: 'monospace',
  },
});

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0d0d0d' },
  content: { padding: 20, paddingBottom: 40 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  backBtn: { padding: 4 },
  backTxt: { color: '#888', fontSize: 14 },
  playerTxt: { color: '#666', fontSize: 12 },
  title: { color: '#fff', fontSize: 28, fontWeight: '800' },
  subtitle: { color: '#777', fontSize: 14, marginTop: 6, marginBottom: 20, lineHeight: 20 },
  stepBox: {
    backgroundColor: '#111',
    borderRadius: 14,
    padding: 20,
    marginBottom: 16,
  },
  stepNum: { color: '#555', fontSize: 10, letterSpacing: 2, marginBottom: 6 },
  stepTitle: { color: '#fff', fontSize: 18, fontWeight: '700', marginBottom: 10 },
  stepText: { color: '#888', fontSize: 14, lineHeight: 22 },
  actionBtn: {
    backgroundColor: '#0d2d0d',
    borderRadius: 12,
    padding: 18,
    alignItems: 'center',
    marginTop: 16,
  },
  actionBtnBusy: { backgroundColor: '#1a1a1a' },
  actionBtnTxt: { color: '#2ecc71', fontWeight: '700', fontSize: 16 },
  infoTxt: { color: '#f5c76d', fontSize: 12, marginTop: 12, lineHeight: 18 },
  hzTxt: { color: '#aaa', fontSize: 12, textAlign: 'center', marginBottom: 10 },
  progressBg: {
    height: 8,
    backgroundColor: '#1a1a1a',
    borderRadius: 4,
    marginBottom: 10,
    overflow: 'hidden',
  },
  progressFill: { height: '100%', backgroundColor: '#2ecc71' },
  progressTxt: { color: '#aaa', fontSize: 13, textAlign: 'center' },
  linkBtn: { marginTop: 12, alignSelf: 'center' },
  linkTxt: { color: '#777', fontSize: 12 },
  secondaryBtn: {
    backgroundColor: '#15152a',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 14,
  },
  secondaryBtnReady: { backgroundColor: '#0d1f33' },
  secondaryTxt: { color: '#c9d0ff', fontWeight: '700', fontSize: 14 },
  secondaryTxtReady: { color: '#4a9eff' },
  startBtn: {
    backgroundColor: '#0d2d0d',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    marginTop: 8,
  },
  startBtnTxt: { color: '#2ecc71', fontWeight: '800', fontSize: 17, letterSpacing: 1 },
});
