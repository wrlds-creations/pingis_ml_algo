import type { Vector3 } from './types';

export const SERVICE_UUID = '07C80000-07C8-07C8-07C8-07C807C807C8';
export const ACCEL_UUID = '07C80001-07C8-07C8-07C8-07C807C807C8';
export const ACCEL_UUID_ALT = '07C80203-07C8-07C8-07C8-07C807C807C8';
export const GYRO_UUID = '07C80004-07C8-07C8-07C8-07C807C807C8';
export const MAG_UUID = '07C80010-07C8-07C8-07C8-07C807C807C8';

export type ParsedPacket =
  | { type: 'accel' | 'gyro' | 'mag'; x: number; y: number; z: number; sensor_ts: number }
  | null;

export function parsePacket(uuid: string, base64Data: string): ParsedPacket {
  const binaryStr = atob(base64Data);
  if (binaryStr.length < 9) return null;
  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const x = view.getInt16(0, false);
  const y = view.getInt16(2, false);
  const z = view.getInt16(4, false);
  const sensor_ts = (bytes[6] << 16) | (bytes[7] << 8) | bytes[8];
  const upper = uuid.toUpperCase();
  if (upper === ACCEL_UUID || upper === ACCEL_UUID_ALT) return { type: 'accel', x, y, z, sensor_ts };
  if (upper === GYRO_UUID) return { type: 'gyro', x, y, z, sensor_ts };
  if (upper === MAG_UUID) return { type: 'mag', x: -x / 10, y: -y / 10, z: -z / 10, sensor_ts };
  return null;
}

export function magnitude3(x: number, y: number, z: number): number {
  return Math.sqrt(x * x + y * y + z * z);
}

export function normalize(v: Vector3): Vector3 {
  const mag = magnitude3(v.x, v.y, v.z);
  if (mag < 1e-6) return { x: 0, y: 0, z: 0 };
  return {
    x: v.x / mag,
    y: v.y / mag,
    z: v.z / mag,
  };
}

export function dot(a: Vector3, b: Vector3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function averageVector(vectors: Vector3[]): Vector3 {
  if (vectors.length === 0) return { x: 0, y: 0, z: 0 };
  let x = 0;
  let y = 0;
  let z = 0;
  for (const vector of vectors) {
    x += vector.x;
    y += vector.y;
    z += vector.z;
  }
  return {
    x: x / vectors.length,
    y: y / vectors.length,
    z: z / vectors.length,
  };
}

export function formatClock(tsMs: number): string {
  return new Date(tsMs).toLocaleTimeString('sv-SE', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}
