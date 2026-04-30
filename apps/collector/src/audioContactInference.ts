import contactModelJson from './models/audio_contact_model.json';
import { predictWithRfModel } from './rfRuntime';

const CONTACT_MODEL = contactModelJson;

export function predictAudioContact(features: Record<string, number>) {
  return predictWithRfModel(CONTACT_MODEL, features);
}
