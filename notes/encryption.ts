import crypto from 'crypto';
import path from 'path';
import fs from 'fs';
import os from 'os';

const DATA_DIR = process.env.EPITO_DATA_DIR || path.resolve(process.cwd(), 'data');
const KEY_PATH = path.join(DATA_DIR, 'encryption.key');

let _cachedKey: Buffer | null = null;

function getEncryptionKey(): Buffer {
  if (_cachedKey) return _cachedKey;

  if (fs.existsSync(KEY_PATH)) {
    const hex = fs.readFileSync(KEY_PATH, 'utf8').trim();
    if (!/^[0-9a-f]{64}$/i.test(hex)) {
      throw new Error('Corrupted encryption key file');
    }
    _cachedKey = Buffer.from(hex, 'hex');
    return _cachedKey;
  }
  const key = crypto.randomBytes(32);
  fs.mkdirSync(DATA_DIR, { recursive: true });
  fs.writeFileSync(KEY_PATH, key.toString('hex'), { mode: 0o600 });
  if (os.platform() !== 'win32') {
    fs.chmodSync(KEY_PATH, 0o600);
  }
  _cachedKey = key;
  return key;
}

export function encrypt(text: string): string {
  const key = getEncryptionKey();
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
  let encrypted = cipher.update(text, 'utf8', 'hex');
  encrypted += cipher.final('hex');
  const tag = cipher.getAuthTag().toString('hex');
  return `${iv.toString('hex')}:${tag}:${encrypted}`;
}

export function decrypt(encrypted: string): string {
  const key = getEncryptionKey();
  const [ivHex, tagHex, data] = encrypted.split(':');
  const iv = Buffer.from(ivHex, 'hex');
  const tag = Buffer.from(tagHex, 'hex');
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv);
  decipher.setAuthTag(tag);
  let decrypted = decipher.update(data, 'hex', 'utf8');
  decrypted += decipher.final('utf8');
  return decrypted;
}
