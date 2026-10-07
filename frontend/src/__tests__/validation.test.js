import { describe, it, expect } from 'vitest';
import { validateVideoFile } from '../services/detectionService';

describe('Client-Side Input Validation (validateVideoFile)', () => {
  it('rejects null or undefined input', () => {
    const resultNull = validateVideoFile(null);
    expect(resultNull.valid).toBe(false);
    expect(resultNull.error).toContain('No video file provided');

    const resultUndefined = validateVideoFile(undefined);
    expect(resultUndefined.valid).toBe(false);
  });

  it('rejects files exceeding the 100 MB hard limit', () => {
    // 101 MB file
    const oversizedFile = new File([new Uint8Array(101 * 1024 * 1024)], 'giant_video.mp4', {
      type: 'video/mp4',
    });

    const result = validateVideoFile(oversizedFile);
    expect(result.valid).toBe(false);
    expect(result.error).toContain('exceeds the 100 MB hard limit');
  });

  it('rejects zero-byte files', () => {
    const emptyFile = new File([], 'empty.mp4', { type: 'video/mp4' });
    const result = validateVideoFile(emptyFile);
    expect(result.valid).toBe(false);
    expect(result.error).toContain('0 bytes');
  });

  it('rejects disallowed MIME types (e.g., text, images, audio)', () => {
    const textFile = new File(['hello'], 'notes.txt', { type: 'text/plain' });
    expect(validateVideoFile(textFile).valid).toBe(false);

    const imageFile = new File(['img_data'], 'photo.jpg', { type: 'image/jpeg' });
    expect(validateVideoFile(imageFile).valid).toBe(false);

    const audioFile = new File(['audio_data'], 'song.mp3', { type: 'audio/mpeg' });
    expect(validateVideoFile(audioFile).valid).toBe(false);
  });

  it('accepts valid video MIME types (MP4, QuickTime, WebM, AVI)', () => {
    const validMp4 = new File(['video_data'], 'test.mp4', { type: 'video/mp4' });
    expect(validateVideoFile(validMp4).valid).toBe(true);

    const validMov = new File(['video_data'], 'test.mov', { type: 'video/quicktime' });
    expect(validateVideoFile(validMov).valid).toBe(true);

    const validWebm = new File(['video_data'], 'test.webm', { type: 'video/webm' });
    expect(validateVideoFile(validWebm).valid).toBe(true);

    const validAvi = new File(['video_data'], 'test.avi', { type: 'video/x-msvideo' });
    expect(validateVideoFile(validAvi).valid).toBe(true);
  });

  it('falls back to file extension if browser MIME is empty or generic', () => {
    const fileWithGenericMime = new File(['data'], 'recording.mp4', {
      type: 'application/octet-stream',
    });
    // Even if mime is octet-stream, extension .mp4 matches
    expect(validateVideoFile(fileWithGenericMime).valid).toBe(true);
  });
});
