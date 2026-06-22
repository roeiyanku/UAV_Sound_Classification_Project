import json
import wave
from pathlib import Path

import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f'{path}: expected 16-bit PCM, got {sample_width * 8}-bit')
    audio = np.frombuffer(frames, dtype='<i2').astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f'{path}: expected {SAMPLE_RATE} Hz, got {sample_rate}')
    return audio


def make_windows(audio: np.ndarray) -> list[np.ndarray]:
    starts = list(range(0, max(0, len(audio) - CLIP_SAMPLES + 1), CLIP_SAMPLES))
    if not starts:
        starts = [0]
    remainder = len(audio) % CLIP_SAMPLES
    if len(audio) > CLIP_SAMPLES and remainder >= CLIP_SAMPLES // 2:
        tail_start = len(audio) - CLIP_SAMPLES
        if tail_start > starts[-1]:
            starts.append(tail_start)
    windows = []
    for start in starts:
        clip = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        actual = audio[start:start + CLIP_SAMPLES]
        clip[:len(actual)] = actual
        windows.append(clip)
    return windows


def preprocess(clip: np.ndarray, mode: str, file_peak: float) -> np.ndarray:
    x = clip.astype(np.float32).copy()
    if mode == 'raw':
        return x
    if mode == 'window_peak':
        peak = float(np.max(np.abs(x)))
        return x / peak if peak > 0 else x
    if mode == 'file_peak':
        return x / file_peak if file_peak > 0 else x
    if mode == 'mean_center_window_peak':
        x -= float(np.mean(x))
        peak = float(np.max(np.abs(x)))
        return x / peak if peak > 0 else x
    if mode == 'rms_0_1':
        rms = float(np.sqrt(np.mean(x * x)))
        return np.clip(x * (0.1 / rms), -1.0, 1.0) if rms > 0 else x
    raise ValueError(mode)


def main() -> None:
    model_path = 'docs/model/model_int8.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    in_scale, in_zp = inp['quantization']
    out_scale, out_zp = out['quantization']

    report = {
        'input': {
            'shape': inp['shape'].tolist(),
            'dtype': str(inp['dtype']),
            'quantization': [float(in_scale), int(in_zp)],
        },
        'output': {
            'shape': out['shape'].tolist(),
            'dtype': str(out['dtype']),
            'quantization': [float(out_scale), int(out_zp)],
        },
        'samples': {},
    }

    modes = ['raw', 'window_peak', 'file_peak', 'mean_center_window_peak', 'rms_0_1']
    for name in ['demo-normal', 'demo-drone']:
        audio = load_wav(f'docs/assets/{name}.wav')
        file_peak = float(np.max(np.abs(audio)))
        windows = make_windows(audio)
        sample_report = {
            'duration_s': len(audio) / SAMPLE_RATE,
            'file_peak': file_peak,
            'file_rms': float(np.sqrt(np.mean(audio * audio))),
            'modes': {},
        }
        for mode in modes:
            outputs = []
            for clip in windows:
                x = preprocess(clip, mode, file_peak)
                q = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
                interpreter.set_tensor(inp['index'], q.reshape(1, CLIP_SAMPLES, 1))
                interpreter.invoke()
                raw = interpreter.get_tensor(out['index'])[0]
                probs = (raw.astype(np.float32) - out_zp) * out_scale
                total = float(np.sum(probs))
                if total > 0:
                    probs = probs / total
                outputs.append({
                    'raw': [int(v) for v in raw],
                    'probs': [float(v) for v in probs],
                    'input_min': float(np.min(x)),
                    'input_max': float(np.max(x)),
                    'input_rms': float(np.sqrt(np.mean(x * x))),
                })
            sample_report['modes'][mode] = {
                'windows': outputs,
                'mean_probs': np.mean([o['probs'] for o in outputs], axis=0).tolist(),
                'peak_class0': max(o['probs'][0] for o in outputs),
            }
        report['samples'][name] = sample_report

    Path('diagnostics').mkdir(exist_ok=True)
    Path('diagnostics/tflite_demo_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
