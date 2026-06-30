from pathlib import Path
import re

page = Path("docs/index.html")
html = page.read_text(encoding="utf-8")


def replace_once(pattern: str, replacement: str, description: str) -> None:
    global html
    html, count = re.subn(pattern, replacement, html, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Could not patch {description}; replacements={count}")


replace_once(
    r"  const UAV_THRESHOLD = 0\.70;\n  const NORMAL_THRESHOLD = 0\.30;\n  const MIN_REMAINDER_FRACTION = 0\.50;",
    """  const UAV_THRESHOLD = 0.80;
  const NORMAL_THRESHOLD = 0.30;
  const MIN_REMAINDER_FRACTION = 0.50;
  const MIN_RMS = 0.003;
  const MIN_PEAK = 0.015;
  const DEMO_VERSION = '20260630-calibrated';""",
    "demo thresholds",
)

replace_once(
    r"  let tfliteModel = null;\n  let modelPromise = null;",
    """  let tfliteModel = null;
  let modelPromise = null;
  let calibrationPromise = null;
  let droneOutputIndex = null;
  let normalReferenceScore = null;
  let droneReferenceScore = null;""",
    "demo state",
)

html = html.replace(
    "const model = await tflite.loadTFLiteModel('./model/model_int8.tflite', {numThreads: 1});",
    "const model = await tflite.loadTFLiteModel(`./model/model_int8.tflite?v=${DEMO_VERSION}`, {numThreads: 1});",
    1,
)

new_inference_block = r'''  function quantizeClip(clip) {
    let peak = 0;
    for (let i = 0; i < clip.length; i++) peak = Math.max(peak, Math.abs(clip[i]));
    const quantized = new Int32Array(CLIP_SAMPLES);
    for (let i = 0; i < CLIP_SAMPLES; i++) {
      const normalized = peak > 0 ? clip[i] / peak : clip[i];
      const value = Math.round(normalized / INPUT_SCALE + INPUT_ZERO_POINT);
      quantized[i] = Math.max(-128, Math.min(127, value));
    }
    return quantized;
  }

  function clipStats(clip) {
    let peak = 0;
    let sumSquares = 0;
    for (let i = 0; i < clip.length; i++) {
      const value = clip[i];
      peak = Math.max(peak, Math.abs(value));
      sumSquares += value * value;
    }
    return {peak, rms: Math.sqrt(sumSquares / Math.max(1, clip.length))};
  }

  function classifyScore(droneScore, lowEnergy=false) {
    if (lowEnergy) return 'normal';
    if (droneScore >= UAV_THRESHOLD) return 'drone';
    if (droneScore <= NORMAL_THRESHOLD) return 'normal';
    return 'uncertain';
  }

  function predictRawProbabilities(model, clip) {
    const quantized = quantizeClip(clip);
    const inputTensor = tf.tensor(quantized, [1, CLIP_SAMPLES, 1], 'int32');
    let outputTensor;
    try {
      outputTensor = model.predict(inputTensor);
      const raw = outputTensor.dataSync();
      let p0 = Math.max(0, Math.min(1, (raw[0] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));
      let p1 = Math.max(0, Math.min(1, (raw[1] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));
      const total = p0 + p1;
      if (total > 0) {
        p0 /= total;
        p1 /= total;
      }
      return [p0, p1];
    } finally {
      inputTensor.dispose();
      if (outputTensor && typeof outputTensor.dispose === 'function') outputTensor.dispose();
    }
  }

  function oneSecondWindows(audio) {
    const windows = [];
    if (!audio.length) return windows;
    if (audio.length <= CLIP_SAMPLES) {
      const clip = new Float32Array(CLIP_SAMPLES);
      clip.set(audio);
      windows.push(clip);
      return windows;
    }
    for (let start = 0; start + CLIP_SAMPLES <= audio.length; start += CLIP_SAMPLES) {
      windows.push(audio.slice(start, start + CLIP_SAMPLES));
    }
    return windows;
  }

  function median(values) {
    const ordered = [...values].sort((a, b) => a - b);
    const middle = Math.floor(ordered.length / 2);
    return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
  }

  async function fetchReferenceAudio(path, name) {
    const response = await fetch(`${path}?v=${DEMO_VERSION}`, {cache: 'no-store'});
    if (!response.ok) throw new Error(`Could not load ${name} reference audio.`);
    const blob = await response.blob();
    return decodeAudio(new File([blob], name, {type: blob.type || 'audio/wav'}));
  }

  async function calibrateOutputOrder(model) {
    const [droneAudio, normalAudio] = await Promise.all([
      fetchReferenceAudio('assets/demo-drone.wav', 'drone-reference.wav'),
      fetchReferenceAudio('assets/demo-normal.wav', 'background-reference.wav')
    ]);

    const droneWindows = oneSecondWindows(droneAudio);
    const normalWindows = oneSecondWindows(normalAudio);
    const droneScores = [[], []];
    const normalScores = [[], []];

    droneWindows.forEach(clip => {
      const probabilities = predictRawProbabilities(model, clip);
      droneScores[0].push(probabilities[0]);
      droneScores[1].push(probabilities[1]);
    });
    normalWindows.forEach(clip => {
      const probabilities = predictRawProbabilities(model, clip);
      normalScores[0].push(probabilities[0]);
      normalScores[1].push(probabilities[1]);
    });

    const separation0 = median(droneScores[0]) - median(normalScores[0]);
    const separation1 = median(droneScores[1]) - median(normalScores[1]);
    droneOutputIndex = separation0 >= separation1 ? 0 : 1;
    droneReferenceScore = median(droneScores[droneOutputIndex]);
    normalReferenceScore = median(normalScores[droneOutputIndex]);

    if (droneReferenceScore <= normalReferenceScore) {
      throw new Error('The labeled reference clips did not produce a valid UAV/background separation.');
    }

    modelLive.textContent = `mini-SE-Net int8 · calibrated output ${droneOutputIndex}`;
  }

  async function ensureOutputCalibration(model) {
    if (droneOutputIndex !== null) return;
    if (!calibrationPromise) {
      modelLive.textContent = 'Calibrating model with labeled references…';
      calibrationPromise = calibrateOutputOrder(model).catch(error => {
        calibrationPromise = null;
        throw error;
      });
    }
    await calibrationPromise;
  }

  function calibrateDroneScore(rawScore) {
    const span = droneReferenceScore - normalReferenceScore;
    if (!Number.isFinite(span) || span < 0.10) return rawScore;
    return Math.max(0, Math.min(1, (rawScore - normalReferenceScore) / span));
  }

  function predictClip(model, clip) {
    const stats = clipStats(clip);
    const lowEnergy = stats.rms < MIN_RMS || stats.peak < MIN_PEAK;
    if (lowEnergy) {
      return {droneScore: 0, normalScore: 1, lowEnergy, stats};
    }

    const probabilities = predictRawProbabilities(model, clip);
    const rawDroneScore = probabilities[droneOutputIndex];
    const droneScore = calibrateDroneScore(rawDroneScore);
    return {
      droneScore,
      normalScore: 1 - droneScore,
      rawDroneScore,
      lowEnergy,
      stats
    };
  }

  async function analyze'''

replace_once(
    r"  function quantizeClip\(clip\) \{.*?\n  \}\n\n  async function analyze",
    new_inference_block,
    "inference and calibration functions",
)

html = html.replace(
    "    const model = await loadModel();\n    const started = performance.now();",
    "    const model = await loadModel();\n    await ensureOutputCalibration(model);\n    const started = performance.now();",
    1,
)

replace_once(
    r"      const \[droneScore, normalScore\] = predictClip\(model, clip\);\n      segments\.push\(\{\n        index: segments\.length,\n        start_s: start / SAMPLE_RATE,\n        end_s: Math\.min\(start \+ CLIP_SAMPLES, audio\.length\) / SAMPLE_RATE,\n        drone_score: droneScore,\n        normal_score: normalScore,\n        label: classifyScore\(droneScore\)\n      \}\);",
    """      const prediction = predictClip(model, clip);
      segments.push({
        index: segments.length,
        start_s: start / SAMPLE_RATE,
        end_s: Math.min(start + CLIP_SAMPLES, audio.length) / SAMPLE_RATE,
        drone_score: prediction.droneScore,
        normal_score: prediction.normalScore,
        raw_drone_score: prediction.rawDroneScore,
        low_energy: prediction.lowEnergy,
        rms: prediction.stats.rms,
        peak: prediction.stats.peak,
        label: classifyScore(prediction.droneScore, prediction.lowEnergy)
      });""",
    "per-window prediction",
)

new_aggregation = r'''    if (!segments.length) throw new Error('No audio samples were decoded.');
    const activeSegments = segments.filter(item => !item.low_energy);
    const rankedSegments = [...segments].sort((a, b) => b.drone_score - a.drone_score);
    const peakSegment = rankedSegments[0];
    const droneSegments = activeSegments.filter(item => item.label === 'drone');
    const uncertainSegments = activeSegments.filter(item => item.label === 'uncertain');
    const requiredDroneWindows = activeSegments.length <= 2 ? 1 : Math.max(2, Math.ceil(activeSegments.length * 0.40));
    const strongestScores = rankedSegments.slice(0, Math.min(2, rankedSegments.length)).map(item => item.drone_score);
    const strongestMean = strongestScores.reduce((sum, value) => sum + value, 0) / strongestScores.length;

    let recordingLabel = 'normal';
    if (droneSegments.length >= requiredDroneWindows && strongestMean >= UAV_THRESHOLD) {
      recordingLabel = 'drone';
    } else if (droneSegments.length > 0 || uncertainSegments.length > 0) {
      recordingLabel = 'uncertain';
    }

    const inferenceMs = performance.now() - started;
    return {
      label: recordingLabel,
      drone_score: peakSegment.drone_score,
      peak_start_s: peakSegment.start_s,
      peak_end_s: peakSegment.end_s,
      duration_s: audio.length / SAMPLE_RATE,
      segments_analyzed: segments.length,
      drone_segments: droneSegments.length,
      uncertain_segments: uncertainSegments.length,
      low_energy_segments: segments.filter(item => item.low_energy).length,
      inference_ms_total: inferenceMs,
      segments
    };
  }

  function renderResult'''

replace_once(
    r"    if \(!segments\.length\) throw new Error\('No audio samples were decoded\.'\);.*?\n  \}\n\n  function renderResult",
    new_aggregation,
    "recording-level aggregation",
)

html = html.replace(
    "const response = await fetch(button.dataset.sample);",
    "const response = await fetch(`${button.dataset.sample}?v=${DEMO_VERSION}`, {cache: 'no-store'});",
    1,
)

replace_once(
    r'<p class="demo-hint">.*?</p>',
    '<p class="demo-hint">This demo runs the exported mini-SE-Net int8 model and automatically calibrates its output order using the labeled UAV and background reference clips. Quiet microphone windows are rejected before normalization, and multi-second recordings require sustained UAV evidence.</p>',
    "demo explanation",
)

page.write_text(html, encoding="utf-8")
print("Added reference calibration, low-energy rejection, and sustained-window aggregation to", page)
