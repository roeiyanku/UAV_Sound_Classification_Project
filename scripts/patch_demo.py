from pathlib import Path

path = Path("docs/index.html")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, name: str) -> None:
    global text
    if old not in text:
        raise RuntimeError(f"Missing expected block: {name}")
    text = text.replace(old, new, 1)


replace_once(
    ".prediction-label strong.normal{color:var(--green)}",
    ".prediction-label strong.normal{color:var(--green)}.prediction-label strong.uncertain{color:var(--amber)}",
    "uncertain label CSS",
)
replace_once(
    ".seg.drone{background:var(--red)}",
    ".seg.drone{background:var(--red)}.seg.uncertain{background:var(--amber)}",
    "uncertain segment CSS",
)
replace_once(
    "green = normal · red = drone",
    "green = normal · amber = uncertain · red = drone",
    "timeline legend",
)
replace_once(
    "  const MAX_SECONDS = 60;\n  const THRESHOLD = 0.5;",
    "  const MAX_SECONDS = 60;\n  const UAV_THRESHOLD = 0.70;\n  const NORMAL_THRESHOLD = 0.30;\n  const MIN_REMAINDER_FRACTION = 0.50;",
    "threshold constants",
)
replace_once(
    "The model expects one-second windows. For multi-second audio, the overall decision uses the highest drone score while the chart preserves every window.",
    "The model expects one-second windows. Scores from 30% to 70% are shown as uncertain. Very short trailing fragments are ignored.",
    "demo guidance",
)
replace_once(
    "Research prototype. The displayed score is the model’s quantized softmax output, not an operational threat assessment. Current published metrics use the saved random clip-level test split.",
    "Research prototype. The displayed score is the model’s quantized softmax output, not an operational threat assessment. Scores between 30% and 70% are intentionally reported as uncertain. Current published metrics use the saved random clip-level test split.",
    "demo note",
)

old_predict = """  function predictClip(model, clip) {
    const quantized = quantizeClip(clip);
    const inputTensor = tf.tensor(quantized, [1, CLIP_SAMPLES, 1], 'int32');
    let outputTensor;
    try {
      outputTensor = model.predict(inputTensor);
      const raw = outputTensor.dataSync();
      const droneScore = Math.max(0, Math.min(1, (raw[0] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));
      const normalScore = Math.max(0, Math.min(1, (raw[1] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));
      return [droneScore, normalScore];
    } finally {
      inputTensor.dispose();
      if (outputTensor && typeof outputTensor.dispose === 'function') outputTensor.dispose();
    }
  }
"""
new_predict = """  function classifyScore(droneScore) {
    if (droneScore >= UAV_THRESHOLD) return 'drone';
    if (droneScore <= NORMAL_THRESHOLD) return 'normal';
    return 'uncertain';
  }

  function predictClip(model, clip) {
    const quantized = quantizeClip(clip);
    const inputTensor = tf.tensor(quantized, [1, CLIP_SAMPLES, 1], 'int32');
    let outputTensor;
    try {
      outputTensor = model.predict(inputTensor);
      const raw = outputTensor.dataSync();
      let droneScore = Math.max(0, Math.min(1, (raw[0] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));
      let normalScore = Math.max(0, Math.min(1, (raw[1] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));
      const total = droneScore + normalScore;
      if (total > 0) {
        droneScore /= total;
        normalScore /= total;
      }
      return [droneScore, normalScore];
    } finally {
      inputTensor.dispose();
      if (outputTensor && typeof outputTensor.dispose === 'function') outputTensor.dispose();
    }
  }
"""
replace_once(old_predict, new_predict, "prediction function")

analyze_start = text.index("  async function analyze(audio) {")
analyze_end = text.index("\n  function renderResult(data) {", analyze_start)
new_analyze = """  async function analyze(audio) {
    const model = await loadModel();
    const started = performance.now();
    const segments = [];
    const starts = [];

    if (audio.length <= CLIP_SAMPLES) {
      starts.push(0);
    } else {
      for (let start = 0; start + CLIP_SAMPLES <= audio.length; start += CLIP_SAMPLES) {
        starts.push(start);
      }
      const remainder = audio.length % CLIP_SAMPLES;
      if (remainder >= CLIP_SAMPLES * MIN_REMAINDER_FRACTION) {
        const tailStart = audio.length - CLIP_SAMPLES;
        if (!starts.length || tailStart > starts[starts.length - 1]) starts.push(tailStart);
      }
    }

    for (const start of starts) {
      const actualSamples = Math.min(CLIP_SAMPLES, audio.length - start);
      const clip = new Float32Array(CLIP_SAMPLES);
      clip.set(audio.subarray(start, start + actualSamples));
      const [droneScore, normalScore] = predictClip(model, clip);
      segments.push({
        index: segments.length,
        start_s: start / SAMPLE_RATE,
        end_s: Math.min(start + CLIP_SAMPLES, audio.length) / SAMPLE_RATE,
        drone_score: droneScore,
        normal_score: normalScore,
        label: classifyScore(droneScore)
      });
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    if (!segments.length) throw new Error('No audio samples were decoded.');
    const peakSegment = segments.reduce((best, item) => item.drone_score > best.drone_score ? item : best);
    const inferenceMs = performance.now() - started;
    return {
      label: classifyScore(peakSegment.drone_score),
      drone_score: peakSegment.drone_score,
      peak_start_s: peakSegment.start_s,
      peak_end_s: peakSegment.end_s,
      duration_s: audio.length / SAMPLE_RATE,
      segments_analyzed: segments.length,
      drone_segments: segments.filter(item => item.label === 'drone').length,
      uncertain_segments: segments.filter(item => item.label === 'uncertain').length,
      inference_ms_total: inferenceMs,
      segments
    };
  }
"""
text = text[:analyze_start] + new_analyze + text[analyze_end:]

render_start = text.index("  function renderResult(data) {")
render_end = text.index("\n  async function runFile(file) {", render_start)
new_render = """  function renderResult(data) {
    const labelText = {
      drone: 'UAV detected',
      normal: 'No UAV detected',
      uncertain: 'Uncertain result'
    };
    label.textContent = labelText[data.label];
    label.className = data.label;
    score.textContent = `${(data.drone_score * 100).toFixed(1)}%`;
    document.getElementById('peakWindow').textContent = `${data.peak_start_s.toFixed(1)}–${data.peak_end_s.toFixed(1)} s`;
    document.getElementById('windowCount').textContent = `${data.drone_segments} UAV · ${data.uncertain_segments} uncertain · ${data.segments_analyzed} total`;
    document.getElementById('audioDuration').textContent = `${data.duration_s.toFixed(2)} s`;
    document.getElementById('inferenceTime').textContent = `${data.inference_ms_total.toFixed(1)} ms`;
    timeline.innerHTML = '';
    data.segments.forEach(seg => {
      const bar = document.createElement('div');
      bar.className = `seg ${seg.label}`;
      bar.style.height = `${Math.max(8, seg.drone_score * 100)}%`;
      bar.dataset.tip = `${seg.start_s.toFixed(1)}–${seg.end_s.toFixed(1)}s · UAV score ${(seg.drone_score * 100).toFixed(1)}% · ${seg.label}`;
      timeline.appendChild(bar);
    });
  }
"""
text = text[:render_start] + new_render + text[render_end:]

path.write_text(text, encoding="utf-8")
