(() => {
  const SAMPLE_RATE = 16000;
  const CLIP_SAMPLES = 16000;
  const MAX_SECONDS = 60;
  const UAV_THRESHOLD = 0.80;
  const NORMAL_THRESHOLD = 0.30;
  const MIN_REMAINDER_FRACTION = 0.50;
  const MIN_RMS = 0.003;
  const MIN_PEAK = 0.015;
  const DEMO_VERSION = '20260630-calibrated-v3';
  const INPUT_SCALE = 0.007843137718737125;
  const INPUT_ZERO_POINT = -1;
  const OUTPUT_SCALE = 0.00390625;
  const OUTPUT_ZERO_POINT = -128;

  const fileInput = document.getElementById('audioFile');
  const dropzone = document.getElementById('dropzone');
  const status = document.getElementById('demoStatus');
  const label = document.getElementById('predictionLabel');
  const score = document.getElementById('predictionScore');
  const timeline = document.getElementById('scoreTimeline');
  const audioPreview = document.getElementById('audioPreview');
  const canvas = document.getElementById('waveCanvas');
  const recordBtn = document.getElementById('recordBtn');
  const modelLive = document.getElementById('modelLive');

  let currentObjectUrl = null;
  let recorder = null;
  let recordingStream = null;
  let recordingChunks = [];
  let tfliteModel = null;
  let modelPromise = null;
  let calibrationPromise = null;
  let droneOutputIndex = null;
  let normalReferenceScore = null;
  let droneReferenceScore = null;

  function setStatus(message, kind='') {
    status.textContent = message;
    status.className = `demo-status ${kind}`.trim();
  }

  function drawWaveform(data) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const width = canvas.width, height = canvas.height, mid = height / 2;
    const step = Math.max(1, Math.floor(data.length / width));
    ctx.strokeStyle = '#55d6d9';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      let min = 1, max = -1;
      const begin = x * step;
      const stop = Math.min(begin + step, data.length);
      for (let i = begin; i < stop; i++) {
        min = Math.min(min, data[i]);
        max = Math.max(max, data[i]);
      }
      ctx.moveTo(x, mid + min * mid * .86);
      ctx.lineTo(x, mid + max * mid * .86);
    }
    ctx.stroke();
  }

  function showAudio(file) {
    if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = URL.createObjectURL(file);
    audioPreview.src = currentObjectUrl;
    audioPreview.hidden = false;
  }

  function mixToMono(audioBuffer) {
    const length = audioBuffer.length;
    const mono = new Float32Array(length);
    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
      const samples = audioBuffer.getChannelData(channel);
      for (let i = 0; i < length; i++) mono[i] += samples[i] / audioBuffer.numberOfChannels;
    }
    return mono;
  }

  function resampleLinear(input, sourceRate, targetRate) {
    if (sourceRate === targetRate) return input;
    const outputLength = Math.max(1, Math.round(input.length * targetRate / sourceRate));
    const output = new Float32Array(outputLength);
    const ratio = sourceRate / targetRate;
    for (let i = 0; i < outputLength; i++) {
      const position = i * ratio;
      const left = Math.floor(position);
      const right = Math.min(left + 1, input.length - 1);
      const fraction = position - left;
      output[i] = input[left] * (1 - fraction) + input[right] * fraction;
    }
    return output;
  }

  async function decodeAudio(file) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) throw new Error('Web Audio is not supported in this browser.');
    const audioCtx = new AudioCtx();
    try {
      const buffer = await audioCtx.decodeAudioData((await file.arrayBuffer()).slice(0));
      const mono = mixToMono(buffer);
      const resampled = resampleLinear(mono, buffer.sampleRate, SAMPLE_RATE);
      return resampled.slice(0, MAX_SECONDS * SAMPLE_RATE);
    } catch (error) {
      throw new Error('Could not decode this audio format. Try WAV, MP3, M4A, or a Chrome microphone recording.');
    } finally {
      await audioCtx.close();
    }
  }

  async function loadModel() {
    if (tfliteModel) return tfliteModel;
    if (modelPromise) return modelPromise;
    modelPromise = (async () => {
      modelLive.textContent = 'Loading browser model…';
      await tf.setBackend('cpu');
      await tf.ready();
      tflite.setWasmPath('./vendor/tflite/wasm/');
      const model = await tflite.loadTFLiteModel(`./model/model_int8.tflite?v=${DEMO_VERSION}`, {numThreads: 1});
      tfliteModel = model;
      modelLive.textContent = 'mini-SE-Net int8 · browser model ready';
      return model;
    })().catch(error => {
      modelPromise = null;
      modelLive.textContent = 'Browser model failed to load';
      modelLive.style.color = '#efb45c';
      throw error;
    });
    return modelPromise;
  }

  function quantizeClip(clip) {
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

  function quantile(values, fraction) {
    const ordered = [...values].sort((a, b) => a - b);
    if (!ordered.length) return 0;
    const position = (ordered.length - 1) * fraction;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    if (lower === upper) return ordered[lower];
    const weight = position - lower;
    return ordered[lower] * (1 - weight) + ordered[upper] * weight;
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
    const chosenDrone = droneScores[droneOutputIndex];
    const chosenNormal = normalScores[droneOutputIndex];
    normalReferenceScore = Math.max.apply(null, chosenNormal);
    droneReferenceScore = quantile(chosenDrone, 0.25);
    if (droneReferenceScore <= normalReferenceScore + 0.02) {
      normalReferenceScore = quantile(chosenNormal, 0.90);
      droneReferenceScore = quantile(chosenDrone, 0.10);
    }
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
    if (!Number.isFinite(span) || span <= 1e-6) return rawScore;
    return Math.max(0, Math.min(1, (rawScore - normalReferenceScore) / span));
  }

  function predictClip(model, clip) {
    const stats = clipStats(clip);
    const lowEnergy = stats.rms < MIN_RMS || stats.peak < MIN_PEAK;
    if (lowEnergy) return {droneScore: 0, normalScore: 1, lowEnergy, stats};
    const probabilities = predictRawProbabilities(model, clip);
    const rawDroneScore = probabilities[droneOutputIndex];
    const droneScore = calibrateDroneScore(rawDroneScore);
    return {droneScore, normalScore: 1 - droneScore, rawDroneScore, lowEnergy, stats};
  }

  async function analyze(audio) {
    const model = await loadModel();
    await ensureOutputCalibration(model);
    const started = performance.now();
    const segments = [];
    const starts = [];
    if (audio.length <= CLIP_SAMPLES) starts.push(0);
    else {
      for (let start = 0; start + CLIP_SAMPLES <= audio.length; start += CLIP_SAMPLES) starts.push(start);
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
      const prediction = predictClip(model, clip);
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
      });
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    if (!segments.length) throw new Error('No audio samples were decoded.');
    const activeSegments = segments.filter(item => !item.low_energy);
    const rankedSegments = [...segments].sort((a, b) => b.drone_score - a.drone_score);
    const peakSegment = rankedSegments[0];
    const droneSegments = activeSegments.filter(item => item.label === 'drone');
    const uncertainSegments = activeSegments.filter(item => item.label === 'uncertain');
    const requiredDroneWindows = activeSegments.length <= 2 ? 1 : Math.max(2, Math.ceil(activeSegments.length * 0.40));
    const requiredUncertainWindows = activeSegments.length <= 2 ? 1 : Math.max(2, Math.ceil(activeSegments.length * 0.40));
    const strongestScores = rankedSegments.slice(0, Math.min(2, rankedSegments.length)).map(item => item.drone_score);
    const strongestMean = strongestScores.reduce((sum, value) => sum + value, 0) / strongestScores.length;
    let recordingLabel = 'normal';
    if (droneSegments.length >= requiredDroneWindows && strongestMean >= UAV_THRESHOLD) recordingLabel = 'drone';
    else if (droneSegments.length > 0 || uncertainSegments.length >= requiredUncertainWindows) recordingLabel = 'uncertain';
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

  function renderResult(data) {
    const labelText = {drone: 'UAV detected', normal: 'No UAV detected', uncertain: 'Uncertain result'};
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
      bar.title = `${seg.start_s.toFixed(1)}–${seg.end_s.toFixed(1)}s · UAV score ${(seg.drone_score * 100).toFixed(1)}% · ${seg.label}`;
      timeline.appendChild(bar);
    });
  }

  async function runFile(file) {
    if (!file) return;
    showAudio(file);
    setStatus(`Decoding ${file.name} and running local int8 inference…`, 'busy');
    label.textContent = 'Analyzing…';
    label.className = '';
    score.textContent = '—';
    try {
      const audio = await decodeAudio(file);
      drawWaveform(audio);
      const result = await analyze(audio);
      renderResult(result);
      setStatus(`Finished locally: ${result.segments_analyzed} one-second window${result.segments_analyzed === 1 ? '' : 's'} analyzed. Audio was not uploaded.`);
    } catch (error) {
      console.error(error);
      setStatus(error.message || 'Could not run browser inference.', 'error');
      label.textContent = 'Inference unavailable';
      score.textContent = '—';
    }
  }

  fileInput.addEventListener('change', () => runFile(fileInput.files[0]));
  ['dragenter', 'dragover'].forEach(type => dropzone.addEventListener(type, event => {
    event.preventDefault();
    dropzone.classList.add('drag');
  }));
  ['dragleave', 'drop'].forEach(type => dropzone.addEventListener(type, event => {
    event.preventDefault();
    dropzone.classList.remove('drag');
  }));
  dropzone.addEventListener('drop', event => runFile(event.dataTransfer.files[0]));

  document.querySelectorAll('[data-sample]').forEach(button => button.addEventListener('click', async () => {
    setStatus('Loading the example…', 'busy');
    try {
      const response = await fetch(`${button.dataset.sample}?v=${DEMO_VERSION}`, {cache: 'no-store'});
      if (!response.ok) throw new Error('Example file not found.');
      const blob = await response.blob();
      await runFile(new File([blob], button.dataset.name, {type: blob.type || 'audio/wav'}));
    } catch (error) {
      setStatus(error.message || 'Could not load the example.', 'error');
    }
  }));

  recordBtn.addEventListener('click', async () => {
    if (recorder && recorder.state === 'recording') {
      recorder.stop();
      return;
    }
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      setStatus('Microphone recording is not supported in this browser.', 'error');
      return;
    }
    try {
      recordingStream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
      recordingChunks = [];
      const preferred = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : '';
      recorder = preferred ? new MediaRecorder(recordingStream, {mimeType: preferred}) : new MediaRecorder(recordingStream);
      recorder.ondataavailable = event => { if (event.data.size) recordingChunks.push(event.data); };
      recorder.onstop = async () => {
        recordingStream.getTracks().forEach(track => track.stop());
        recordBtn.textContent = '● Record from microphone';
        recordBtn.classList.remove('recording');
        const blob = new Blob(recordingChunks, {type: recorder.mimeType || 'audio/webm'});
        await runFile(new File([blob], 'microphone-recording.webm', {type: blob.type}));
      };
      recorder.start();
      recordBtn.textContent = '■ Stop and analyze';
      recordBtn.classList.add('recording');
      setStatus('Recording raw microphone audio… press stop after at least one second.', 'busy');
    } catch (error) {
      setStatus('Microphone permission was denied or unavailable.', 'error');
    }
  });

  loadModel().catch(error => {
    console.error('Model initialization failed:', error);
    setStatus('The browser model could not be loaded. Check the browser console.', 'error');
  });
})();