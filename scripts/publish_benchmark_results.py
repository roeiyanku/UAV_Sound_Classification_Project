from pathlib import Path

page = Path("docs/index.html")
html = page.read_text(encoding="utf-8")

# The training metadata uses class 0 for the rare UAV class and class 1 for
# background. Keep the browser output mapping consistent with that encoding.
wrong_mapping = '''      // Model class order: 0 = background, 1 = drone.
      let normalScore = Math.max(0, Math.min(1, (raw[0] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));
      let droneScore = Math.max(0, Math.min(1, (raw[1] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));'''
correct_mapping = '''      // Training class order: 0 = drone, 1 = background.
      let droneScore = Math.max(0, Math.min(1, (raw[0] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));
      let normalScore = Math.max(0, Math.min(1, (raw[1] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));'''

if wrong_mapping in html:
    html = html.replace(wrong_mapping, correct_mapping, 1)
elif correct_mapping not in html:
    raise RuntimeError("Could not verify the browser model class mapping.")

# Prevent browser voice-processing from removing a UAV sound played through a
# speaker. These settings are especially important for Chrome microphone tests.
html = html.replace(
    "recordingStream = await navigator.mediaDevices.getUserMedia({audio: true});",
    '''recordingStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });''',
    1,
)

html = html.replace(
    "Recording… press stop when the sample is ready.",
    "Recording raw microphone audio… play the sound from another device or speaker, then press stop after at least one second.",
    1,
)

html = html.replace(
    "The model expects one-second windows. Scores from 30% to 70% are shown as uncertain. Very short trailing fragments are ignored.",
    "The built-in buttons use labeled project WAV recordings: one UAV example and one background example. For microphone testing, play audio from a separate device or speaker and record for at least one second. Scores from 30% to 70% are shown as uncertain.",
    1,
)

page.write_text(html, encoding="utf-8")
print("Corrected live-demo class order and microphone capture settings in", page)
