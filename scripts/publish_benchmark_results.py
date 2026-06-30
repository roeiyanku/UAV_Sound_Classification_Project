from pathlib import Path

page = Path("docs/index.html")
html = page.read_text(encoding="utf-8")

wrong = '''      let droneScore = Math.max(0, Math.min(1, (raw[0] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));
      let normalScore = Math.max(0, Math.min(1, (raw[1] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));'''

correct = '''      // Model class order: 0 = background, 1 = drone.
      let normalScore = Math.max(0, Math.min(1, (raw[0] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));
      let droneScore = Math.max(0, Math.min(1, (raw[1] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE));'''

if wrong in html:
    html = html.replace(wrong, correct, 1)
elif correct not in html:
    raise RuntimeError("Could not verify the browser model class mapping.")

html = html.replace(
    "The server resamples it to mono 16 kHz and runs the exact training-time normalization.",
    "The browser resamples it to mono 16 kHz and runs the exact training-time normalization.",
    1,
)

page.write_text(html, encoding="utf-8")
print("Fixed live-demo class mapping in", page)
