from pathlib import Path

page = Path("docs/index.html")
html = page.read_text(encoding="utf-8")

old_calibration = """  function calibrateDroneScore(rawScore) {
    const span = droneReferenceScore - normalReferenceScore;
    if (!Number.isFinite(span) || span < 0.10) return rawScore;
    return Math.max(0, Math.min(1, (rawScore - normalReferenceScore) / span));
  }"""

new_calibration = """  function calibrateDroneScore(rawScore) {
    const span = droneReferenceScore - normalReferenceScore;
    if (!Number.isFinite(span) || span <= 1e-6) return rawScore;
    return Math.max(0, Math.min(1, (rawScore - normalReferenceScore) / span));
  }"""

if old_calibration not in html:
    raise RuntimeError("Could not find the current score calibration function.")
html = html.replace(old_calibration, new_calibration, 1)

old_aggregation = """    const requiredDroneWindows = activeSegments.length <= 2 ? 1 : Math.max(2, Math.ceil(activeSegments.length * 0.40));
    const strongestScores = rankedSegments.slice(0, Math.min(2, rankedSegments.length)).map(item => item.drone_score);
    const strongestMean = strongestScores.reduce((sum, value) => sum + value, 0) / strongestScores.length;

    let recordingLabel = 'normal';
    if (droneSegments.length >= requiredDroneWindows && strongestMean >= UAV_THRESHOLD) {
      recordingLabel = 'drone';
    } else if (droneSegments.length > 0 || uncertainSegments.length > 0) {
      recordingLabel = 'uncertain';
    }"""

new_aggregation = """    const requiredDroneWindows = activeSegments.length <= 2 ? 1 : Math.max(2, Math.ceil(activeSegments.length * 0.40));
    const requiredUncertainWindows = activeSegments.length <= 2 ? 1 : Math.max(2, Math.ceil(activeSegments.length * 0.40));
    const strongestScores = rankedSegments.slice(0, Math.min(2, rankedSegments.length)).map(item => item.drone_score);
    const strongestMean = strongestScores.reduce((sum, value) => sum + value, 0) / strongestScores.length;
    const activeMedian = activeSegments.length ? median(activeSegments.map(item => item.drone_score)) : 0;

    let recordingLabel = 'normal';
    if (droneSegments.length >= requiredDroneWindows && strongestMean >= UAV_THRESHOLD) {
      recordingLabel = 'drone';
    } else if (uncertainSegments.length >= requiredUncertainWindows && activeMedian > NORMAL_THRESHOLD) {
      recordingLabel = 'uncertain';
    }"""

if old_aggregation not in html:
    raise RuntimeError("Could not find the current recording aggregation block.")
html = html.replace(old_aggregation, new_aggregation, 1)

html = html.replace(
    "const DEMO_VERSION = '20260630-calibrated';",
    "const DEMO_VERSION = '20260630-calibrated-v2';",
    1,
)

page.write_text(html, encoding="utf-8")
print("Corrected normal-reference calibration and uncertainty aggregation in", page)
