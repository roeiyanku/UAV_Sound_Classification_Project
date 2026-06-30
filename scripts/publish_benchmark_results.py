from pathlib import Path

p = Path("docs/index.html")
s = p.read_text(encoding="utf-8")

s = s.replace("const DEMO_VERSION = '20260630-calibrated';", "const DEMO_VERSION = '20260630-calibrated-v2';", 1)

old = """  function median(values) {
    const ordered = [...values].sort((a, b) => a - b);
    const middle = Math.floor(ordered.length / 2);
    return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
  }
"""
new = old + """
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
"""
if "function quantile(values, fraction)" not in s:
    if old not in s:
        raise RuntimeError("median helper not found")
    s = s.replace(old, new, 1)

old = """    droneOutputIndex = separation0 >= separation1 ? 0 : 1;
    droneReferenceScore = median(droneScores[droneOutputIndex]);
    normalReferenceScore = median(normalScores[droneOutputIndex]);

    if (droneReferenceScore <= normalReferenceScore) {
      throw new Error('The labeled reference clips did not produce a valid UAV/background separation.');
    }
"""
new = """    droneOutputIndex = separation0 >= separation1 ? 0 : 1;
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
"""
if old in s:
    s = s.replace(old, new, 1)
elif new not in s:
    raise RuntimeError("calibration anchors not found")

old = """    const requiredDroneWindows = activeSegments.length <= 2 ? 1 : Math.max(2, Math.ceil(activeSegments.length * 0.40));
    const strongestScores = rankedSegments.slice(0, Math.min(2, rankedSegments.length)).map(item => item.drone_score);
    const strongestMean = strongestScores.reduce((sum, value) => sum + value, 0) / strongestScores.length;

    let recordingLabel = 'normal';
    if (droneSegments.length >= requiredDroneWindows && strongestMean >= UAV_THRESHOLD) {
      recordingLabel = 'drone';
    } else if (droneSegments.length > 0 || uncertainSegments.length > 0) {
      recordingLabel = 'uncertain';
    }
"""
new = """    const requiredDroneWindows = activeSegments.length <= 2 ? 1 : Math.max(2, Math.ceil(activeSegments.length * 0.40));
    const requiredUncertainWindows = activeSegments.length <= 2 ? 1 : Math.max(2, Math.ceil(activeSegments.length * 0.40));
    const strongestScores = rankedSegments.slice(0, Math.min(2, rankedSegments.length)).map(item => item.drone_score);
    const strongestMean = strongestScores.reduce((sum, value) => sum + value, 0) / strongestScores.length;

    let recordingLabel = 'normal';
    if (droneSegments.length >= requiredDroneWindows && strongestMean >= UAV_THRESHOLD) {
      recordingLabel = 'drone';
    } else if (droneSegments.length > 0 || uncertainSegments.length >= requiredUncertainWindows) {
      recordingLabel = 'uncertain';
    }
"""
if old in s:
    s = s.replace(old, new, 1)
elif new not in s:
    raise RuntimeError("aggregation block not found")

p.write_text(s, encoding="utf-8")
print("Updated normal-reference calibration")
