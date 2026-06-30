from __future__ import annotations

import re
from pathlib import Path

PAGE = Path("docs/index.html")
html = PAGE.read_text(encoding="utf-8")

# Add a visible route from the main site to the dedicated datasets page.
nav_before = '<a href="#data">Data</a><a href="#demo">Live demo</a>'
nav_after = '<a href="#data">Data</a><a href="datasets/">Datasets</a><a href="#demo">Live demo</a>'
if '<a href="datasets/">Datasets</a>' not in html:
    if nav_before not in html:
        raise RuntimeError("Could not find the main navigation data link.")
    html = html.replace(nav_before, nav_after, 1)

hero_before = '<a class="btn" href="#data">Watch field recording</a>'
hero_after = '<a class="btn" href="#data">Watch field recording</a><a class="btn" href="datasets/">Explore datasets</a>'
if 'href="datasets/">Explore datasets</a>' not in html:
    if hero_before not in html:
        raise RuntimeError("Could not find the hero field-recording button.")
    html = html.replace(hero_before, hero_after, 1)

# Clarify that the historical per-model latency column measured only the saved
# classifier, then place the new stage-by-stage benchmark directly below it.
old_transition = (
    '<p class="result-note">* drone (anomaly) class &middot; ms/samp = classifier '
    'inference only; feature-extraction time not recorded in any run &middot; '
    'PRELIMINARY random clip-level split.</p><article class="deploy-result card">'
)

benchmark_block = '''<p class="result-note">* drone (anomaly) class &middot; the historical ms/samp column measures classifier inference only &middot; see the dedicated stage benchmark below for feature extraction, projection, score/loss, decision, and directly measured end-to-end latency &middot; PRELIMINARY random clip-level split.</p><div class="section-head" style="margin-top:54px"><div><div class="eyebrow">Measured latency &amp; component cost</div><h2 style="font-size:clamp(24px,3vw,34px)">One-second CPU benchmark, split by stage.</h2></div><p class="lead">A dedicated batch-size-one benchmark now separates feature extraction, optional metric-learning projection, anomaly score, final decision, and a directly measured waveform-to-prediction total.</p></div><div class="metrics"><article class="metric card"><small>Fastest frontend</small><strong>14.15 ms</strong><span>YAMNet feature extraction</span></article><article class="metric card"><small>Fastest complete pipeline</small><strong>15.22 ms</strong><span>YAMNet + One-Class SVM</span></article><article class="metric card"><small>AST frontend</small><strong>4.83 s</strong><span>Feature extraction dominates total latency</span></article></div><div class="table-card card"><table><thead><tr><th>Pipeline</th><th>Feature extraction</th><th>Projection</th><th>Score / loss</th><th>Decision</th><th>Measured total</th></tr></thead><tbody><tr class="best"><td>YAMNet + One-Class SVM</td><td>14.153 ms</td><td>&mdash;</td><td>0.373 ms</td><td>0.010 ms</td><td>15.218 ms</td></tr><tr><td>YAMNet triplet + One-Class SVM</td><td>14.153 ms</td><td>6.254 ms</td><td>0.275 ms</td><td>0.010 ms</td><td>33.318 ms</td></tr><tr><td>Wav2Vec2 + One-Class SVM</td><td>341.521 ms</td><td>&mdash;</td><td>0.369 ms</td><td>0.011 ms</td><td>334.804 ms</td></tr><tr><td>AST triplet + One-Class SVM</td><td>4,834.740 ms</td><td>6.791 ms</td><td>0.406 ms</td><td>0.017 ms</td><td>4,867.334 ms</td></tr><tr><td>AST triplet + Isolation Forest</td><td>4,834.740 ms</td><td>6.791 ms</td><td>22.406 ms</td><td>0.010 ms</td><td>4,927.690 ms</td></tr></tbody></table></div><p class="result-note">CPU notebook environment &middot; decoded mono 16 kHz, one-second waveform &middot; batch size 1 &middot; 10 warm-up runs and 100 timed repetitions; tiny decision operations used 10,000 repetitions &middot; directly measured totals are timed independently, so they need not equal the arithmetic component sum &middot; training-only triplet-loss latency: AST 0.629 ms, YAMNet 0.626 ms &middot; triplet projection timing uses architecture-equivalent projection models because trained projection artifacts were not saved &middot; CNN and teacher-student latency rows were skipped because their required benchmark artifacts were unavailable.</p><article class="deploy-result card">'''

if "Measured latency &amp; component cost" not in html:
    if old_transition not in html:
        raise RuntimeError("Could not find the model-table-to-deployment transition.")
    html = html.replace(old_transition, benchmark_block, 1)

# Correct the compact-model wording: the 97.84% value is from the Keras
# random clip-level test run, not a separate full int8 test evaluation.
html = html.replace(
    '<div><small>Accuracy</small><strong>0.9784</strong></div>',
    '<div><small>Keras accuracy</small><strong>0.9784</strong></div>',
    1,
)
html = html.replace(
    '<li><span>Quantized test accuracy</span><b>97.84%</b></li>',
    '<li><span>Keras clip-level accuracy</span><b>97.84%</b></li>',
    1,
)

new_footprint = '''<section id="footprint"><div class="wrap"><div class="section-head"><div><div class="eyebrow">Model footprint &amp; cost</div><h2>Measured stage cost determines edge feasibility.</h2></div><p class="lead">The dedicated benchmark confirms that the pretrained frontend dominates both latency and memory. The full-int8 mini-SE-Net remains the only complete deployment path measured in kilobytes rather than megabytes.</p></div><div class="table-card card"><table><thead><tr><th>Component</th><th>Role</th><th>Parameters</th><th>Measured / reported footprint</th><th>Mean latency / 1-s window</th></tr></thead><tbody><tr><td>AST</td><td>Pretrained frontend</td><td>~86 M</td><td>330.33 MB</td><td>4,834.74 ms</td></tr><tr><td>Wav2Vec2</td><td>Pretrained frontend</td><td>~95 M</td><td>360.00 MB</td><td>341.52 ms</td></tr><tr><td>YAMNet</td><td>Pretrained frontend</td><td>~3.7 M</td><td>~15 MB*</td><td>14.15 ms</td></tr><tr><td>AST triplet projection</td><td>Embedding projection</td><td>&mdash;</td><td>2.13 MB</td><td>6.79 ms</td></tr><tr><td>YAMNet triplet projection</td><td>Embedding projection</td><td>&mdash;</td><td>2.63 MB</td><td>6.25 ms</td></tr><tr class="best"><td>mini-SE-Net</td><td>Complete deployed model &middot; full-int8</td><td>7,350</td><td>22.6 KB</td><td>1.32 ms</td></tr></tbody></table></div><p class="result-note">AST and Wav2Vec2 footprint values are parameter-and-buffer bytes measured by the benchmark &middot; *the benchmark did not capture YAMNet variable bytes, so ~15 MB remains an architecture-level estimate and is excluded from benchmark pipeline-size totals &middot; projection rows are architecture-equivalent models, not recovered trained artifacts &middot; mini-SE-Net latency was measured with the saved TFLite model in the notebook host environment, not on a physical MCU.</p></div></section>'''

footprint_pattern = re.compile(
    r'<section id="footprint">.*?</section>(?=<section id="team">)',
    flags=re.DOTALL,
)
html, replacements = footprint_pattern.subn(new_footprint, html, count=1)
if replacements != 1:
    raise RuntimeError(f"Expected one footprint section, replaced {replacements}.")

PAGE.write_text(html, encoding="utf-8")
print("Updated", PAGE)
