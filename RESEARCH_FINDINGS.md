# RESEARCH_FINDINGS.md

Findings from a deep-research pass (2026-09-12) commissioned to replace
assumption/assertion with cited fact in four places: the friction
governor's mu values, the negative-obstacle detection claim's actual
novelty, the explainability pitch's legal/policy grounding, and how
DRISHTI compares to real fielded UGV programs. Full research output
(with every citation) is preserved in this session's transcript; this
file distills it into what changed and why. Where the research itself
flagged a gap (no source found), that gap is stated here too — this
project's own discipline is to say "we don't know" rather than paper
over it.

---

## 1. Friction coefficients — `planning/friction.py` updated

My original `CLASS_TO_MU` values (0.80 / 0.45 / 0.35 dry/vegetation/mud)
were engineering guesses, flagged as such at the time. Real wheeled-UGV
terramechanics/braking-friction literature gives materially different,
and generally LOWER, numbers:

| Class | Old (guessed) | New (cited) | Source |
|---|---|---|---|
| DRIVABLE (dry compacted dirt/gravel) | 0.80 | **0.40** | Wong 2001; Samuelraj et al. 2018 (range 0.18–0.40) |
| VEGETATION (wet grass, proxy) | 0.45 | **0.15** | Salimi et al. 2015 (range 0.10–0.20, extrapolated from documented 55–81% ice/snow friction reduction — no direct wet-grass wheeled-UGV study was found) |
| CAUTION (mud/saturated soil/standing water) | 0.35 | **0.07** | Samuelraj et al. 2018 (0.05 locked-wheel, 0.076 ABS-regulated, measured directly) |

**A genuinely useful cross-check fell out of this**: `vehicle_ugv.yaml`'s
own declared `braking_a_ms2 = 4.0` implies `mu = a/g ≈ 0.408` under the
simplified friction-limited-braking model — landing almost exactly at
the top of the cited dry-dirt/gravel range (0.18–0.40). `MU_DRY_REFERENCE`
is now set to 0.40 for this reason: it is both the top of a real cited
range AND consistent with a platform parameter this project already
declared independently, before this research existed.

The mud number is the biggest correction: 0.35 → 0.07 is a 5x drop, not
a rounding difference. `planning/friction.py`, `tests/test_friction.py`
(no hardcoded numbers, so unaffected), and `frontend/src/lib/frictionMath.ts`
are all updated and re-tested (23/23 passing).

**Still not found**: a wheeled-UGV-specific study isolating "standing
water over a hard off-road substrate" as its own condition — treated as
≤ the mud value per the module's own conservative-fallback discipline,
not a separately cited number.

---

## 2. Negative-obstacle detection — this is NOT a novel technique, and that's fine

**This is the important finding to internalize before any pitch.**
DRISHTI's core idea — inferring a ditch/trench from the ABSENCE of
expected LiDAR returns (a "range shadow") rather than requiring a
positive return — is not new. It was pioneered by Arturo Rankin and
Larry Matthies at NASA JPL for DARPA's Demo III UGV program
(Matthies & Rankin 2003; Rankin et al. 2006), fielded in TerraMax at the
2005 DARPA Grand Challenge, and is still the active technique in current
literature (Shang et al. 2024, MDPI Sensors — tilted-LiDAR negative
obstacle detection in orchards, explicitly modeling "spacing jumps"
between points as the detection signal).

**Do not present range-shadow reasoning to judges as DRISHTI's own
invention.** That claim is checkable and a technically literate judge
(this is a DRDO problem statement — assume they know the field) may
already know Demo III/TerraMax. Claiming novelty where 20-year-old prior
art exists is a credibility risk this project has otherwise been careful
to avoid (see the baseline-comparison honesty framing, the CPU/GPU
discrepancy write-up, etc.).

**The corrected, still-strong framing**: DRISHTI's actual contribution
is not the physical principle (range shadows are physics, not an
algorithm choice) but the SYSTEM around it — a field-validated,
JPL/DARPA-proven detection cue combined with:
- a variable-resolution clipmap tied to sensor Nyquist physics (not
  present in the 2003-2006 or 2005 work, which used fixed-resolution
  scans),
- a modern deep semantic segmentation network fused with the geometric
  cue (Demo III/TerraMax were pre-deep-learning; they used purely
  geometric/thermal signal processing),
- an open, tested, reproducible pipeline (the JPL/DARPA work is
  decades-old, government-funded, and not published as reusable code),
- and the Sparsity Trap / conservatism argument (Bible's own framing)
  that treats an AMBIGUOUS range shadow as a hazard rather than
  requiring confirmation — a design decision Demo III/TerraMax's
  contemporaneous papers do not appear to make as explicitly.

**"Built on NASA JPL/DARPA-validated physics, not an unproven idea" is a
STRONGER defense-context pitch than "we invented this."** Frame it that
way explicitly in any slide or demo script.

**Related recent benchmarks worth knowing about** (not yet adopted,
noted for completeness): ORFD (2022, off-road freespace detection,
12,000+ LiDAR/RGB pairs), RELLIS-OCC (2024, extends RELLIS-3D — the
SAME base dataset this project already trains on — with 3D occupancy/
traversability labels), Verti-Bench (2025, multi-physics off-road
mobility simulator with explicit ditches/banks). RELLIS-OCC in
particular is worth a look in a future session since it shares RELLIS-3D
lineage with this project's own training data.

**Gap found**: no modern public dataset isolates negative obstacles as
their own standalone semantic class (separate from general
non-traversable/freespace/occupancy labels) — RELLIS-3D's own approach
(this project's own `perception/taxonomy.py`, which derives
NEGATIVE_OBSTACLE from geometry rather than any dataset label) is
consistent with that gap, not an oversight.

---

## 3. XAI / explainability — now has real citations behind it

The "XAI is mandatory for defense certification" line from the earlier
wow-factor brainstorm was an assertion. It's actually true and citable:

- **NATO AI Strategy** (PO(2021)0350, 2021) — "Explainability and
  Traceability" is one of NATO's own named Principles of Responsible Use
  for allied AI systems.
- **DoD Instruction 3000.09** (updated 2023) — requires human-machine
  interfaces that let operators understand an autonomous system's
  intent before fielding approval.
- **DoD Machine Learning System Safety Engineering Guidebook** (Jan
  2026) — explicitly extends MIL-STD-882E to ML systems and names "lack
  of explainability" as a core hazard to be mitigated during
  certification.

**Gap found, stated honestly**: no public, unclassified DRDO/DGQA/MoD
document was found mandating XAI or algorithmic traceability for Indian
autonomous systems specifically. The correct claim is therefore: "this
architecture is built to satisfy the same explainability principles
NATO and US DoD already mandate for defense-certified autonomy — not
'DRDO requires this' (which is not a claim this project can currently
back with a citation)." Use the NATO/DoD framing; do not imply an
Indian-specific mandate exists.

This gives `perception/segnet.py`'s `return_attention=True` real,
citable weight it didn't have before: it is not just a demo gimmick, it
directly answers a named DoD-identified certification hazard ("lack of
explainability").

---

## 4. Comparable UGV programs — strong, DRDO-specific competitive positioning

Two findings matter here, one validating and one competitive:

**DARPA RACER program (Rivière et al. 2023) directly validates
DRISHTI's own core thesis**, independently. RACER pushes off-road UGVs
to 7–10 m/s and found that dense semantic classification at that speed
introduces latency that causes the planner to act on "misrepresented or
delayed semantic obstacles and terrain geometries" — their own words for
exactly the failure mode this project's perception-limited speed
envelope (Claim 4) and variable-resolution mapping (Claim 2) exist to
prevent. This is close to a direct, independent, DARPA-funded validation
of this project's central architectural bet. Cite it explicitly.

**DRDO's own currently-known UGVs are a real, favorable competitive
baseline**:
- **Daksh** — EOD remote-operated vehicle, multi-camera teleoperation +
  X-ray payload. No documented autonomous LiDAR mapping stack at all.
- **Muntra** (BMP-2-based, S/N/M variants) — GPS/INS waypoint autonomy
  with radar/EO obstacle detection, tested in the flat terrain of the
  Mahajan field firing range. No public documentation of adaptive-
  resolution LiDAR or negative-obstacle/range-shadow reasoning.

**This is genuinely useful, specific-to-this-problem-statement
positioning**: "DRISHTI's adaptive-resolution 2.5D LiDAR mapping and
range-shadow negative-obstacle detection are capabilities not documented
in DRDO's own currently fielded UGV programs" is a real, checkable,
favorable comparison — stronger for THIS audience than the academic
SalsaNext/CENet/FIDNet comparison already in `BASELINE_COMPARISON.md`
(different judges care about different baselines; a DRDO problem
statement's own judges will likely know Daksh/Muntra by name).

**Gap found**: no public technical detail exists on Muntra/Daksh/THeMIS's
actual internal algorithms (proprietary/classified) — the comparison
above is necessarily at the capability level (what's documented as
existing) not the algorithmic level (how RELLIS-3D-trained FusionSegNet
compares numerically to a system whose numbers were never published).
Don't overstate this into a metrics comparison; it isn't one.

---

## What to change next (not yet done)

1. Add a short, cited paragraph to `BASELINE_COMPARISON.md` reframing
   negative-obstacle detection as JPL/DARPA-validated (not novel) and
   adding the DARPA RACER + Daksh/Muntra positioning above.
2. Anywhere the pitch narrative currently implies "we invented range-shadow
   reasoning" needs a find-and-fix pass (slides, demo script — not yet
   written as of this session).
3. Consider a stretch-goal look at RELLIS-OCC (2024) given it shares
   RELLIS-3D lineage with this project's own training data.
