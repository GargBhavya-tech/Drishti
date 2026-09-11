/**
 * mockData.ts -- a synthetic demo sequence shaped exactly like the
 * real backend export contract would be (per-cell class/height/
 * observability/sparsity, plus per-frame HUD + speed-envelope state),
 * so swapping in a real Python export later is a data-source change,
 * not a component rewrite. No backend touched for this build, per
 * the user's own instruction.
 */

import type { DrishtiClassId } from "./theme"

export type Observability = "UNOBSERVED" | "FREE" | "OCCUPIED" | "OCCLUDED"
export type SparsityVerdict = "NORMAL" | "SPARSE_STRUCTURED" | "NOISE_SUPPRESSED" | "FREE" | "UNKNOWN"

export interface CellSample {
  i: number
  j: number
  classId: DrishtiClassId
  heightM: number
  observability: Observability
  sparsityVerdict: SparsityVerdict
  isMoving: boolean
}

export interface HudStats {
  cellsByLevel: number[]
  mbUsed: number
  latencyP50Ms: number
  latencyP95Ms: number
  stampMismatches: number
}

export interface SpeedEnvelopeState {
  vMaxKmh: number
  bindingHazard: string
  bindingRangeM: number
}

export interface DemoFrame {
  frame: number
  cells: CellSample[]
  hud: HudStats
  speedEnvelope: SpeedEnvelopeState
}

// Deterministic PRNG (mulberry32) -- same sequence every load, so the
// demo looks identical across reloads/refreshes during a live pitch.
function mulberry32(seed: number) {
  let a = seed
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const GRID_HALF = 34
const rand = mulberry32(20260911)

function heightField(i: number, j: number): number {
  return 0.02 * i + 0.6 * Math.sin(i / 9) * Math.cos(j / 11) + 0.15 * Math.sin(j / 4)
}

function baseClassFor(i: number, j: number): DrishtiClassId {
  // A verge of CAUTION along the edges, a vegetation patch, a static
  // obstacle cluster, a negative-obstacle trench line, an overhang
  // strip -- everything else DRIVABLE.
  if (Math.abs(j) > GRID_HALF - 4) return 2 // CAUTION verge
  if (j > 6 && j < 16 && i > -10 && i < 14) return 5 // VEGETATION patch
  if (i > 14 && i < 20 && j > -6 && j < 0) return 4 // STATIC_OBSTACLE cluster
  if (i > -4 && i < 2 && j < -8 && j > -22) return 8 // NEGATIVE_OBSTACLE trench
  if (i < -18 && j > -6 && j < 6) return 9 // OVERHANG strip
  return 1 // DRIVABLE
}

function sparsityFor(rangeCells: number): SparsityVerdict {
  if (rangeCells > 30) return rand() > 0.4 ? "UNKNOWN" : "SPARSE_STRUCTURED"
  if (rangeCells > 22) return rand() > 0.75 ? "SPARSE_STRUCTURED" : "NORMAL"
  return "NORMAL"
}

/** A small pedestrian-like cluster walking a straight path across the
 * whole sequence, used to drive the motion overlay and the speed
 * envelope's own occlusion dip. */
function pedestrianPositionAt(frame: number, nFrames: number): { i: number; j: number } {
  const t = frame / (nFrames - 1)
  return { i: Math.round(-24 + t * 40), j: Math.round(20 - t * 6) }
}

export function generateSequence(nFrames = 48): DemoFrame[] {
  const frames: DemoFrame[] = []

  for (let frame = 0; frame < nFrames; frame++) {
    const ped = pedestrianPositionAt(frame, nFrames)
    const cells: CellSample[] = []

    for (let i = -GRID_HALF; i <= GRID_HALF; i += 1) {
      for (let j = -GRID_HALF; j <= GRID_HALF; j += 1) {
        // Decimate coarser rings for display density, matching Ticket
        // #51's own "Watch out: log only occupied cells, decimate
        // coarse levels for display" -- skip most far-out cells.
        const rangeCells = Math.hypot(i, j)
        if (rangeCells > 26 && (i + j) % 2 !== 0) continue
        if (rangeCells > 32 && (i + j) % 3 !== 0) continue

        const isPedestrianCell = Math.abs(i - ped.i) <= 1 && Math.abs(j - ped.j) <= 1
        const classId = isPedestrianCell ? 7 : baseClassFor(i, j)

        let observability: Observability = "OCCUPIED"
        if (rangeCells > 30 && rand() > 0.5) observability = "UNOBSERVED"
        else if (i > 14 && i < 20 && j > -12 && j < -6) observability = "OCCLUDED" // shadow behind the static cluster
        else if (rand() > 0.85) observability = "FREE"

        cells.push({
          i,
          j,
          classId,
          heightM: heightField(i, j) + (classId === 8 ? -0.6 : 0) + (classId === 9 ? 2.0 : 0),
          observability,
          sparsityVerdict: sparsityFor(rangeCells),
          isMoving: isPedestrianCell,
        })
      }
    }

    // Occlusion event mid-sequence -- the speed gauge should visibly
    // fall here (Ticket #54's own test: "drive into an occluded
    // region; the limit visibly falls and the binding hazard changes").
    const inOcclusionEvent = frame > nFrames * 0.55 && frame < nFrames * 0.75
    const bindingHazard = inOcclusionEvent ? "occluded_corridor" : "2m_ditch"
    const bindingRangeM = inOcclusionEvent ? 6.7 + rand() * 1.5 : 21.6 + rand() * 0.6
    const vMaxKmh = inOcclusionEvent ? 22.5 + rand() * 3 : 43.2 + rand() * 0.8 - 0.4

    frames.push({
      frame,
      cells,
      hud: {
        cellsByLevel: [262144, 262144, 262144, 262144],
        mbUsed: 12.58 + (rand() - 0.5) * 0.02,
        latencyP50Ms: 8.0 + rand() * 1.5,
        latencyP95Ms: 13.5 + rand() * 2.5,
        stampMismatches: 0,
      },
      speedEnvelope: { vMaxKmh, bindingHazard, bindingRangeM },
    })
  }

  return frames
}

export const DEMO_SEQUENCE = generateSequence()

/** A DENSE (undecimated) baseline covering the SAME extent -- the
 * "uniform 5cm grid" comparison from eval/baselines.py's own Ticket
 * #56, at the SAME extent as the foveated demo sequence, computed
 * once (not per playback frame) since it represents a fixed reference
 * memory figure, not a moving scene. */
/** A naive flattened-2D occupancy view of a real frame: every
 * NEGATIVE_OBSTACLE cell is reclassified as DRIVABLE -- exactly what a
 * system that only asks "is there a return here" (rather than
 * reasoning about the RANGE SHADOW behind a drop-off, Bible Part 11)
 * would report, since the ground beyond a trench returns points same
 * as any other ground. This is the concrete, demoable form of demo
 * beat 1: "2D occupancy says CLEAR, DRISHTI says LETHAL." */
export function generateFlattenedVariant(frame: DemoFrame): DemoFrame {
  return {
    ...frame,
    cells: frame.cells.map((c) => (c.classId === 8 ? { ...c, classId: 1, heightM: 0 } : c)),
  }
}

let _denseBaseline: DemoFrame | null = null
export function generateDenseBaselineFrame(): DemoFrame {
  if (_denseBaseline) return _denseBaseline
  const cells: CellSample[] = []
  for (let i = -GRID_HALF; i <= GRID_HALF; i += 1) {
    for (let j = -GRID_HALF; j <= GRID_HALF; j += 1) {
      cells.push({
        i,
        j,
        classId: baseClassFor(i, j),
        heightM: heightField(i, j),
        observability: "OCCUPIED",
        sparsityVerdict: "NORMAL",
        isMoving: false,
      })
    }
  }
  _denseBaseline = {
    frame: -1,
    cells,
    hud: { cellsByLevel: [16_777_216], mbUsed: 201.3, latencyP50Ms: 0, latencyP95Ms: 0, stampMismatches: 0 },
    speedEnvelope: { vMaxKmh: 0, bindingHazard: "n/a", bindingRangeM: 0 },
  }
  return _denseBaseline
}
