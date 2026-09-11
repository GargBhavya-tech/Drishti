/**
 * pathSmoothing.ts -- a direct TypeScript port of planning/path_smoothing.py.
 * Same Catmull-Rom spline (NOT a true Dubins curve -- see the Python
 * module's own docstring for why: a Dubins curve's six case-by-case
 * word constructions are a lot of extra surface for a demo-time
 * addition, and Catmull-Rom gives a visually/numerically smooth path
 * through the same A* waypoints at a fraction of the complexity), same
 * circumradius curvature estimate, same friction-limited cornering-
 * speed relation, reusing frictionMath.ts's own CLASS_TO_MU so a tight
 * turn through mud is doubly penalised for ONE shared, honest reason
 * (low friction), not two independently-tuned numbers.
 */

import type { DrishtiClassId } from "./theme"
import { muForClass } from "./frictionMath"

export const G_MS2 = 9.81
const DEFAULT_SAMPLES_PER_SEGMENT = 8

export type GridPoint = [number, number]

function catmullRomAxis(a0: number, a1: number, a2: number, a3: number, t: number): number {
  const t2 = t * t
  const t3 = t2 * t
  return 0.5 * (
    2 * a1 +
    (-a0 + a2) * t +
    (2 * a0 - 5 * a1 + 4 * a2 - a3) * t2 +
    (-a0 + 3 * a1 - 3 * a2 + a3) * t3
  )
}

function catmullRomPoint(p0: GridPoint, p1: GridPoint, p2: GridPoint, p3: GridPoint, t: number): GridPoint {
  return [catmullRomAxis(p0[0], p1[0], p2[0], p3[0], t), catmullRomAxis(p0[1], p1[1], p2[1], p3[1], t)]
}

/** Catmull-Rom-interpolated points through `waypoints`. Endpoints are
 * duplicated (standard convention for an open curve) so the curve
 * reaches the true start/goal. Fewer than 3 waypoints passes through
 * unchanged -- nothing to smooth in a 0-2 point "path." */
export const smoothPath = (waypoints: GridPoint[], samplesPerSegment = DEFAULT_SAMPLES_PER_SEGMENT): GridPoint[] => {
  if (waypoints.length < 3) return waypoints.map((p) => [p[0], p[1]])
  const padded: GridPoint[] = [waypoints[0], ...waypoints, waypoints[waypoints.length - 1]]
  const out: GridPoint[] = []
  for (let i = 0; i < padded.length - 3; i++) {
    const [p0, p1, p2, p3] = [padded[i], padded[i + 1], padded[i + 2], padded[i + 3]]
    for (let s = 0; s < samplesPerSegment; s++) {
      out.push(catmullRomPoint(p0, p1, p2, p3, s / samplesPerSegment))
    }
  }
  out.push([...waypoints[waypoints.length - 1]])
  return out
}

/** Circumradius of the triangle p0-p1-p2 -- R = (a*b*c)/(4*Area).
 * Returns Infinity for (near-)collinear or coincident points (a
 * straight stretch's true zero curvature), not a divide-by-zero. */
export const curvatureRadius = (p0: GridPoint, p1: GridPoint, p2: GridPoint): number => {
  const a = Math.hypot(p1[0] - p0[0], p1[1] - p0[1])
  const b = Math.hypot(p2[0] - p1[0], p2[1] - p1[1])
  const c = Math.hypot(p2[0] - p0[0], p2[1] - p0[1])
  if (a === 0 || b === 0 || c === 0) return Infinity
  const area = Math.abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])) / 2
  if (area < 1e-9) return Infinity
  return (a * b * c) / (4 * area)
}

/** v_max(R) = sqrt(mu * g * R) -- the friction-limited cornering
 * speed. Infinity radius (a straight stretch) returns Infinity: no
 * curvature-imposed limit there. */
export const lateralSpeedLimitMs = (radiusM: number, mu: number, gMs2 = G_MS2): number => {
  if (mu <= 0) throw new Error(`mu must be positive, got ${mu}`)
  if (!Number.isFinite(radiusM)) return Infinity
  return Math.sqrt(mu * gMs2 * radiusM)
}

export interface CurvatureSpeedSample {
  point: GridPoint
  radiusM: number
  vMaxMs: number
  classId: DrishtiClassId
  mu: number
}

/** For every interior point of `smoothedPoints`, estimate curvature
 * radius (converted to metres via `cellSizeM`), look up the class
 * under it (`classAtPoint`), and derive the friction-limited cornering
 * speed via frictionMath's own muForClass -- mirrors
 * planning/path_smoothing.py's curvature_speed_profile() exactly. */
export const curvatureSpeedProfile = (
  smoothedPoints: GridPoint[],
  cellSizeM: number,
  classAtPoint: (p: GridPoint) => DrishtiClassId,
): CurvatureSpeedSample[] => {
  if (smoothedPoints.length < 3) return []
  const out: CurvatureSpeedSample[] = []
  for (let i = 1; i < smoothedPoints.length - 1; i++) {
    const pPrev = smoothedPoints[i - 1]
    const pCurr = smoothedPoints[i]
    const pNext = smoothedPoints[i + 1]
    const rGrid = curvatureRadius(pPrev, pCurr, pNext)
    const rM = Number.isFinite(rGrid) ? rGrid * cellSizeM : Infinity
    const classId = classAtPoint(pCurr)
    const mu = muForClass(classId)
    const vMax = lateralSpeedLimitMs(rM, mu)
    out.push({ point: pCurr, radiusM: rM, vMaxMs: vMax, classId, mu })
  }
  return out
}
