/**
 * missionStatus.ts -- derives the single plain-language "what is
 * happening right now" line (mission-control redesign's Level 1
 * status) from the SAME per-frame state every other panel already
 * reads: the live-replanned A* path (pathPlanner.ts) and the frame's
 * own cell list. Never a hardcoded/fabricated state -- if there is no
 * hazard cell within `PROXIMITY_THRESHOLD_M` of the current route, the
 * status is genuinely "clear," and it flips the moment a hazard cell
 * (moving pedestrian or the static negative-obstacle trench) comes
 * within range of the route the vehicle would actually drive.
 */

import type { DemoFrame } from "./mockData"
import type { GridPoint } from "./pathSmoothing"

const PROXIMITY_THRESHOLD_M = 12
// Hazard-relevant classes: PEDESTRIAN (moving) and NEGATIVE_OBSTACLE
// (structural) -- the same two classes attention/fovea_controller.py's
// gaze-steering treats as saccade-worthy (see Scene.tsx's GazeBeam).
const HAZARD_CLASS_IDS = new Set([7, 8])

export type MissionStatusLevel = "clear" | "hazard"

export interface MissionStatus {
  level: MissionStatusLevel
  headline: string
  detail: string
  distanceM: number | null
}

function pointToSegmentDistance(p: [number, number], a: [number, number], b: [number, number]): number {
  const [px, py] = p
  const [ax, ay] = a
  const [bx, by] = b
  const dx = bx - ax
  const dy = by - ay
  const lenSq = dx * dx + dy * dy
  const t = lenSq === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq))
  const cx = ax + t * dx
  const cy = ay + t * dy
  return Math.hypot(px - cx, py - cy)
}

/** Minimum distance (metres) from any (i, j) point in `points` to the
 * polyline `pathPoints` (grid-index units, one unit == one metre -- this
 * codebase's own established convention). `null` if `points` is empty. */
export function minDistancePointsToPath(points: { i: number; j: number }[], pathPoints: GridPoint[]): number | null {
  if (points.length === 0 || pathPoints.length < 2) return null

  let minDist = Infinity
  for (const point of points) {
    const p: [number, number] = [point.i, point.j]
    for (let i = 0; i < pathPoints.length - 1; i++) {
      const a: [number, number] = [pathPoints[i][0], pathPoints[i][1]]
      const b: [number, number] = [pathPoints[i + 1][0], pathPoints[i + 1][1]]
      const d = pointToSegmentDistance(p, a, b)
      if (d < minDist) minDist = d
    }
  }
  return Number.isFinite(minDist) ? minDist : null
}

/** Minimum distance (metres) from any hazard-class cell in `frame` to
 * the polyline `pathPoints`. `null` if there is no hazard-class cell in
 * the frame at all. */
export function distanceFromHazardToPath(frame: DemoFrame, pathPoints: GridPoint[]): number | null {
  const hazardCells = frame.cells.filter((c) => HAZARD_CLASS_IDS.has(c.classId))
  return minDistancePointsToPath(hazardCells, pathPoints)
}

export function deriveMissionStatus(distanceM: number | null): MissionStatus {
  if (distanceM === null || distanceM > PROXIMITY_THRESHOLD_M) {
    return {
      level: "clear",
      headline: "Path clear",
      detail: "Safe route confirmed",
      distanceM,
    }
  }
  return {
    level: "hazard",
    headline: "Hazard detected ahead",
    detail: "Safe route recalculated",
    distanceM,
  }
}
