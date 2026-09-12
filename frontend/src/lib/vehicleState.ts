/**
 * vehicleState.ts -- the ONE place that derives "where is the vehicle,
 * which way is it facing, is the route clear, what hazards exist" from
 * real per-frame data (cell classes, the live-replanned A* route). Used
 * by every UI panel that needs to agree on these numbers (VehicleStatus,
 * CurrentHazards, MissionStatus) so they can never drift from each
 * other or from what Scene.tsx's 3D view actually shows. Nothing here
 * is fabricated: every figure traces back to frame.cells or the real
 * path-planning/-smoothing modules.
 */

import { deriveMissionStatus, distanceFromHazardToPath, minDistancePointsToPath } from "./missionStatus"
import type { DemoFrame } from "./mockData"
import { costGridFromFrame, findPath } from "./pathPlanner"
import { smoothPath } from "./pathSmoothing"
import type { GridPoint } from "./pathSmoothing"

const PATH_GRID_HALF = 34 // kept in sync with Scene.tsx's own constant of the same name
const PATH_START: GridPoint = [PATH_GRID_HALF - 30, PATH_GRID_HALF + 17]
const PATH_GOAL: GridPoint = [PATH_GRID_HALF + 30, PATH_GRID_HALF + 17]

const PEDESTRIAN_CLASS_ID = 7
const NEGATIVE_OBSTACLE_CLASS_ID = 8
const STATIC_OBSTACLE_CLASS_ID = 4
const ON_PATH_THRESHOLD_M = 12 // matches missionStatus.ts's own PROXIMITY_THRESHOLD_M

export interface HazardInfo {
  label: string
  distanceM: number
  onPath: boolean
}

export interface VehicleState {
  /** (i, j) in metres -- this codebase's own "one grid-index unit is
   * one metre" convention -- plus the nearest real cell's own height,
   * as a real (not interpolated-for-display) reading. */
  positionLocal: { x: number; z: number; heightM: number }
  headingDeg: number
  route: "clear" | "hazard"
  confidencePct: number
  hazards: HazardInfo[]
}

function headingDegrees(from: GridPoint, to: GridPoint): number {
  const dx = to[0] - from[0]
  const dz = to[1] - from[1]
  const rad = Math.atan2(dx, dz) // same convention Scene.tsx's vehicle group yaw uses
  return Math.round(((rad * 180) / Math.PI + 360) % 360)
}

function nearestCellHeight(frame: DemoFrame, i: number, j: number): number {
  let best = 0
  let bestDist = Infinity
  for (const c of frame.cells) {
    const d = Math.hypot(c.i - i, c.j - j)
    if (d < bestDist) {
      bestDist = d
      best = c.heightM
    }
  }
  return best
}

function hazardInfoForClass(frame: DemoFrame, classId: number, label: string, pathPoints: GridPoint[]): HazardInfo | null {
  const cells = frame.cells.filter((c) => c.classId === classId)
  if (cells.length === 0) return null

  let sumI = 0
  let sumJ = 0
  for (const c of cells) {
    sumI += c.i
    sumJ += c.j
  }
  const centerI = sumI / cells.length
  const centerJ = sumJ / cells.length
  const distanceM = Math.hypot(centerI, centerJ) // distance from the map's own origin, matching Scene.tsx's marker labels

  const distToPath = minDistancePointsToPath(cells, pathPoints)
  const onPath = distToPath !== null && distToPath <= ON_PATH_THRESHOLD_M

  return { label, distanceM, onPath }
}

/** `nFrames` is the demo sequence length -- passed in rather than
 * imported, so this module stays a pure function of its arguments. */
export function computeVehicleState(frame: DemoFrame, frameIndex: number, nFrames: number): VehicleState {
  const { grid, size } = costGridFromFrame(frame, PATH_GRID_HALF)
  const result = findPath(grid, size, size, PATH_START, PATH_GOAL)
  const smoothed = result.path && result.path.length >= 2 ? smoothPath(result.path as GridPoint[]) : []

  const distanceM = distanceFromHazardToPath(frame, smoothed)
  const status = deriveMissionStatus(distanceM)

  const progress = nFrames > 1 ? frameIndex / (nFrames - 1) : 0
  const idxF = progress * Math.max(0, smoothed.length - 1)
  const i0 = Math.max(0, Math.min(smoothed.length - 2, Math.floor(idxF)))
  const i1 = Math.min(smoothed.length - 1, i0 + 1)
  const t = smoothed.length >= 2 ? idxF - i0 : 0

  const posGrid: GridPoint =
    smoothed.length >= 2
      ? [smoothed[i0][0] + (smoothed[i1][0] - smoothed[i0][0]) * t, smoothed[i0][1] + (smoothed[i1][1] - smoothed[i0][1]) * t]
      : [PATH_START[0], PATH_START[1]]

  const posI = posGrid[0] - PATH_GRID_HALF
  const posJ = posGrid[1] - PATH_GRID_HALF
  const headingDeg = smoothed.length >= 2 ? headingDegrees(smoothed[i0], smoothed[i1]) : 0

  const occupied = frame.cells.filter((c) => c.observability === "OCCUPIED").length
  const confidencePct = frame.cells.length > 0 ? (occupied / frame.cells.length) * 100 : 0

  const hazards: HazardInfo[] = [
    hazardInfoForClass(frame, NEGATIVE_OBSTACLE_CLASS_ID, "Trench (high risk)", smoothed),
    hazardInfoForClass(frame, STATIC_OBSTACLE_CLASS_ID, "Rock cluster", smoothed),
    hazardInfoForClass(frame, PEDESTRIAN_CLASS_ID, "Pedestrian (moving)", smoothed),
  ].filter((h): h is HazardInfo => h !== null)

  return {
    positionLocal: { x: posI, z: posJ, heightM: nearestCellHeight(frame, posI, posJ) },
    headingDeg,
    route: status.level,
    confidencePct,
    hazards,
  }
}
