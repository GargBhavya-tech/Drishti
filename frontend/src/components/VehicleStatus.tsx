/**
 * VehicleStatus.tsx -- a compact professional status card (engineering-
 * portal redesign, Section 11), replacing the old circular game-style
 * speedometer. Every value is read from real application state, never
 * fabricated:
 *
 * - Speed: the store's own `egoSpeedMs` (already the real value driving
 *   the gaze/fovea calculations elsewhere -- see Scene.tsx's GazeBeam),
 *   not the speed-envelope LIMIT (a different quantity: "how fast it's
 *   safe to go" vs. "how fast it's going").
 * - Heading: the bearing of the first segment of the SAME live-
 *   replanned A* route MissionStatus/GovernorPanel already compute.
 * - Route: MissionStatus's own clear/hazard derivation.
 * - Terrain confidence: the real fraction of this frame's cells whose
 *   observability is OCCUPIED (an actual confirmed return), not FREE/
 *   OCCLUDED/UNOBSERVED -- a genuine per-frame metric, not a placeholder.
 */

import { useMemo } from "react"
import { deriveMissionStatus, distanceFromHazardToPath } from "../lib/missionStatus"
import type { DemoFrame } from "../lib/mockData"
import { costGridFromFrame, findPath } from "../lib/pathPlanner"
import { smoothPath } from "../lib/pathSmoothing"
import type { GridPoint } from "../lib/pathSmoothing"
import { useDashboardStore } from "../state/store"

const PATH_GRID_HALF = 34 // kept in sync with Scene.tsx's own constant of the same name
const PATH_START: GridPoint = [PATH_GRID_HALF - 30, PATH_GRID_HALF + 17]
const PATH_GOAL: GridPoint = [PATH_GRID_HALF + 30, PATH_GRID_HALF + 17]

function headingDegrees(from: GridPoint, to: GridPoint): number {
  const dx = to[0] - from[0]
  const dz = to[1] - from[1]
  const rad = Math.atan2(dx, dz) // same convention Scene.tsx's VehicleMarker yaw uses
  return Math.round(((rad * 180) / Math.PI + 360) % 360)
}

function Row({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex items-baseline justify-between py-1 border-b border-[#D9E2EC] last:border-0">
      <span className="text-[12px] text-[#52606D]">{label}</span>
      <span className="text-[13px] font-mono-tech font-medium" style={{ color: accent ?? "#1F2933" }}>
        {value}
      </span>
    </div>
  )
}

export function VehicleStatus({ frame }: { frame: DemoFrame }) {
  const egoSpeedMs = useDashboardStore((s) => s.egoSpeedMs)

  const { headingDeg, route, confidencePct } = useMemo(() => {
    const { grid, size } = costGridFromFrame(frame, PATH_GRID_HALF)
    const result = findPath(grid, size, size, PATH_START, PATH_GOAL)
    const smoothed = result.path && result.path.length >= 2 ? smoothPath(result.path as GridPoint[]) : []
    const distanceM = distanceFromHazardToPath(frame, smoothed)
    const status = deriveMissionStatus(distanceM)

    const heading = smoothed.length >= 2 ? headingDegrees(smoothed[0], smoothed[1]) : 0

    const occupied = frame.cells.filter((c) => c.observability === "OCCUPIED").length
    const confidence = frame.cells.length > 0 ? (occupied / frame.cells.length) * 100 : 0

    return { headingDeg: heading, route: status.level, confidencePct: confidence }
  }, [frame])

  return (
    <div className="panel-light p-3">
      <div className="text-[11px] uppercase tracking-widest text-[#52606D] mb-2 font-semibold">Vehicle status</div>
      <Row label="Speed" value={`${(egoSpeedMs * 3.6).toFixed(1)} km/h`} />
      <Row label="Heading" value={`${String(headingDeg).padStart(3, "0")}°`} />
      <Row label="Route" value={route === "hazard" ? "REROUTED" : "SAFE"} accent={route === "hazard" ? "#C83C32" : "#087E8B"} />
      <Row label="Terrain confidence" value={`${confidencePct.toFixed(0)}%`} />
    </div>
  )
}
