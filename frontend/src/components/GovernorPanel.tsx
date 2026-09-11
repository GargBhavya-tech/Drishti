/**
 * GovernorPanel.tsx -- reports WHY the kinodynamic + semantic-friction
 * governor would tell the vehicle to slow down right now: the worst-
 * friction DRISHTI class along the live-replanned path (the braking
 * budget) and the tightest corner's own curvature-limited cornering
 * speed. Reads the SAME path Scene.tsx's own PlannedPath draws --
 * recomputed independently here (cheap on this grid size) rather than
 * threaded through props, matching this app's existing loose coupling
 * between Scene and the sidebar via the zustand store. Mirrors
 * planning/speed_envelope.py's own "report the binding constraint, not
 * just the number" discipline.
 */

import { motion } from "motion/react"
import { useMemo } from "react"
import { bindingMu, derateBrakingA } from "../lib/frictionMath"
import type { DemoFrame } from "../lib/mockData"
import { classGridFromFrame, costGridFromFrame, findPath } from "../lib/pathPlanner"
import type { GridPoint } from "../lib/pathSmoothing"
import { curvatureSpeedProfile, smoothPath } from "../lib/pathSmoothing"
import { DRISHTI_CLASS_NAMES, motionTokens } from "../lib/theme"

const PATH_GRID_HALF = 34 // kept in sync with Scene.tsx's own constant of the same name
const PATH_START: GridPoint = [PATH_GRID_HALF - 30, PATH_GRID_HALF + 17]
const PATH_GOAL: GridPoint = [PATH_GRID_HALF + 30, PATH_GRID_HALF + 17]

// Mirrors configs/vehicle_ugv.yaml's own declared braking_a_ms2 (the
// dry-ground reference this project's speed envelope already assumes).
const BRAKING_A_MS2_DRY = 4.0

interface GovernorSummary {
  mu: number
  bindingClass: number | null
  deratedAMs2: number
  tightestVMaxMs: number
  tightestRadiusM: number
}

function computeGovernorSummary(frame: DemoFrame): GovernorSummary | null {
  const { grid, size } = costGridFromFrame(frame, PATH_GRID_HALF)
  const result = findPath(grid, size, size, PATH_START, PATH_GOAL)
  if (!result.path || result.path.length < 2) return null

  const { classGrid } = classGridFromFrame(frame, PATH_GRID_HALF)
  const classesAlongPath = result.path.map(([r, c]) => classGrid[r * size + c])
  const { mu, bindingClass } = bindingMu(classesAlongPath)
  const deratedAMs2 = derateBrakingA(BRAKING_A_MS2_DRY, mu)

  const smoothed = smoothPath(result.path as GridPoint[])
  const classAt = (p: GridPoint): number => {
    const r = Math.round(Math.max(0, Math.min(size - 1, p[0])))
    const c = Math.round(Math.max(0, Math.min(size - 1, p[1])))
    return classGrid[r * size + c]
  }
  const profile = curvatureSpeedProfile(smoothed, 1.0, classAt)
  const tightest = profile.reduce<{ vMaxMs: number; radiusM: number }>(
    (worst, s) => (s.vMaxMs < worst.vMaxMs ? { vMaxMs: s.vMaxMs, radiusM: s.radiusM } : worst),
    { vMaxMs: Infinity, radiusM: Infinity },
  )

  return { mu, bindingClass, deratedAMs2, tightestVMaxMs: tightest.vMaxMs, tightestRadiusM: tightest.radiusM }
}

export function GovernorPanel({ frame }: { frame: DemoFrame }) {
  const summary = useMemo(() => computeGovernorSummary(frame), [frame])
  if (!summary) return null

  const bindingLabel = summary.bindingClass !== null ? DRISHTI_CLASS_NAMES[summary.bindingClass] : "DRIVABLE (dry)"
  const hasCornerLimit = Number.isFinite(summary.tightestVMaxMs)

  return (
    <motion.div
      className="rounded-xl border border-white/10 bg-white/[0.03] backdrop-blur-md p-3"
      initial={{ opacity: 0, y: motionTokens.distance.sm }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: motionTokens.duration.normal, ease: motionTokens.easing.smooth }}
    >
      <div className="text-[11px] uppercase tracking-widest text-cyan-400/80 mb-2">Kinodynamic governor</div>
      <div className="flex items-baseline justify-between py-1 border-b border-white/5">
        <span className="text-[11px] uppercase tracking-wider text-slate-500">Terrain grip</span>
        <span className="text-sm text-slate-200">
          {bindingLabel} (mu {summary.mu.toFixed(2)})
        </span>
      </div>
      <div className="flex items-baseline justify-between py-1 border-b border-white/5">
        <span className="text-[11px] uppercase tracking-wider text-slate-500">Derated braking a_max</span>
        <span className="text-sm text-slate-200">{summary.deratedAMs2.toFixed(2)} m/s^2</span>
      </div>
      <div className="flex items-baseline justify-between py-1">
        <span className="text-[11px] uppercase tracking-wider text-slate-500">Tightest corner</span>
        <span className="text-sm text-slate-200">
          {hasCornerLimit
            ? `R ${summary.tightestRadiusM.toFixed(1)}m -> ${(summary.tightestVMaxMs * 3.6).toFixed(1)} km/h`
            : "no limit"}
        </span>
      </div>
    </motion.div>
  )
}
