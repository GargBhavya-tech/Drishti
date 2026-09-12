/**
 * MissionStatus.tsx -- Level 1 of the information hierarchy: a floating
 * card over the 3D viewport itself (not header text), so it reads as
 * "the current situation, overlaid on the map it's about" rather than
 * a banner. A DRDO evaluator should read this alone and know whether
 * the route is safe, with zero explanation -- see missionStatus.ts for
 * how it's derived from the SAME path-planning state the 3D scene and
 * every other panel already compute (never a hardcoded/fabricated
 * state).
 */

import { motion } from "motion/react"
import { useMemo } from "react"
import { deriveMissionStatus, distanceFromHazardToPath } from "../lib/missionStatus"
import type { DemoFrame } from "../lib/mockData"
import { costGridFromFrame, findPath } from "../lib/pathPlanner"
import { smoothPath } from "../lib/pathSmoothing"
import type { GridPoint } from "../lib/pathSmoothing"
import { motionTokens } from "../lib/theme"

const PATH_GRID_HALF = 34 // kept in sync with Scene.tsx's own constant of the same name
const PATH_START: GridPoint = [PATH_GRID_HALF - 30, PATH_GRID_HALF + 17]
const PATH_GOAL: GridPoint = [PATH_GRID_HALF + 30, PATH_GRID_HALF + 17]

function ShieldIcon({ ok }: { ok: boolean }) {
  return ok ? (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2 4 5v6c0 5 3.4 8.7 8 10 4.6-1.3 8-5 8-10V5z" fill="none" />
      <path d="M8.5 12.5l2.5 2.5 4.5-5" />
    </svg>
  ) : (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2 4 5v6c0 5 3.4 8.7 8 10 4.6-1.3 8-5 8-10V5z" fill="none" />
      <line x1="12" y1="8" x2="12" y2="13" />
      <circle cx="12" cy="16.5" r="0.9" fill="white" stroke="none" />
    </svg>
  )
}

export function MissionStatus({ frame }: { frame: DemoFrame }) {
  const status = useMemo(() => {
    const { grid, size } = costGridFromFrame(frame, PATH_GRID_HALF)
    const result = findPath(grid, size, size, PATH_START, PATH_GOAL)
    const smoothed = result.path && result.path.length >= 2 ? smoothPath(result.path as GridPoint[]) : []
    const distanceM = distanceFromHazardToPath(frame, smoothed)
    return deriveMissionStatus(distanceM)
  }, [frame])

  const isHazard = status.level === "hazard"
  const accent = isHazard ? "#C83C32" : "#0F9D58"

  return (
    <motion.div
      className="flex items-center gap-3 pointer-events-auto bg-white rounded-xl border border-[#D9E2EC] shadow-[0_4px_16px_rgba(16,42,67,0.12)] px-4 py-3 pr-5"
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: motionTokens.duration.normal, ease: motionTokens.easing.smooth }}
    >
      <motion.div
        key={status.level}
        className="h-9 w-9 rounded-full flex items-center justify-center flex-none"
        animate={{ backgroundColor: accent }}
        transition={{ duration: motionTokens.duration.fast }}
      >
        <ShieldIcon ok={!isHazard} />
      </motion.div>
      <div className="flex flex-col">
        <motion.span
          key={`headline-${status.level}`}
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: motionTokens.duration.normal, ease: motionTokens.easing.smooth }}
          className="font-sans text-[19px] leading-tight font-bold uppercase"
          style={{ color: accent }}
        >
          {status.headline}
        </motion.span>
        <span className="text-[13px] text-[#52606D] leading-tight">
          {status.detail}
          {status.distanceM !== null && ` · ${status.distanceM.toFixed(0)}m ahead`}
        </span>
      </div>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#9AA5AB" strokeWidth="2" className="ml-2 flex-none">
        <path d="M9 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </motion.div>
  )
}
