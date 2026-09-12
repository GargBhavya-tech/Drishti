/**
 * MissionStatus.tsx -- Level 1 of the mission-control information
 * hierarchy: the single largest, most prominent thing on screen. A
 * DRDO evaluator should be able to read this line alone and know
 * whether the vehicle's current route is safe, with zero verbal
 * explanation -- see missionStatus.ts for how it's derived from the
 * SAME path-planning state the 3D scene and GovernorPanel already
 * compute (never a hardcoded/fabricated state).
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

export function MissionStatus({ frame }: { frame: DemoFrame }) {
  const status = useMemo(() => {
    const { grid, size } = costGridFromFrame(frame, PATH_GRID_HALF)
    const result = findPath(grid, size, size, PATH_START, PATH_GOAL)
    const smoothed = result.path && result.path.length >= 2 ? smoothPath(result.path as GridPoint[]) : []
    const distanceM = distanceFromHazardToPath(frame, smoothed)
    return deriveMissionStatus(distanceM)
  }, [frame])

  const isHazard = status.level === "hazard"

  return (
    <motion.div
      className="flex items-center gap-3"
      animate={{ opacity: 1 }}
      initial={{ opacity: 0 }}
      transition={{ duration: motionTokens.duration.normal }}
    >
      <motion.span
        key={status.level}
        className="h-2.5 w-2.5 rounded-full flex-none"
        animate={{ backgroundColor: isHazard ? "#E05245" : "#55D6E8" }}
        transition={{ duration: motionTokens.duration.fast }}
      />
      <div className="flex items-baseline gap-2.5 flex-wrap">
        <motion.span
          key={`headline-${status.level}`}
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: motionTokens.duration.normal, ease: motionTokens.easing.smooth }}
          className="font-sans text-[22px] leading-none font-semibold tracking-tight uppercase"
          style={{ color: isHazard ? "#E05245" : "#E7ECEE" }}
        >
          {status.headline}
        </motion.span>
        <span className="text-[13px] text-[#96A3A8]">
          {status.detail}
          {status.distanceM !== null && ` · ${status.distanceM.toFixed(0)}m ahead`}
        </span>
      </div>
    </motion.div>
  )
}
