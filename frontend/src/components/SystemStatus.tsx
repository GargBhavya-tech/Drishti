/**
 * SystemStatus.tsx -- Level 4 of the mission-control information
 * hierarchy: subsystem health. Each dot is tied to a real, checkable
 * condition in this frame's own data rather than a hardcoded "always
 * green" -- e.g. "Planning" genuinely goes to a degraded state if A*
 * fails to find a route this frame, which the app already has to
 * check anyway (Scene.tsx's PlannedPath silently renders nothing in
 * that case; this surfaces the same fact in words).
 */

import { useMemo } from "react"
import type { DemoFrame } from "../lib/mockData"
import { costGridFromFrame, findPath } from "../lib/pathPlanner"

const PATH_GRID_HALF = 34
const PATH_START: [number, number] = [PATH_GRID_HALF - 30, PATH_GRID_HALF + 17]
const PATH_GOAL: [number, number] = [PATH_GRID_HALF + 30, PATH_GRID_HALF + 17]

interface SubsystemState {
  label: string
  online: boolean
}

function useSubsystems(frame: DemoFrame): SubsystemState[] {
  return useMemo(() => {
    const hasCells = frame.cells.length > 0
    const { grid, size } = costGridFromFrame(frame, PATH_GRID_HALF)
    const result = findPath(grid, size, size, PATH_START, PATH_GOAL)
    const hasPath = !!result.path && result.path.length >= 2

    return [
      { label: "LiDAR", online: hasCells },
      { label: "Mapping", online: hasCells },
      { label: "Navigation", online: hasPath },
      { label: "Planning", online: hasPath },
    ]
  }, [frame])
}

export function SystemStatus({ frame }: { frame: DemoFrame }) {
  const subsystems = useSubsystems(frame)

  return (
    <div className="panel-light p-3">
      <div className="text-[11px] uppercase tracking-widest text-[#52606D] mb-2 font-semibold">System status</div>
      <div className="flex flex-col gap-1.5">
        {subsystems.map((s) => (
          <div key={s.label} className="flex items-center justify-between">
            <span className="text-[13px] text-[#1F2933]">{s.label}</span>
            <span
              className="flex items-center gap-1.5 text-[11px] font-mono-tech uppercase tracking-wide"
              style={{ color: s.online ? "#087E8B" : "#C83C32" }}
            >
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: s.online ? "#087E8B" : "#C83C32" }} />
              {s.online ? "Online" : "Degraded"}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
