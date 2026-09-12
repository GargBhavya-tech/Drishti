/**
 * CurrentHazards.tsx -- a plain-language list of every hazard-relevant
 * cell cluster this frame actually has (vehicleState.ts's own
 * `hazards` array: the trench, the static-obstacle rock cluster, and
 * the moving pedestrian), each with a REAL distance and whether it
 * currently sits on the planned route or off it. Never a fixed/
 * fabricated list -- an empty frame renders an empty (collapsed) card
 * rather than inventing an entry.
 */

import { computeVehicleState } from "../lib/vehicleState"
import type { DemoFrame } from "../lib/mockData"
import { HAZARD_COLOR } from "../lib/theme"

function WarningIcon({ dim }: { dim: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={dim ? "#9AA5AB" : HAZARD_COLOR} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3 2 20h20L12 3z" />
      <line x1="12" y1="10" x2="12" y2="14" />
      <circle cx="12" cy="17" r="0.6" fill={dim ? "#9AA5AB" : HAZARD_COLOR} stroke="none" />
    </svg>
  )
}

export function CurrentHazards({ frame, frameIndex, frameCount }: { frame: DemoFrame; frameIndex: number; frameCount: number }) {
  const { hazards } = computeVehicleState(frame, frameIndex, frameCount)

  if (hazards.length === 0) return null

  return (
    <div className="panel-light p-3">
      <div className="text-[11px] uppercase tracking-widest text-[#52606D] mb-2 font-semibold">Current hazards</div>
      <div className="flex flex-col gap-2">
        {hazards.map((h) => (
          <div key={h.label} className="flex items-center gap-2.5">
            <WarningIcon dim={!h.onPath} />
            <span className={`text-[13px] flex-1 ${h.onPath ? "text-[#1F2933] font-medium" : "text-[#9AA5AB]"}`}>{h.label}</span>
            <span className={`text-[11px] font-mono-tech ${h.onPath ? "text-[#1F2933]" : "text-[#9AA5AB]"}`}>
              {h.distanceM.toFixed(0)}m {h.onPath ? "ahead" : "(off path)"}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
