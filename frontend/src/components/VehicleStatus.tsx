/**
 * VehicleStatus.tsx -- a compact professional status card, replacing
 * the old circular game-style speedometer. Every value comes from
 * vehicleState.ts's single shared derivation (see that module's own
 * doc comment) so this card can never disagree with CurrentHazards,
 * MissionStatus, or the 3D scene about where the vehicle is or whether
 * the route is clear.
 */

import { computeVehicleState } from "../lib/vehicleState"
import type { DemoFrame } from "../lib/mockData"
import { useDashboardStore } from "../state/store"

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

/** A small top-down vehicle glyph -- a static icon standing in for a
 * live-rendered thumbnail (this app has no snapshot-of-the-3D-scene
 * pipeline), styled to match UgvModel.tsx's own silhouette/material so
 * it reads as "the same vehicle," not a generic stock icon. */
function VehicleGlyph() {
  return (
    <div className="h-11 w-11 rounded-md bg-[#F3F5F6] border border-[#D9E2EC] flex items-center justify-center flex-none">
      <svg width="26" height="26" viewBox="0 0 32 32">
        <rect x="4" y="10" width="24" height="14" rx="2" fill="#3E4648" />
        <rect x="9" y="7" width="14" height="7" rx="1.5" fill="#7C8789" />
        <circle cx="9" cy="24" r="3" fill="#20272A" />
        <circle cx="23" cy="24" r="3" fill="#20272A" />
        <circle cx="9" cy="10" r="3" fill="#20272A" />
        <circle cx="23" cy="10" r="3" fill="#20272A" />
        <circle cx="25" cy="16" r="1.4" fill="#087E8B" />
      </svg>
    </div>
  )
}

export function VehicleStatus({ frame, frameIndex, frameCount }: { frame: DemoFrame; frameIndex: number; frameCount: number }) {
  const egoSpeedMs = useDashboardStore((s) => s.egoSpeedMs)
  const state = computeVehicleState(frame, frameIndex, frameCount)

  return (
    <div className="panel-light p-3">
      <div className="flex items-center gap-3 mb-2.5">
        <VehicleGlyph />
        <div className="text-[11px] uppercase tracking-widest text-[#52606D] font-semibold">Vehicle status</div>
      </div>
      <Row label="Speed" value={`${(egoSpeedMs * 3.6).toFixed(1)} km/h`} />
      <Row label="Heading" value={`${String(state.headingDeg).padStart(3, "0")}°`} />
      <Row
        label="Route"
        value={state.route === "hazard" ? "REROUTED" : "SAFE"}
        accent={state.route === "hazard" ? "#C83C32" : "#087E8B"}
      />
      <Row label="Terrain confidence" value={`${state.confidencePct.toFixed(0)}%`} />
      <Row label="Position (local)" value={`${state.positionLocal.x.toFixed(1)}, ${state.positionLocal.z.toFixed(1)}, ${state.positionLocal.heightM.toFixed(1)}`} />
    </div>
  )
}
