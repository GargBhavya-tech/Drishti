/**
 * HUD.tsx -- the stats panel (Ticket #52's own HUD text list: cells
 * per level, MB in use, latency P50/P95, stamp-mismatch counter,
 * estimated extrinsics) plus the overlay-layer toggle switches.
 * Numbers animate via a spring-driven counter (communicates a state
 * CHANGE, not decoration) rather than snapping between frames.
 */

import { motion, useReducedMotion, useSpring } from "motion/react"
import { useEffect, useRef, useState } from "react"
import type { OverlayMode } from "../state/store"
import { useDashboardStore } from "../state/store"
import type { HudStats } from "../lib/mockData"
import { motionTokens } from "../lib/theme"

function AnimatedNumber({ value, decimals = 2, suffix = "" }: { value: number; decimals?: number; suffix?: string }) {
  const reduce = useReducedMotion()
  const spring = useSpring(value, { stiffness: reduce ? 1000 : 120, damping: 20 })
  const [display, setDisplay] = useState(value.toFixed(decimals))

  useEffect(() => {
    spring.set(value)
  }, [value, spring])

  useEffect(() => {
    const unsub = spring.on("change", (v) => setDisplay(v.toFixed(decimals)))
    return unsub
  }, [spring, decimals])

  return (
    <span className="font-mono-tech tabular-nums">
      {display}
      {suffix}
    </span>
  )
}

function StatRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between py-1 border-b border-[#D9E2EC] last:border-0">
      <span className="text-[11px] uppercase tracking-wider text-[#52606D]">{label}</span>
      <span className="text-sm text-[#1F2933]">{children}</span>
    </div>
  )
}

const OVERLAY_OPTIONS: { id: OverlayMode; label: string }[] = [
  { id: "class", label: "Terrain type" },
  { id: "observability", label: "Sensor coverage" },
  { id: "sparsity", label: "Confidence" },
  { id: "height", label: "Elevation" },
  { id: "motion", label: "Moving objects" },
  { id: "attention", label: "Model attention (synthetic demo)" },
]

function OverlayToggle({ id, label }: { id: OverlayMode; label: string }) {
  const active = useDashboardStore((s) => s.overlayMode === id)
  const setOverlayMode = useDashboardStore((s) => s.setOverlayMode)

  return (
    <button
      onClick={() => setOverlayMode(id)}
      className="relative w-full text-left px-2.5 py-1.5 rounded-md text-xs text-[#52606D] overflow-hidden"
    >
      {active && (
        <motion.span
          layoutId="overlay-active-pill"
          className="absolute inset-0 bg-[#087E8B]/10 border border-[#087E8B]/40 rounded-md"
          transition={{ duration: motionTokens.duration.fast, ease: motionTokens.easing.smooth }}
        />
      )}
      <span className="relative z-10 flex items-center gap-2" style={active ? { color: "#17324D" } : undefined}>
        <span className={`h-1.5 w-1.5 rounded-full ${active ? "bg-[#087E8B]" : "bg-[#9AA5AB]"}`} />
        {label}
      </span>
    </button>
  )
}

export function HUD({ hud, frameIndex, frameCount }: { hud: HudStats; frameIndex: number; frameCount: number }) {
  const prevMismatches = useRef(hud.stampMismatches)
  const [flash, setFlash] = useState(false)

  useEffect(() => {
    if (hud.stampMismatches > prevMismatches.current) {
      setFlash(true)
      const t = setTimeout(() => setFlash(false), 400)
      return () => clearTimeout(t)
    }
    prevMismatches.current = hud.stampMismatches
  }, [hud.stampMismatches])

  return (
    <div className="flex flex-col gap-3">
      <motion.div
        className="panel-light p-3"
        initial={{ opacity: 0, y: -motionTokens.distance.sm }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: motionTokens.duration.normal, ease: motionTokens.easing.smooth }}
      >
        <div className="text-[11px] uppercase tracking-widest text-[#52606D] mb-2 font-semibold">Engineering telemetry</div>
        <StatRow label="Frame">
          {frameIndex + 1} / {frameCount}
        </StatRow>
        <StatRow label="Cells / level">262,144 x4</StatRow>
        <StatRow label="Memory (v1)">
          <AnimatedNumber value={hud.mbUsed} decimals={2} suffix=" MB" />
        </StatRow>
        <StatRow label="Latency P50">
          <AnimatedNumber value={hud.latencyP50Ms} decimals={1} suffix=" ms" />
        </StatRow>
        <StatRow label="Latency P95">
          <AnimatedNumber value={hud.latencyP95Ms} decimals={1} suffix=" ms" />
        </StatRow>
        <StatRow label="Stamp mismatches">
          <motion.span
            animate={flash ? { color: "#C83C32", scale: 1.15 } : { color: "#1F2933", scale: 1 }}
            transition={{ duration: motionTokens.duration.fast }}
            className="inline-block font-mono-tech"
          >
            {hud.stampMismatches}
          </motion.span>
        </StatRow>
        <StatRow label="Extrinsics">est. +-0.2 deg</StatRow>
      </motion.div>

      <motion.div
        className="panel-light p-3"
        initial={{ opacity: 0, y: -motionTokens.distance.sm }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: motionTokens.duration.normal, ease: motionTokens.easing.smooth, delay: 0.05 }}
      >
        <div className="text-[11px] uppercase tracking-widest text-[#52606D] mb-2 font-semibold">View layer</div>
        <div className="flex flex-col gap-1">
          {OVERLAY_OPTIONS.map((opt) => (
            <OverlayToggle key={opt.id} id={opt.id} label={opt.label} />
          ))}
        </div>
      </motion.div>
    </div>
  )
}
