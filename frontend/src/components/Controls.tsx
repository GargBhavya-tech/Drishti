/**
 * Controls.tsx -- the bottom control strip: view-layer tabs (which real
 * overlay mode is active), the play/timeline transport, the split/
 * compare/real-data toggles, and the gamma slider.
 *
 * The reference layout calls for five tabs (Terrain/Path/Hazards/
 * Sensors/3D View). This app only has FOUR overlay modes worth
 * surfacing as primary, mutually-exclusive tabs -- there is no distinct
 * "path-only" rendering mode to point a fifth tab at (the planned route
 * is always visible regardless of which overlay is active), so adding
 * one would either duplicate Terrain or be a label with no real state
 * behind it. Four real tabs instead of five-with-one-fake one:
 * TERRAIN (class colour), HAZARDS (motion/moving-hazard emphasis),
 * SENSORS (observability), and 3D VIEW, which is the existing real-data
 * toggle (switching to the real RELLIS-3D + FusionSegNet 3D view) --
 * not a placeholder, an already-real feature that maps cleanly onto
 * exactly what "3D View" would mean.
 */

import { motion } from "motion/react"
import { useEffect, useRef } from "react"
import { DEFAULT_GAMMA } from "../lib/foveaMath"
import type { OverlayMode } from "../state/store"
import { useDashboardStore } from "../state/store"

function PlayIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 14 14" fill="currentColor">
      <path d="M2 1.5v11l10-5.5z" />
    </svg>
  )
}
function PauseIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 14 14" fill="currentColor">
      <rect x="2.5" y="1.5" width="3" height="11" />
      <rect x="8.5" y="1.5" width="3" height="11" />
    </svg>
  )
}
function StepIcon({ dir }: { dir: "back" | "fwd" }) {
  return (
    <svg width="11" height="11" viewBox="0 0 14 14" fill="currentColor" style={{ transform: dir === "back" ? "scaleX(-1)" : undefined }}>
      <path d="M2 1.5v11l8-5.5zM10.5 1.5h1.5v11h-1.5z" />
    </svg>
  )
}
function TerrainIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round">
      <path d="M3 20 9 8l4 6 2-3 6 9z" />
    </svg>
  )
}
function RouteIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M4 20c4-8 6 4 10-4s4-8 6-8" />
    </svg>
  )
}
function HazardIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round">
      <path d="M12 3 2 20h20L12 3z" />
      <line x1="12" y1="10" x2="12" y2="14" />
    </svg>
  )
}
function SensorsIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="12" r="2" fill="currentColor" stroke="none" />
      <path d="M8.5 8.5a5 5 0 0 0 0 7M15.5 8.5a5 5 0 0 1 0 7M5.5 5.5a9 9 0 0 0 0 13M18.5 5.5a9 9 0 0 1 0 13" />
    </svg>
  )
}
function CubeIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round">
      <path d="M12 3 4 7v10l8 4 8-4V7z" />
      <path d="M4 7l8 4 8-4M12 11v10" />
    </svg>
  )
}
function SplitIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4">
      <rect x="1" y="1.5" width="12" height="11" rx="1.2" />
      <line x1="7" y1="1.5" x2="7" y2="12.5" />
    </svg>
  )
}
function WipeIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4">
      <rect x="1" y="1.5" width="12" height="11" rx="1.2" />
      <path d="M5 3 L3 7 L5 11 M9 3 L11 7 L9 11" />
    </svg>
  )
}

/** The primary view-layer tab strip -- a mutually-exclusive selector,
 * visually distinct from the (secondary) split/compare toggles. */
export function ViewLayerTabs() {
  const overlayMode = useDashboardStore((s) => s.overlayMode)
  const setOverlayMode = useDashboardStore((s) => s.setOverlayMode)
  const realDataMode = useDashboardStore((s) => s.realDataMode)
  const toggleRealDataMode = useDashboardStore((s) => s.toggleRealDataMode)

  const tabs: { id: OverlayMode | "3d-view"; label: string; icon: React.ReactNode }[] = [
    { id: "class", label: "Terrain", icon: <TerrainIcon /> },
    { id: "motion", label: "Hazards", icon: <HazardIcon /> },
    { id: "observability", label: "Sensors", icon: <SensorsIcon /> },
    { id: "3d-view", label: "3D View", icon: <CubeIcon /> },
  ]

  return (
    <div className="flex items-center gap-1">
      {tabs.map((tab) => {
        const active = tab.id === "3d-view" ? realDataMode : !realDataMode && overlayMode === tab.id
        return (
          <button
            key={tab.id}
            onClick={() => (tab.id === "3d-view" ? toggleRealDataMode() : setOverlayMode(tab.id as OverlayMode))}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-[13px] font-semibold transition-colors ${
              active ? "bg-[#17324D] text-white" : "bg-white text-[#52606D] border border-[#D9E2EC]"
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        )
      })}
    </div>
  )
}

/** The transport pill: play/pause/step + progress + frame count, in a
 * single dark control -- visually the "this is a live replay" anchor of
 * the bottom bar. */
export function Timeline({ frameCount }: { frameCount: number }) {
  const frameIndex = useDashboardStore((s) => s.frameIndex)
  const isPlaying = useDashboardStore((s) => s.isPlaying)
  const setFrameIndex = useDashboardStore((s) => s.setFrameIndex)
  const stepFrame = useDashboardStore((s) => s.stepFrame)
  const togglePlaying = useDashboardStore((s) => s.togglePlaying)

  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)

  useEffect(() => {
    if (!isPlaying) return
    const FRAME_MS = 140

    const tick = (t: number) => {
      if (t - lastTickRef.current >= FRAME_MS) {
        stepFrame(1)
        lastTickRef.current = t
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [isPlaying, stepFrame])

  return (
    <div className="flex items-center gap-3 bg-[#17324D] rounded-full pl-2 pr-4 py-2 flex-1 max-w-xl">
      <button
        onClick={() => stepFrame(-1)}
        aria-label="Previous frame"
        className="h-7 w-7 flex items-center justify-center rounded-full text-white/70 hover:text-white"
      >
        <StepIcon dir="back" />
      </button>
      <button
        onClick={togglePlaying}
        aria-label={isPlaying ? "Pause" : "Play"}
        className="h-8 w-8 flex items-center justify-center rounded-full bg-white text-[#17324D]"
      >
        {isPlaying ? <PauseIcon /> : <PlayIcon />}
      </button>
      <button
        onClick={() => stepFrame(1)}
        aria-label="Next frame"
        className="h-7 w-7 flex items-center justify-center rounded-full text-white/70 hover:text-white"
      >
        <StepIcon dir="fwd" />
      </button>

      <input
        type="range"
        min={0}
        max={frameCount - 1}
        value={frameIndex}
        onChange={(e) => setFrameIndex(Number(e.target.value))}
        className="flex-1 accent-[#087E8B]"
      />
      <span className="font-mono-tech text-xs text-white/80 w-14 text-right">
        {frameIndex + 1}/{frameCount}
      </span>
    </div>
  )
}

export function GammaSlider() {
  const gamma = useDashboardStore((s) => s.gamma)
  const setGamma = useDashboardStore((s) => s.setGamma)

  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] uppercase tracking-wider text-[#52606D] font-mono-tech">gamma</span>
      <input
        type="range"
        min={0}
        max={2.5}
        step={0.05}
        value={gamma}
        onChange={(e) => setGamma(Number(e.target.value))}
        className="w-20 accent-[#17324D]"
      />
      <motion.span
        key={gamma.toFixed(2)}
        initial={{ opacity: 0.3 }}
        animate={{ opacity: 1 }}
        className="font-mono-tech text-xs text-[#17324D] w-9"
      >
        {gamma.toFixed(2)}
      </motion.span>
      {gamma !== DEFAULT_GAMMA && (
        <button onClick={() => setGamma(DEFAULT_GAMMA)} className="text-[10px] text-[#52606D] underline underline-offset-2">
          reset
        </button>
      )}
    </div>
  )
}

function IconButton({
  onClick,
  active,
  children,
  label,
}: {
  onClick: () => void
  active?: boolean
  children: React.ReactNode
  label: string
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      title={label}
      className={`h-8 w-8 flex items-center justify-center rounded-md border transition-colors ${
        active ? "border-[#087E8B]/50 bg-[#087E8B]/10 text-[#087E8B]" : "border-[#D9E2EC] bg-white text-[#52606D]"
      }`}
    >
      {children}
    </button>
  )
}

export function SplitScreenToggle() {
  const splitScreen = useDashboardStore((s) => s.splitScreen)
  const toggleSplitScreen = useDashboardStore((s) => s.toggleSplitScreen)
  return (
    <IconButton
      onClick={toggleSplitScreen}
      active={splitScreen}
      label="Split view: uniform-grid baseline vs. DRISHTI foveated map"
    >
      <SplitIcon />
    </IconButton>
  )
}

export function CompareWipeToggle() {
  const compareWipe = useDashboardStore((s) => s.compareWipe)
  const toggleCompareWipe = useDashboardStore((s) => s.toggleCompareWipe)
  return (
    <IconButton onClick={toggleCompareWipe} active={compareWipe} label="Compare: 2D occupancy vs. real DRISHTI map">
      <WipeIcon />
    </IconButton>
  )
}

/** Kept for backward compatibility with any remaining import -- 3D View
 * is now one of ViewLayerTabs's own tabs (real-data mode), so this
 * standalone toggle is no longer rendered in App.tsx's bottom bar. */
export function RealDataToggle() {
  const realDataMode = useDashboardStore((s) => s.realDataMode)
  const toggleRealDataMode = useDashboardStore((s) => s.toggleRealDataMode)
  return (
    <IconButton onClick={toggleRealDataMode} active={realDataMode} label="Switch to real RELLIS-3D + FusionSegNet 3D view">
      <RouteIcon />
    </IconButton>
  )
}
