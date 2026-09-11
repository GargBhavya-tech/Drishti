/**
 * Controls.tsx -- the bottom control strip: timeline scrubber +
 * play/pause, the live gamma slider (Ticket #47's own "put gamma on a
 * live slider and drive" -- computed client-side via foveaMath.ts, a
 * direct port of the real Python formulas), and the split-screen
 * toggle (Ticket #53).
 */

import { motion } from "motion/react"
import { useEffect, useRef } from "react"
import { DEFAULT_GAMMA } from "../lib/foveaMath"
import { motionTokens } from "../lib/theme"
import { useDashboardStore } from "../state/store"

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
    <motion.button
      onClick={onClick}
      aria-label={label}
      whileHover={{ scale: 1.06 }}
      whileTap={{ scale: 0.94 }}
      transition={{ duration: motionTokens.duration.fast, ease: motionTokens.easing.sharp }}
      className={`h-9 w-9 flex items-center justify-center rounded-full border transition-colors ${
        active ? "border-cyan-400/60 bg-cyan-400/10 text-cyan-300" : "border-white/10 bg-white/5 text-slate-300"
      }`}
    >
      {children}
    </motion.button>
  )
}

function PlayIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
      <path d="M2 1.5v11l10-5.5z" />
    </svg>
  )
}
function PauseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
      <rect x="2.5" y="1.5" width="3" height="11" />
      <rect x="8.5" y="1.5" width="3" height="11" />
    </svg>
  )
}
function StepIcon({ dir }: { dir: "back" | "fwd" }) {
  return (
    <svg width="12" height="12" viewBox="0 0 14 14" fill="currentColor" style={{ transform: dir === "back" ? "scaleX(-1)" : undefined }}>
      <path d="M2 1.5v11l8-5.5zM10.5 1.5h1.5v11h-1.5z" />
    </svg>
  )
}
function SplitIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4">
      <rect x="1" y="1.5" width="12" height="11" rx="1.2" />
      <line x1="7" y1="1.5" x2="7" y2="12.5" />
    </svg>
  )
}

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
    <div className="flex items-center gap-3 flex-1">
      <IconButton onClick={() => stepFrame(-1)} label="Previous frame">
        <StepIcon dir="back" />
      </IconButton>
      <IconButton onClick={togglePlaying} active={isPlaying} label={isPlaying ? "Pause" : "Play"}>
        {isPlaying ? <PauseIcon /> : <PlayIcon />}
      </IconButton>
      <IconButton onClick={() => stepFrame(1)} label="Next frame">
        <StepIcon dir="fwd" />
      </IconButton>

      <div className="flex-1 flex items-center gap-3">
        <input
          type="range"
          min={0}
          max={frameCount - 1}
          value={frameIndex}
          onChange={(e) => {
            setFrameIndex(Number(e.target.value))
          }}
          className="flex-1 accent-cyan-400"
        />
        <span className="font-mono-tech text-xs text-slate-400 w-16 text-right">
          {String(frameIndex + 1).padStart(2, "0")}/{frameCount}
        </span>
      </div>
    </div>
  )
}

export function GammaSlider() {
  const gamma = useDashboardStore((s) => s.gamma)
  const setGamma = useDashboardStore((s) => s.setGamma)

  return (
    <div className="flex items-center gap-3 min-w-[220px]">
      <span className="text-[11px] uppercase tracking-wider text-slate-500 font-mono-tech">gamma</span>
      <input
        type="range"
        min={0}
        max={2.5}
        step={0.05}
        value={gamma}
        onChange={(e) => setGamma(Number(e.target.value))}
        className="w-32 accent-amber-400"
      />
      <motion.span
        key={gamma.toFixed(2)}
        initial={{ opacity: 0.3 }}
        animate={{ opacity: 1 }}
        className="font-mono-tech text-sm text-amber-300 w-12"
      >
        {gamma.toFixed(2)}
      </motion.span>
      {gamma !== DEFAULT_GAMMA && (
        <button onClick={() => setGamma(DEFAULT_GAMMA)} className="text-[10px] text-slate-500 underline underline-offset-2">
          reset
        </button>
      )}
    </div>
  )
}

export function SplitScreenToggle() {
  const splitScreen = useDashboardStore((s) => s.splitScreen)
  const toggleSplitScreen = useDashboardStore((s) => s.toggleSplitScreen)
  return (
    <IconButton onClick={toggleSplitScreen} active={splitScreen} label="Toggle split-screen comparison">
      <SplitIcon />
    </IconButton>
  )
}
