/**
 * ComparisonWipe.tsx -- an interactive "spot the difference" the judge
 * drags themselves: a 2D-flattened view ("clear") on the left, the
 * real multi-layer DRISHTI view ("LETHAL") on the right, revealed by
 * a draggable divider over the SAME real hazard. Demo beat 1, made
 * interactive instead of narrated -- a judge who finds the trench
 * themselves remembers it; one who is told about it does not.
 */

import { motion } from "motion/react"
import { useCallback, useRef, useState } from "react"
import type { DemoFrame } from "../lib/mockData"
import { generateFlattenedVariant } from "../lib/mockData"
import { motionTokens } from "../lib/theme"
import { Scene } from "./Scene"

export function ComparisonWipe({ frame }: { frame: DemoFrame }) {
  const [wipe, setWipe] = useState(50) // 0-100, percent revealed of the RIGHT (DRISHTI) side
  const containerRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef(false)

  const flattened = generateFlattenedVariant(frame)

  const updateFromClientX = useCallback((clientX: number) => {
    const el = containerRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const pct = ((clientX - rect.left) / rect.width) * 100
    setWipe(Math.min(100, Math.max(0, pct)))
  }, [])

  return (
    <motion.div
      ref={containerRef}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: motionTokens.duration.normal, ease: motionTokens.easing.smooth }}
      className="relative w-full h-full select-none cursor-ew-resize"
      onPointerDown={(e) => {
        draggingRef.current = true
        updateFromClientX(e.clientX)
        ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
      }}
      onPointerMove={(e) => {
        if (draggingRef.current) updateFromClientX(e.clientX)
      }}
      onPointerUp={() => {
        draggingRef.current = false
      }}
    >
      {/* Left, full width: the naive flattened 2D view -- "clear" */}
      <div className="absolute inset-0">
        <Scene frame={flattened} />
        <div className="absolute top-3 left-3 rounded-lg border border-white/10 bg-black/60 backdrop-blur px-3 py-1.5 pointer-events-none">
          <div className="text-[11px] uppercase tracking-widest text-slate-400">2D occupancy: CLEAR</div>
        </div>
      </div>

      {/* Right, clipped to the wipe position: the real DRISHTI view -- "LETHAL" */}
      <div
        className="absolute inset-0 overflow-hidden pointer-events-none"
        style={{ clipPath: `inset(0 0 0 ${wipe}%)` }}
      >
        <div className="absolute inset-0 pointer-events-auto">
          <Scene frame={frame} />
        </div>
        <div className="absolute top-3 right-3 rounded-lg border border-red-400/30 bg-black/60 backdrop-blur px-3 py-1.5">
          <div className="text-[11px] uppercase tracking-widest text-red-400">DRISHTI: LETHAL</div>
        </div>
      </div>

      {/* The divider handle itself */}
      <div className="absolute top-0 bottom-0 w-[2px] bg-cyan-300/80 pointer-events-none" style={{ left: `${wipe}%` }}>
        <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-9 w-9 rounded-full bg-cyan-300 flex items-center justify-center shadow-lg">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="#05070c" strokeWidth="1.6">
            <path d="M4 3 L1 7 L4 11 M10 3 L13 7 L10 11" />
          </svg>
        </div>
      </div>

      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full border border-white/10 bg-black/60 backdrop-blur px-4 py-1.5 text-[11px] text-slate-400 pointer-events-none">
        drag to compare -- same real hazard, two ways of seeing it
      </div>
    </motion.div>
  )
}
