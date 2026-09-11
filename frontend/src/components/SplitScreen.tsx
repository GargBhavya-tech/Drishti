/**
 * SplitScreen.tsx -- Ticket #53: uniform 5cm grid (left) vs. the
 * foveated DRISHTI map (right), memory counters under each, matching
 * eval/baselines.py's own "identical extent, identical bytes per
 * cell" framing. The left panel is a fixed dense reference (the
 * finest map buildable from the same field, not a live moving scene,
 * since a uniform grid has no concept of foveation to animate).
 */

import { motion } from "motion/react"
import type { DemoFrame } from "../lib/mockData"
import { generateDenseBaselineFrame } from "../lib/mockData"
import { motionTokens } from "../lib/theme"
import { Scene } from "./Scene"

function PanelLabel({ title, mb, cells, accent }: { title: string; mb: number; cells: string; accent: string }) {
  return (
    <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none">
      <div className="rounded-lg border border-white/10 bg-black/50 backdrop-blur px-3 py-1.5">
        <div className="text-[11px] uppercase tracking-widest" style={{ color: accent }}>
          {title}
        </div>
      </div>
      <div className="rounded-lg border border-white/10 bg-black/50 backdrop-blur px-3 py-1.5 text-right font-mono-tech">
        <div className="text-sm text-slate-100">{mb.toFixed(2)} MB</div>
        <div className="text-[10px] text-slate-500">{cells} cells</div>
      </div>
    </div>
  )
}

export function SplitScreen({ frame }: { frame: DemoFrame }) {
  const dense = generateDenseBaselineFrame()
  const ratio = dense.hud.mbUsed / frame.hud.mbUsed

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: motionTokens.duration.normal, ease: motionTokens.easing.smooth }}
      className="relative w-full h-full grid grid-cols-2 gap-[2px] bg-white/10"
    >
      <div className="relative bg-[#05070c]">
        <Scene frame={dense} />
        <PanelLabel title="Uniform 5cm grid (naive baseline)" mb={dense.hud.mbUsed} cells="16,777,216" accent="#94a3b8" />
      </div>
      <div className="relative bg-[#05070c]">
        <Scene frame={frame} />
        <PanelLabel title="DRISHTI foveated clipmap" mb={frame.hud.mbUsed} cells="1,048,576" accent="#4fd1ff" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15, duration: motionTokens.duration.normal }}
        className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full border border-cyan-400/30 bg-black/60 backdrop-blur px-4 py-1.5 font-mono-tech text-sm text-cyan-300"
      >
        {ratio.toFixed(1)}x smaller
      </motion.div>
    </motion.div>
  )
}
