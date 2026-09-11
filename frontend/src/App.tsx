/**
 * App.tsx -- the dashboard shell: header, main 3D view (or split-
 * screen), HUD/overlay/speed-gauge sidebar, and the bottom control
 * strip (timeline, gamma slider, split-screen toggle).
 */

import { AnimatePresence, motion } from "motion/react"
import { GammaSlider, SplitScreenToggle, Timeline } from "./components/Controls"
import { HUD } from "./components/HUD"
import { Scene } from "./components/Scene"
import { SpeedGauge } from "./components/SpeedGauge"
import { SplitScreen } from "./components/SplitScreen"
import { DEMO_SEQUENCE } from "./lib/mockData"
import { motionTokens } from "./lib/theme"
import { useDashboardStore } from "./state/store"

function Header() {
  return (
    <div className="flex items-center justify-between px-5 py-3 border-b border-white/8">
      <div className="flex items-center gap-3">
        <div className="h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_8px_2px_rgba(79,209,255,0.6)]" />
        <span className="font-mono-tech text-sm tracking-[0.3em] text-slate-200">DRISHTI</span>
        <span className="text-xs text-slate-500">adaptive variable-resolution 2.5D LiDAR mapping</span>
      </div>
      <div className="text-[11px] text-slate-600 font-mono-tech">demo sequence -- synthetic, frontend-only</div>
    </div>
  )
}

export default function App() {
  const frameIndex = useDashboardStore((s) => s.frameIndex)
  const splitScreen = useDashboardStore((s) => s.splitScreen)
  const frame = DEMO_SEQUENCE[frameIndex]

  return (
    <div className="h-screen w-screen flex flex-col bg-[#05070c] text-slate-200">
      <Header />

      <div className="flex-1 flex min-h-0">
        <div className="relative flex-1 min-w-0">
          <AnimatePresence mode="wait">
            {splitScreen ? (
              <motion.div
                key="split"
                className="absolute inset-0"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: motionTokens.duration.fast }}
              >
                <SplitScreen frame={frame} />
              </motion.div>
            ) : (
              <motion.div
                key="single"
                className="absolute inset-0"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: motionTokens.duration.fast }}
              >
                <Scene frame={frame} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="w-72 shrink-0 border-l border-white/8 bg-black/20 p-3 overflow-y-auto flex flex-col gap-3">
          <HUD hud={frame.hud} frameIndex={frameIndex} frameCount={DEMO_SEQUENCE.length} />
          <SpeedGauge envelope={frame.speedEnvelope} />
        </div>
      </div>

      <div className="border-t border-white/8 bg-black/30 px-5 py-3 flex items-center gap-6">
        <Timeline frameCount={DEMO_SEQUENCE.length} />
        <div className="w-px h-6 bg-white/10" />
        <GammaSlider />
        <div className="w-px h-6 bg-white/10" />
        <SplitScreenToggle />
      </div>
    </div>
  )
}
