import { AnimatePresence, motion } from "motion/react"
import { lazy, Suspense, useCallback, useEffect, useState } from "react"
import { CompareWipeToggle, EvidenceToggle, GammaSlider, RealTimeline, SplitScreenToggle, Timeline } from "./components/Controls"
import { DecisionStack } from "./components/DecisionStack"
import { EvidenceDrawer } from "./components/EvidenceDrawer"
import { GovernorPanel } from "./components/GovernorPanel"
import { HUD } from "./components/HUD"
import { RealLayerControls } from "./components/RealLayerControls"
import type { RealSceneStatus } from "./components/RealScene"
import { RunStatusBar } from "./components/RunStatusBar"
import { Scene } from "./components/Scene"
import { SceneToolbar } from "./components/SceneToolbar"
import { SpeedGauge } from "./components/SpeedGauge"
import { DEMO_SEQUENCE } from "./lib/mockData"
import { preloadAllFrames } from "./lib/realData"
import { motionTokens } from "./lib/theme"
import { useDashboardStore } from "./state/store"

// Code-split: RealScene/ComparisonWipe/SplitScreen are never all needed
// on first paint (default view is plain synthetic Scene) -- each is its
// own chunk, fetched only once its mode is actually selected, instead of
// bundled unconditionally into the initial JS payload.
const RealScene = lazy(() => import("./components/RealScene").then((m) => ({ default: m.RealScene })))
const ComparisonWipe = lazy(() => import("./components/ComparisonWipe").then((m) => ({ default: m.ComparisonWipe })))
const SplitScreen = lazy(() => import("./components/SplitScreen").then((m) => ({ default: m.SplitScreen })))

function ScenePaneFallback() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-[#07101a]">
      <p className="font-mono-tech text-xs text-slate-400">Loading view…</p>
    </div>
  )
}

function SyntheticControls({ frameCount }: { frameCount: number }) {
  return (
    <>
      <Timeline frameCount={frameCount} />
      <div className="hidden h-7 w-px bg-white/10 md:block" />
      <div className="hidden xl:block">
        <GammaSlider />
      </div>
      <div className="hidden h-7 w-px bg-white/10 md:block" />
      <div className="flex shrink-0 items-center gap-2">
        <SplitScreenToggle />
        <CompareWipeToggle />
      </div>
    </>
  )
}

export default function App() {
  const frameIndex = useDashboardStore((s) => s.frameIndex)
  const splitScreen = useDashboardStore((s) => s.splitScreen)
  const compareWipe = useDashboardStore((s) => s.compareWipe)
  const realDataMode = useDashboardStore((s) => s.realDataMode)
  const [realStatus, setRealStatus] = useState<RealSceneStatus | null>(null)
  const frame = DEMO_SEQUENCE[frameIndex]

  const handleRealStatus = useCallback((status: RealSceneStatus | null) => {
    setRealStatus(status)
  }, [])

  // Warm the frame cache in the background once real-export mode is
  // actually selected -- not on app start (that would compete with the
  // initial synthetic-mode paint for bandwidth for a mode most sessions
  // may never open), and not blocking: loadFrame's own in-memory cache
  // (realData.ts) means later scrubbing hits this cache instead of the
  // network, without this call gating anything on screen.
  useEffect(() => {
    if (!realDataMode) return
    preloadAllFrames().catch(() => {
      /* best-effort warmup; a real per-frame load still happens on demand */
    })
  }, [realDataMode])

  return (
    <div className="relative flex h-dvh w-screen flex-col overflow-hidden bg-[#07101a] text-slate-200">
      <a className="skip-link" href="#perception-map">Skip to perception map</a>
      <RunStatusBar realStatus={realStatus} />

      <main className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <section id="perception-map" aria-label="3D LiDAR perception map" className="relative min-h-[48dvh] min-w-0 flex-1 bg-[#07101a] lg:min-h-0">
          <div className="pointer-events-none absolute right-3 top-3 z-10">
            <SceneToolbar />
          </div>
          <AnimatePresence mode="wait">
            {realDataMode ? (
              <motion.div
                key="real-export"
                className="absolute inset-0"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: motionTokens.duration.fast }}
              >
                <Suspense fallback={<ScenePaneFallback />}>
                  <RealScene onStatusChange={handleRealStatus} />
                </Suspense>
              </motion.div>
            ) : compareWipe ? (
              <motion.div
                key="wipe"
                className="absolute inset-0"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: motionTokens.duration.fast }}
              >
                <Suspense fallback={<ScenePaneFallback />}>
                  <ComparisonWipe frame={frame} />
                </Suspense>
              </motion.div>
            ) : splitScreen ? (
              <motion.div
                key="split"
                className="absolute inset-0"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: motionTokens.duration.fast }}
              >
                <Suspense fallback={<ScenePaneFallback />}>
                  <SplitScreen frame={frame} />
                </Suspense>
              </motion.div>
            ) : (
              <motion.div
                key="synthetic"
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
        </section>

        <aside aria-label="Operational evidence and controls" className="flex w-full shrink-0 flex-col gap-3 overflow-y-auto border-t border-white/10 bg-[#09101b]/88 p-3 lg:w-[20rem] lg:border-l lg:border-t-0">
          <DecisionStack frame={frame} realStatus={realStatus} realDataMode={realDataMode} />
          {realDataMode ? (
            <RealLayerControls />
          ) : (
            <>
              <HUD hud={frame.hud} frameIndex={frameIndex} frameCount={DEMO_SEQUENCE.length} />
              <SpeedGauge envelope={frame.speedEnvelope} />
              <GovernorPanel frame={frame} />
            </>
          )}
        </aside>
      </main>

      <footer className="z-20 flex shrink-0 items-center gap-3 border-t border-white/10 bg-[#09101b]/95 px-3 py-2 backdrop-blur-xl sm:px-5">
        {realDataMode ? <RealTimeline /> : <SyntheticControls frameCount={DEMO_SEQUENCE.length} />}
        <EvidenceToggle />
      </footer>

      <EvidenceDrawer demoFrame={frame} realStatus={realStatus} realDataMode={realDataMode} />
    </div>
  )
}
