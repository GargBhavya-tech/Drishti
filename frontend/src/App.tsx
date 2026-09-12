/**
 * App.tsx -- the dashboard shell: header, main 3D view (or split-
 * screen), HUD/overlay/speed-gauge sidebar, and the bottom control
 * strip (timeline, gamma slider, split-screen toggle).
 */

import { AnimatePresence, motion } from "motion/react"
import { CompareWipeToggle, GammaSlider, RealDataToggle, SplitScreenToggle, Timeline } from "./components/Controls"
import { ComparisonWipe } from "./components/ComparisonWipe"
import { GovernorPanel } from "./components/GovernorPanel"
import { HUD } from "./components/HUD"
import { Legend } from "./components/Legend"
import { MissionStatus } from "./components/MissionStatus"
import { RealScene } from "./components/RealScene"
import { Scene } from "./components/Scene"
import { SplitScreen } from "./components/SplitScreen"
import { SystemStatus } from "./components/SystemStatus"
import { VehicleStatus } from "./components/VehicleStatus"
import { DEMO_SEQUENCE } from "./lib/mockData"
import { motionTokens } from "./lib/theme"
import { useDashboardStore } from "./state/store"

/** Header carries Level 1 of the information hierarchy -- the mission
 * status is the single most prominent thing on screen, not the
 * wordmark or the "synthetic demo" disclosure (both still present, just
 * visually subordinate). See DRISHTI_Build_Map.md's own framing and the
 * mission-control redesign brief's layout spec. */
function Header({ frame }: { frame: (typeof DEMO_SEQUENCE)[number] }) {
  return (
    <div className="flex items-center justify-between gap-6 px-5 py-3 bg-white border-b border-[#D9E2EC]">
      <div className="flex items-center gap-2.5 flex-none">
        <div className="h-2 w-2 rounded-full bg-[#087E8B]" />
        <span className="text-sm font-semibold tracking-wide text-[#17324D]">DRISHTI</span>
      </div>
      <div className="flex-1 min-w-0">
        <MissionStatus frame={frame} />
      </div>
      <div className="flex flex-col items-end gap-0.5 flex-none">
        <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-widest text-[#087E8B] font-medium">
          <span className="h-1.5 w-1.5 rounded-full bg-[#087E8B]" />
          Operational
        </div>
        <div className="text-[10px] text-[#52606D] font-mono-tech">synthetic demo sequence</div>
      </div>
    </div>
  )
}

export default function App() {
  const frameIndex = useDashboardStore((s) => s.frameIndex)
  const splitScreen = useDashboardStore((s) => s.splitScreen)
  const compareWipe = useDashboardStore((s) => s.compareWipe)
  const realDataMode = useDashboardStore((s) => s.realDataMode)
  const frame = DEMO_SEQUENCE[frameIndex]

  return (
    <div className="h-screen w-screen flex flex-col bg-[#F3F5F6] text-[#1F2933]">
      <Header frame={frame} />

      <div className="flex-1 flex min-h-0 gap-3 p-3">
        <div className="relative flex-1 min-w-0 rounded-lg border border-[#D9E2EC] overflow-hidden">
          <AnimatePresence mode="wait">
            {realDataMode ? null : compareWipe ? (
              <motion.div
                key="wipe"
                className="absolute inset-0"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: motionTokens.duration.fast }}
              >
                <ComparisonWipe frame={frame} />
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
          {realDataMode && (
            <div className="absolute inset-0">
              <RealScene />
            </div>
          )}
        </div>

        <div className="w-72 shrink-0 overflow-y-auto flex flex-col gap-3">
          <Legend />
          <VehicleStatus frame={frame} />
          <SystemStatus frame={frame} />
          <GovernorPanel frame={frame} />
          <HUD hud={frame.hud} frameIndex={frameIndex} frameCount={DEMO_SEQUENCE.length} />
        </div>
      </div>

      <div className="bg-white border-t border-[#D9E2EC] px-5 py-3 flex items-center gap-6">
        <Timeline frameCount={DEMO_SEQUENCE.length} />
        <div className="w-px h-6 bg-[#D9E2EC]" />
        <GammaSlider />
        <div className="w-px h-6 bg-[#D9E2EC]" />
        <SplitScreenToggle />
        <CompareWipeToggle />
        <RealDataToggle />
      </div>
    </div>
  )
}
