/**
 * App.tsx -- the dashboard shell. Header carries branding + system
 * status; the 3D viewport carries its own floating overlays (mission
 * status, terrain key, compass/scale) directly on the map, GIS-viewer
 * style, rather than in surrounding chrome; the sidebar is vehicle/
 * hazard/system state; the bottom bar is view-layer tabs + the replay
 * transport.
 */

import { AnimatePresence, motion } from "motion/react"
import { useEffect, useState } from "react"
import { CompareWipeToggle, GammaSlider, SplitScreenToggle, Timeline, ViewLayerTabs } from "./components/Controls"
import { ComparisonWipe } from "./components/ComparisonWipe"
import { CurrentHazards } from "./components/CurrentHazards"
import { GovernorPanel } from "./components/GovernorPanel"
import { HUD } from "./components/HUD"
import { Legend } from "./components/Legend"
import { MapOverlay } from "./components/MapOverlay"
import { MissionStatus } from "./components/MissionStatus"
import { RealScene } from "./components/RealScene"
import { Scene } from "./components/Scene"
import { SplitScreen } from "./components/SplitScreen"
import { SystemStatus } from "./components/SystemStatus"
import { VehicleStatus } from "./components/VehicleStatus"
import { DEMO_SEQUENCE } from "./lib/mockData"
import { motionTokens } from "./lib/theme"
import { useDashboardStore } from "./state/store"

/** A generic compass-rose emblem -- deliberately NOT a reproduction of
 * any real organization's insignia (this is a hackathon submission
 * addressing a sponsor's problem statement, not an official product of
 * that sponsor). */
function OrgEmblem() {
  return (
    <svg width="34" height="34" viewBox="0 0 34 34" className="flex-none">
      <circle cx="17" cy="17" r="15.5" fill="none" stroke="#17324D" strokeWidth="1.4" />
      <circle cx="17" cy="17" r="11.5" fill="none" stroke="#17324D" strokeWidth="0.8" />
      <path d="M17 6 L20 17 L17 15 L14 17 Z" fill="#17324D" />
      <path d="M17 28 L14 17 L17 19 L20 17 Z" fill="#087E8B" />
      <circle cx="17" cy="17" r="1.6" fill="#17324D" />
    </svg>
  )
}

function useLiveClock() {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  return now
}

function Header() {
  const now = useLiveClock()
  const dateStr = now.toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" })
  const timeStr = now.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })

  return (
    <div className="flex items-center justify-between gap-6 px-5 py-2.5 bg-white border-b border-[#D9E2EC]">
      <div className="flex items-center gap-3 flex-none">
        <OrgEmblem />
        <div className="leading-tight">
          <div className="text-[13px] font-bold text-[#17324D] tracking-wide">DRDO</div>
          <div className="text-[10px] text-[#52606D]">Science for a Safer Tomorrow</div>
        </div>
      </div>

      <div className="flex-1 text-center">
        <div className="text-[22px] font-bold text-[#17324D] tracking-wide leading-tight">DRISHTI</div>
        <div className="text-[12px] text-[#52606D]">LiDAR Terrain Mapping &amp; Safe Navigation System</div>
      </div>

      <div className="flex flex-col items-end gap-0.5 flex-none">
        <div className="flex items-center gap-1.5 text-[13px] font-semibold text-[#0F9D58]">
          <span className="h-2 w-2 rounded-full bg-[#0F9D58]" />
          System operational
        </div>
        <div className="text-[10px] text-[#52606D]">Replaying recorded mapping run</div>
        <div className="text-[10px] font-mono-tech text-[#9AA5AB]">
          {dateStr} &middot; {timeStr}
        </div>
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
  const frameCount = DEMO_SEQUENCE.length

  return (
    <div className="h-screen w-screen flex flex-col bg-[#F3F5F6] text-[#1F2933]">
      <Header />

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

          {/* GIS-viewer chrome floating directly on the map, not in
              surrounding page chrome -- mission status top-left, terrain
              key bottom-left, compass/scale bottom-right. Hidden in
              split/compare modes, which have their own overlay labels. */}
          {!splitScreen && !compareWipe && !realDataMode && (
            <div className="absolute inset-0 pointer-events-none p-3 flex flex-col justify-between">
              <div className="pointer-events-auto self-start">
                <MissionStatus frame={frame} />
              </div>
              <div className="flex items-end justify-between">
                <Legend />
                <MapOverlay />
              </div>
            </div>
          )}
        </div>

        <div className="w-72 shrink-0 overflow-y-auto flex flex-col gap-3">
          <VehicleStatus frame={frame} frameIndex={frameIndex} frameCount={frameCount} />
          <SystemStatus frame={frame} />
          <CurrentHazards frame={frame} frameIndex={frameIndex} frameCount={frameCount} />
          <GovernorPanel frame={frame} />
          <HUD hud={frame.hud} frameIndex={frameIndex} frameCount={frameCount} />
        </div>
      </div>

      <div className="bg-white border-t border-[#D9E2EC] px-4 py-2.5 flex items-center gap-4">
        <ViewLayerTabs />
        <div className="w-px h-7 bg-[#D9E2EC]" />
        <Timeline frameCount={frameCount} />
        <div className="w-px h-7 bg-[#D9E2EC]" />
        <SplitScreenToggle />
        <CompareWipeToggle />
        <div className="w-px h-7 bg-[#D9E2EC]" />
        <GammaSlider />
      </div>
    </div>
  )
}
