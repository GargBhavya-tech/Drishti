import type { RealSceneStatus } from "./RealScene"
import { useDashboardStore } from "../state/store"

function StatusChip({ label, value, tone = "neutral" }: { label: string; value: string; tone?: "neutral" | "live" | "warning" }) {
  const toneClass = {
    neutral: "border-white/10 bg-white/[0.035] text-slate-300",
    live: "border-cyan-300/25 bg-cyan-300/[0.08] text-cyan-100",
    warning: "border-amber-300/25 bg-amber-300/[0.08] text-amber-100",
  }[tone]

  return (
    <div className={`hidden items-center gap-2 rounded-md border px-2.5 py-1 lg:flex ${toneClass}`}>
      <span className="text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">{label}</span>
      <span className="font-mono-tech text-[11px] tabular-nums">{value}</span>
    </div>
  )
}

function ModeSwitch() {
  const realDataMode = useDashboardStore((s) => s.realDataMode)
  const toggleRealDataMode = useDashboardStore((s) => s.toggleRealDataMode)

  return (
    <button
      type="button"
      onClick={toggleRealDataMode}
      aria-pressed={realDataMode}
      className="inline-flex min-h-11 items-center gap-2 rounded-lg border border-white/10 bg-white/[0.035] px-3 text-left transition-colors hover:border-cyan-300/35 hover:bg-cyan-300/[0.08] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300"
    >
      <span className={`h-2 w-2 rounded-full ${realDataMode ? "bg-cyan-300 shadow-[0_0_10px_rgba(79,209,255,0.8)]" : "bg-slate-500"}`} />
      <span>
        <span className="block text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">Source</span>
        <span className="block font-mono-tech text-[11px] text-slate-200">{realDataMode ? "REAL EXPORT" : "SYNTHETIC DEMO"}</span>
      </span>
    </button>
  )
}

function checkpointName(path: string): string {
  const normalized = path.replaceAll("\\", "/")
  const parent = normalized.split("/").slice(-2, -1)[0]
  return parent ? `${parent}/best.pt` : normalized
}

export function RunStatusBar({ realStatus }: { realStatus: RealSceneStatus | null }) {
  const realDataMode = useDashboardStore((s) => s.realDataMode)

  const sourceDetail = realDataMode && realStatus ? realStatus.manifest.sequenceDir.replace("data/", "") : "frontend generated"
  const frameValue = realDataMode && realStatus ? `${realStatus.frameIndex + 1}/${realStatus.manifest.nFrames}` : "48-frame loop"
  const modelValue = realDataMode && realStatus ? checkpointName(realStatus.manifest.checkpointPath) : "demo planner"

  return (
    <header className="z-20 flex min-h-[68px] shrink-0 items-center justify-between gap-3 border-b border-white/10 bg-[#09101b]/95 px-3 py-2 backdrop-blur-xl sm:px-5">
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-cyan-300/30 bg-cyan-300/[0.09]" aria-hidden="true">
          <svg viewBox="0 0 24 24" className="h-5 w-5 text-cyan-200" fill="none" stroke="currentColor" strokeWidth="1.7">
            <path d="M4 16.5c2.2-6.3 5.1-9.5 8-9.5s5.8 3.2 8 9.5" />
            <path d="M7.5 16.5c1.2-3.3 2.7-5 4.5-5s3.3 1.7 4.5 5" />
            <path d="M2.5 18.5h19" />
          </svg>
        </div>
        <div className="min-w-0">
          <h1 className="font-mono-tech text-sm font-semibold tracking-[0.22em] text-slate-100">DRISHTI</h1>
          <p className="truncate text-[11px] text-slate-500">Adaptive 2.5D LiDAR perception console</p>
        </div>
      </div>

      <div className="hidden min-w-0 flex-1 items-center justify-end gap-2 xl:flex">
        <StatusChip label="Run" value={sourceDetail} tone={realDataMode ? "live" : "neutral"} />
        <StatusChip label="Model" value={modelValue} tone={realDataMode ? "live" : "neutral"} />
        <StatusChip label="Frame" value={frameValue} />
        {realDataMode && realStatus && (
          <StatusChip label="Map" value={realStatus.useAccumulated ? "accumulated memory" : "single sweep"} tone={realStatus.useAccumulated ? "warning" : "neutral"} />
        )}
      </div>

      <ModeSwitch />
    </header>
  )
}
