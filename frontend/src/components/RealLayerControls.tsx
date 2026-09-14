import { useDashboardStore } from "../state/store"
import { CLASS_COLOR } from "../lib/theme"

const LEGEND_ITEMS = [
  { label: "Drivable", color: CLASS_COLOR[1] },
  { label: "Caution", color: CLASS_COLOR[2] },
  { label: "Static", color: CLASS_COLOR[4] },
  { label: "Dynamic", color: CLASS_COLOR[6] },
  { label: "Lethal", color: CLASS_COLOR[8] },
]

function LayerButton({ active, label, detail, onClick }: { active: boolean; label: string; detail: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`flex min-h-11 w-full items-center justify-between gap-3 rounded-lg border px-3 py-2 text-left transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300 ${
        active ? "border-cyan-300/35 bg-cyan-300/[0.09]" : "border-white/[0.08] bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.05]"
      }`}
    >
      <span>
        <span className="block text-xs font-medium text-slate-200">{label}</span>
        <span className="block text-[10px] leading-4 text-slate-500">{detail}</span>
      </span>
      <span className={`font-mono-tech text-[10px] uppercase tracking-wide ${active ? "text-cyan-200" : "text-slate-500"}`}>{active ? "ON" : "OFF"}</span>
    </button>
  )
}

export function RealLayerControls() {
  const showRealTerrain = useDashboardStore((s) => s.showRealTerrain)
  const showRealPoints = useDashboardStore((s) => s.showRealPoints)
  const showRealDetections = useDashboardStore((s) => s.showRealDetections)
  const showResolutionGrid = useDashboardStore((s) => s.showResolutionGrid)
  const realUseAccumulated = useDashboardStore((s) => s.realUseAccumulated)
  const toggleRealTerrain = useDashboardStore((s) => s.toggleRealTerrain)
  const toggleRealPoints = useDashboardStore((s) => s.toggleRealPoints)
  const toggleRealDetections = useDashboardStore((s) => s.toggleRealDetections)
  const toggleResolutionGrid = useDashboardStore((s) => s.toggleResolutionGrid)
  const toggleRealAccumulated = useDashboardStore((s) => s.toggleRealAccumulated)

  return (
    <section className="rounded-xl border border-white/10 bg-white/[0.035] p-3">
      <div className="mb-2">
        <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-300/85">Scene layers</p>
        <p className="mt-1 text-xs leading-5 text-slate-400">Control the evidence shown in the real export.</p>
      </div>
      <div className="space-y-2">
        <LayerButton active={realUseAccumulated} label="Temporal map memory" detail="Accumulated cells instead of the current sweep" onClick={toggleRealAccumulated} />
        <LayerButton active={showRealTerrain} label="Semantic elevation" detail="Class-coloured 2.5D cells" onClick={toggleRealTerrain} />
        <LayerButton active={showRealPoints} label="LiDAR returns" detail="Per-point semantic predictions" onClick={toggleRealPoints} />
        <LayerButton active={showRealDetections} label="Object instances" detail="Geometric wireframe detections" onClick={toggleRealDetections} />
        <LayerButton active={showResolutionGrid} label="Resolution schedule" detail="GPU-rendered radial cell spacing" onClick={toggleResolutionGrid} />
      </div>
      <div className="mt-3 border-t border-white/[0.08] pt-3">
        <p className="text-[10px] font-medium uppercase tracking-[0.13em] text-slate-500">Semantic legend</p>
        <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1.5">
          {LEGEND_ITEMS.map((item) => (
            <span key={item.label} className="inline-flex items-center gap-1.5 text-[10px] text-slate-400">
              <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: item.color }} aria-hidden="true" />
              {item.label}
            </span>
          ))}
        </div>
      </div>
    </section>
  )
}
