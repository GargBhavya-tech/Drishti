import { useDashboardStore, type CameraPreset } from "../state/store"

const PRESETS: Array<{ id: CameraPreset; label: string; hint: string }> = [
  { id: "driver", label: "Driver", hint: "low forward view" },
  { id: "tactical", label: "Tactical", hint: "top-down context" },
  { id: "hazard", label: "Hazard", hint: "near-field focus" },
]

export function SceneToolbar() {
  const cameraPreset = useDashboardStore((s) => s.cameraPreset)
  const setCameraPreset = useDashboardStore((s) => s.setCameraPreset)

  return (
    <div className="pointer-events-auto flex items-center gap-1 rounded-xl border border-white/10 bg-[#09101b]/85 p-1.5 shadow-lg backdrop-blur-md" aria-label="Scene camera presets">
      {PRESETS.map((preset) => {
        const active = cameraPreset === preset.id
        return (
          <button
            type="button"
            key={preset.id}
            onClick={() => setCameraPreset(preset.id)}
            aria-pressed={active}
            title={preset.hint}
            className={`min-h-9 rounded-lg px-2.5 text-[10px] font-medium uppercase tracking-[0.12em] transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300 ${
              active ? "bg-cyan-300/[0.16] text-cyan-100" : "text-slate-400 hover:bg-white/[0.07] hover:text-slate-200"
            }`}
          >
            {preset.label}
          </button>
        )
      })}
    </div>
  )
}
