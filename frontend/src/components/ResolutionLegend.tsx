import type { LevelInfo } from "../lib/realData"

export function ResolutionLegend({ levels }: { levels: LevelInfo[] }) {
  return (
    <div className="rounded-xl border border-white/10 bg-[#09101b]/80 px-3 py-2.5 shadow-lg backdrop-blur-md">
      <p className="text-[9px] font-semibold uppercase tracking-[0.16em] text-cyan-300/85">Radial resolution schedule</p>
      <div className="mt-2 space-y-1.5">
        {levels.map((level) => (
          <div key={level.level} className="flex items-center justify-between gap-4 font-mono-tech text-[10px] tabular-nums">
            <span className="text-slate-200">L{level.level} · {(level.cellSizeM * 100).toFixed(0)} cm</span>
            <span className="text-slate-500">≤ {level.nyquistRadiusM.toFixed(1)} m</span>
          </div>
        ))}
      </div>
    </div>
  )
}
