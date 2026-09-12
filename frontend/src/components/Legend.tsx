/**
 * Legend.tsx -- a permanent, human-readable terrain key (mission-control
 * redesign's own "TERRAIN KEY" spec). No "CLASS 0 / CLASS 1" jargon --
 * every swatch is a plain-language label using the SAME colors the
 * terrain mesh and hazard/path renderers actually draw with
 * (lib/theme.ts), so the key can never silently drift out of sync with
 * the 3D view.
 */

import { CLASS_COLOR, HAZARD_COLOR, PATH_COLOR } from "../lib/theme"

const TERRAIN_KEY: { label: string; color: string }[] = [
  { label: "Drivable", color: CLASS_COLOR[1] },
  { label: "Vegetation", color: CLASS_COLOR[5] },
  { label: "Obstacle", color: CLASS_COLOR[4] },
  { label: "Not yet seen", color: CLASS_COLOR[0] },
]

export function Legend() {
  return (
    <div className="panel-glass p-3">
      <div className="text-[11px] uppercase tracking-widest text-[#96A3A8] mb-2 font-medium">Terrain key</div>
      <div className="flex flex-col gap-1.5">
        {TERRAIN_KEY.map((item) => (
          <div key={item.label} className="flex items-center gap-2.5">
            <span className="h-2.5 w-2.5 rounded-sm flex-none" style={{ background: item.color }} />
            <span className="text-[13px] text-[#E7ECEE]">{item.label}</span>
          </div>
        ))}
        <div className="flex items-center gap-2.5 pt-1 mt-1 border-t border-white/[0.06]">
          <span className="text-[13px] flex-none" style={{ color: HAZARD_COLOR }}>
            &#9888;
          </span>
          <span className="text-[13px] text-[#E7ECEE]">Hazard</span>
        </div>
        <div className="flex items-center gap-2.5">
          <span className="h-[2px] w-3 flex-none" style={{ background: PATH_COLOR }} />
          <span className="text-[13px] text-[#E7ECEE]">Safe path</span>
        </div>
      </div>
    </div>
  )
}
