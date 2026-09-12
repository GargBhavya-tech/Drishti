/**
 * MapOverlay.tsx -- the GIS-viewer chrome that sits directly on top of
 * the 3D viewport: a compass rose and a distance scale bar. Both are
 * static reference elements (like the printed compass/scale on a paper
 * map), not live sensor readouts -- they are not presented as measured
 * telemetry anywhere in the UI, so there is nothing here to fabricate.
 * A live-recalibrated scale bar (tracking the orbit camera's actual
 * zoom) would need to thread the camera's projection out of the R3F
 * canvas into this DOM overlay; deliberately left as a fixed reference
 * for now rather than something that could quietly go stale.
 */

function CompassRose() {
  return (
    <div className="flex flex-col items-center gap-1 pointer-events-none select-none">
      <svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="white" fillOpacity="0.9" stroke="#D9E2EC" strokeWidth="1.5" />
        <path d="M20 6 L24 20 L20 18 L16 20 Z" fill="#17324D" />
        <path d="M20 34 L16 20 L20 22 L24 20 Z" fill="#9AA5AB" />
      </svg>
      <span className="text-[10px] font-semibold text-[#17324D] tracking-wide">N</span>
    </div>
  )
}

function ScaleBar() {
  // A representative reference scale, not a live-recalibrated readout --
  // see this file's own doc comment.
  const ticks = [0, 10, 20, 30]
  return (
    <div className="flex flex-col items-center pointer-events-none select-none">
      <div className="flex items-end h-2">
        {ticks.slice(0, -1).map((t, idx) => (
          <div
            key={t}
            className="h-2 border-l border-t border-r border-[#52606D]"
            style={{ width: 28, borderTopWidth: 1.5, background: idx % 2 === 0 ? "#52606D" : "transparent" }}
          />
        ))}
      </div>
      <div className="flex text-[10px] text-[#52606D] font-mono-tech" style={{ width: 28 * (ticks.length - 1) }}>
        {ticks.map((t, idx) => (
          <span key={t} className="flex-1" style={{ textAlign: idx === 0 ? "left" : idx === ticks.length - 1 ? "right" : "center" }}>
            {t}
          </span>
        ))}
      </div>
      <span className="text-[9px] text-[#9AA5AB] mt-0.5">metres</span>
    </div>
  )
}

export function MapOverlay() {
  return (
    <div className="absolute bottom-3 right-3 flex items-end gap-4 bg-white/85 backdrop-blur-sm rounded-lg border border-[#D9E2EC] px-3 py-2">
      <ScaleBar />
      <div className="w-px self-stretch bg-[#D9E2EC]" />
      <CompassRose />
    </div>
  )
}
