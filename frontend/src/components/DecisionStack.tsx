import { useMemo } from "react"
import type { DemoFrame } from "../lib/mockData"
import { DRISHTI_CLASS_NAMES } from "../lib/theme"
import type { RealSceneStatus } from "./RealScene"

function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(Math.round(value))
}

function PanelHeading({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return (
    <div className="mb-3">
      <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-300/85">{eyebrow}</p>
      <h2 className="mt-1 text-sm font-semibold text-slate-100">{title}</h2>
      <p className="mt-1 text-xs leading-5 text-slate-400">{description}</p>
    </div>
  )
}

function Row({ label, value, valueClass = "text-slate-200" }: { label: string; value: string; valueClass?: string }) {
  return (
    <div className="flex items-start justify-between gap-3 border-t border-white/[0.07] py-2.5 first:border-t-0 first:pt-0">
      <span className="text-[10px] font-medium uppercase tracking-[0.13em] text-slate-500">{label}</span>
      <span className={`max-w-[62%] text-right font-mono-tech text-xs leading-5 tabular-nums ${valueClass}`}>{value}</span>
    </div>
  )
}

function Card({ children }: { children: React.ReactNode }) {
  return <section className="rounded-xl border border-white/10 bg-white/[0.035] p-3 shadow-[0_12px_28px_rgba(0,0,0,0.12)]">{children}</section>
}

function DemoDecisionStack({ frame }: { frame: DemoFrame }) {
  const caution = frame.speedEnvelope.bindingHazard === "occluded_corridor"

  return (
    <div className="flex flex-col gap-3">
      <Card>
        <PanelHeading
          eyebrow="Vehicle decision"
          title={caution ? "Reduce speed and preserve braking margin" : "Route is traversable within the current envelope"}
          description="Synthetic planner and speed-governor state. It is not part of the real-data export."
        />
        <Row label="Permitted speed" value={`${frame.speedEnvelope.vMaxKmh.toFixed(1)} km/h`} valueClass={caution ? "text-amber-200" : "text-cyan-100"} />
        <Row label="Binding condition" value={frame.speedEnvelope.bindingHazard.replaceAll("_", " ")} valueClass={caution ? "text-amber-200" : "text-slate-200"} />
        <Row label="Hazard range" value={`${frame.speedEnvelope.bindingRangeM.toFixed(1)} m`} />
      </Card>

      <Card>
        <PanelHeading eyebrow="Map evidence" title="Foveated 2.5D clipmap" description="Semantic cells, visibility state, and local path context update through the synthetic scenario." />
        <Row label="Occupied cells" value={formatNumber(frame.cells.length)} />
        <Row label="Map memory" value={`${frame.hud.mbUsed.toFixed(2)} MB`} />
        <Row label="Latency P95" value={`${frame.hud.latencyP95Ms.toFixed(1)} ms`} />
      </Card>
    </div>
  )
}

function detectionSummary(status: RealSceneStatus): Array<{ className: string; count: number }> {
  const counts = new Map<number, number>()
  for (let i = 0; i < status.frame.detectionCount; i++) {
    const classId = Math.round(status.frame.detections[i * 6 + 3])
    counts.set(classId, (counts.get(classId) ?? 0) + 1)
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([classId, count]) => ({ className: DRISHTI_CLASS_NAMES[classId] ?? "UNKNOWN", count }))
}

function RealDecisionStack({ status }: { status: RealSceneStatus | null }) {
  const detections = useMemo(() => (status ? detectionSummary(status) : []), [status])

  if (!status) {
    return (
      <Card>
        <PanelHeading eyebrow="Run evidence" title="Loading exported run" description="The dashboard will show only metadata and measurements included in the local export." />
        <div className="h-2 animate-pulse rounded-full bg-white/[0.08]" />
        <div className="mt-3 h-2 w-4/5 animate-pulse rounded-full bg-white/[0.06]" />
      </Card>
    )
  }

  const { manifest, frame, frameIndex, useAccumulated } = status
  const sourceFrame = manifest.rellisFrameIndices[frameIndex]
  const displayedCells = useAccumulated ? frame.accumulatedCellCount : frame.cellCount

  return (
    <div className="flex flex-col gap-3">
      <Card>
        <PanelHeading
          eyebrow="Operational scope"
          title="Semantic perception export"
          description="This run exports terrain, grid, and geometric detections. Route and speed decisions are intentionally not inferred by the frontend."
        />
        <Row label="Dataset frame" value={sourceFrame === undefined ? `Export ${frameIndex + 1}` : `RELLIS ${sourceFrame}`} />
        <Row label="Map representation" value={useAccumulated ? "Accumulated memory" : "Single sweep"} valueClass={useAccumulated ? "text-amber-200" : "text-cyan-100"} />
        <Row label="Planner / speed" value="Not exported" valueClass="text-slate-400" />
      </Card>

      <Card>
        <PanelHeading eyebrow="Observed now" title={`${frame.detectionCount} geometric detection${frame.detectionCount === 1 ? "" : "s"}`} description="Wireframes in the scene are geometric instances derived from the exported real frame." />
        <Row label="LiDAR returns" value={formatNumber(frame.pointCount)} />
        <Row label="Rendered grid cells" value={formatNumber(displayedCells)} />
        <Row label="Detected objects" value={formatNumber(frame.detectionCount)} valueClass={frame.detectionCount > 0 ? "text-amber-200" : "text-slate-300"} />
        {detections.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1.5" aria-label="Detection class breakdown">
            {detections.slice(0, 4).map((item) => (
              <span key={item.className} className="rounded border border-amber-300/20 bg-amber-300/[0.07] px-2 py-1 font-mono-tech text-[10px] text-amber-100">
                {item.className} {item.count}
              </span>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}

export function DecisionStack({ frame, realStatus, realDataMode }: { frame: DemoFrame; realStatus: RealSceneStatus | null; realDataMode: boolean }) {
  return realDataMode ? <RealDecisionStack status={realStatus} /> : <DemoDecisionStack frame={frame} />
}
