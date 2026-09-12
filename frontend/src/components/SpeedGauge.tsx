/**
 * SpeedGauge.tsx -- Ticket #54: a speedometer-style gauge with a red
 * zone driven by v_max, the binding hazard named beside it. The
 * needle springs to its new angle (communicates the CHANGE in safe
 * speed, e.g. when driving into an occluded region) rather than
 * snapping.
 */

import { motion, useReducedMotion, useSpring, useTransform } from "motion/react"
import { useEffect } from "react"
import type { SpeedEnvelopeState } from "../lib/mockData"
import { motionTokens } from "../lib/theme"

const MAX_SPEED_KMH = 60
const START_ANGLE = -120
const END_ANGLE = 120

function angleForSpeed(kmh: number): number {
  const t = Math.min(1, Math.max(0, kmh / MAX_SPEED_KMH))
  return START_ANGLE + t * (END_ANGLE - START_ANGLE)
}

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

function arcPath(cx: number, cy: number, r: number, startDeg: number, endDeg: number): string {
  const start = polarToCartesian(cx, cy, r, endDeg)
  const end = polarToCartesian(cx, cy, r, startDeg)
  const largeArc = endDeg - startDeg <= 180 ? 0 : 1
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`
}

export function SpeedGauge({ envelope }: { envelope: SpeedEnvelopeState }) {
  const reduce = useReducedMotion()
  const isOccluded = envelope.bindingHazard === "occluded_corridor"

  const angle = useSpring(angleForSpeed(envelope.vMaxKmh), {
    stiffness: reduce ? 1000 : 90,
    damping: 18,
  })
  useEffect(() => {
    angle.set(angleForSpeed(envelope.vMaxKmh))
  }, [envelope.vMaxKmh, angle])

  const needleTransform = useTransform(angle, (a) => `rotate(${a}deg)`)

  const cx = 90
  const cy = 90
  const r = 66
  const redZoneStart = angleForSpeed(25)

  return (
    <motion.div
      className="panel-glass p-3 flex flex-col items-center"
      initial={{ opacity: 0, y: motionTokens.distance.sm }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: motionTokens.duration.normal, ease: motionTokens.easing.smooth }}
    >
      <div className="text-[11px] uppercase tracking-widest text-[#96A3A8] self-start mb-1 font-medium">Speed envelope</div>
      <svg width={180} height={130} viewBox="0 0 180 130">
        <path d={arcPath(cx, cy, r, START_ANGLE, END_ANGLE)} stroke="#1c2430" strokeWidth={10} fill="none" strokeLinecap="round" />
        <path
          d={arcPath(cx, cy, r, redZoneStart, END_ANGLE)}
          stroke="#E05245"
          strokeOpacity={0.55}
          strokeWidth={10}
          fill="none"
          strokeLinecap="round"
        />
        <path
          d={arcPath(cx, cy, r, START_ANGLE, angleForSpeed(envelope.vMaxKmh))}
          stroke={isOccluded ? "#ffb84f" : "#55D6E8"}
          strokeWidth={5}
          fill="none"
          strokeLinecap="round"
        />
        <g style={{ transformOrigin: `${cx}px ${cy}px` }}>
          <motion.line
            x1={cx}
            y1={cy}
            x2={cx}
            y2={cy - r + 14}
            stroke="#f1f5f9"
            strokeWidth={2.5}
            strokeLinecap="round"
            style={{ transformOrigin: `${cx}px ${cy}px`, rotate: needleTransform }}
          />
        </g>
        <circle cx={cx} cy={cy} r={4} fill="#f1f5f9" />
      </svg>
      <motion.div
        key={Math.round(envelope.vMaxKmh)}
        initial={{ opacity: 0.4, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: motionTokens.duration.fast }}
        className="font-mono-tech text-2xl text-slate-100 -mt-2"
      >
        {envelope.vMaxKmh.toFixed(1)}
        <span className="text-sm text-slate-500 ml-1">km/h</span>
      </motion.div>
      <motion.div
        animate={{ color: isOccluded ? "#ffb84f" : "#64748b" }}
        className="text-[11px] uppercase tracking-wide mt-1"
      >
        binding: {envelope.bindingHazard} ({envelope.bindingRangeM.toFixed(1)}m)
      </motion.div>
    </motion.div>
  )
}
