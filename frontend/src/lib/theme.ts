/**
 * theme.ts -- the design system shared by every component: motion
 * tokens (per the motion-ui skill's own spec) and the DRISHTI class
 * color palette.
 *
 * Class colors follow Build Map Ticket #52's own rule: class is
 * CATEGORICAL (hue), height is CONTINUOUS (a separate ramp), confidence
 * is a separate channel (alpha) -- never three variables in one hue
 * ramp. Palette is a muted mission-control system (2026-09 redesign):
 * terrain reads as recessive, desaturated ground; red is reserved
 * exclusively for LETHAL / NEGATIVE_OBSTACLE (matching the same
 * reservation this project's own Python backend makes for LETHAL in
 * planning/conservatism.py); cyan is reserved exclusively for the
 * planned route / active navigation state -- never decorative.
 */

// -- Mission-control surface tokens --------------------------------------
export const SURFACE = {
  appBg: "#080D12",
  panelBg: "#0D141B",
  panelBorder: "rgba(255,255,255,0.08)",
} as const

export const TEXT = {
  primary: "#E7ECEE",
  secondary: "#96A3A8",
  muted: "#657278",
} as const

// The two semantic accents. Never reused decoratively -- see Ticket
// "mission-control redesign"'s own rule: cyan means route/active-nav,
// red means hazard/danger, nothing else borrows either.
export const HAZARD_COLOR = "#E05245"
export const PATH_COLOR = "#55D6E8"

export const GRID_LINE_COLOR = "#536068"

export const motionTokens = {
  duration: {
    fast: 0.18,
    normal: 0.35,
    slow: 0.6,
  },
  easing: {
    smooth: [0.22, 1, 0.36, 1] as [number, number, number, number],
    sharp: [0.4, 0, 0.2, 1] as [number, number, number, number],
  },
  distance: {
    sm: 8,
    md: 16,
    lg: 24,
  },
} as const

export const isLowEndDevice = (): boolean => {
  if (typeof navigator === "undefined") return false
  const nav = navigator as Navigator & { deviceMemory?: number }
  if (nav.deviceMemory !== undefined) return nav.deviceMemory <= 2
  return nav.hardwareConcurrency !== undefined && nav.hardwareConcurrency <= 4
}

// perception.taxonomy.DrishtiClass, mirrored exactly (id -> name) so a
// real backend export's numeric class ids map straight through with
// no translation table drifting out of sync.
export const DRISHTI_CLASS_NAMES = [
  "UNKNOWN",
  "DRIVABLE",
  "CAUTION",
  "NON_TRAVERSABLE",
  "STATIC_OBSTACLE",
  "VEGETATION",
  "VEHICLE",
  "PEDESTRIAN",
  "NEGATIVE_OBSTACLE",
  "OVERHANG",
] as const

export type DrishtiClassId = number

// Muted mission-control palette, remapped so index 8 (NEGATIVE_OBSTACLE)
// is the ONLY class using the hazard red -- reserved exclusively for
// LETHAL, per Ticket #52's own "Watch out". Every other class sits in a
// desaturated, recessive terrain family (greys, olives, mosses) so nothing
// competes with the hazard/path accents for attention.
export const CLASS_COLOR: Record<DrishtiClassId, string> = {
  0: "#343B40", // UNKNOWN / not yet observed -- cool slate
  1: "#3B4840", // DRIVABLE -- muted sage
  2: "#6B6248", // CAUTION -- muted khaki (rough, passable)
  3: "#55524A", // NON_TRAVERSABLE -- dark muted stone
  4: "#66645B", // STATIC_OBSTACLE -- muted obstacle grey
  5: "#536052", // VEGETATION -- muted moss
  6: "#4D5B66", // VEHICLE -- muted steel blue (another moving platform)
  7: "#8A7A5A", // PEDESTRIAN -- muted warm tan (attention-worthy, not full hazard)
  8: "#E05245", // NEGATIVE_OBSTACLE -- HAZARD, reserved exclusively for danger
  9: "#5A5568", // OVERHANG -- muted violet-grey (overhead, distinct from ground obstacle)
}

export const OBSERVABILITY_COLOR = {
  UNOBSERVED: "#1b212c",
  FREE: "#1f6f5c33",
  OCCUPIED: "#8892a0",
  OCCLUDED: "#3a3f4d",
} as const

export const ACCENT = {
  cyan: PATH_COLOR,
  amber: "#ffb84f",
  red: HAZARD_COLOR,
  green: "#4fe0a0",
} as const
