/**
 * theme.ts -- the design system shared by every component: motion
 * tokens (per the motion-ui skill's own spec) and the DRISHTI class
 * color palette.
 *
 * Class colors follow Build Map Ticket #52's own rule: class is
 * CATEGORICAL (hue), height is CONTINUOUS (a separate ramp), confidence
 * is a separate channel (alpha) -- never three variables in one hue
 * ramp. Palette (2026-09 "engineering portal" redesign): a light
 * institutional chrome (header/sidebar/bottom bar) frames a dark,
 * earthy 3D sensor viewport -- the standard GIS/AV-ops convention of a
 * light app shell around a dark data canvas, not a dark sci-fi HUD.
 * Red is reserved exclusively for LETHAL / NEGATIVE_OBSTACLE (matching
 * the same reservation this project's own Python backend makes for
 * LETHAL in planning/conservatism.py); teal is reserved exclusively for
 * the planned route / active navigation state -- never decorative.
 */

// -- Institutional chrome tokens (header, sidebar, bottom bar) ----------
export const CHROME = {
  appBg: "#F3F5F6",
  surface: "#FFFFFF",
  navy: "#17324D",
  navyDark: "#102A43",
  text: "#1F2933",
  textSecondary: "#52606D",
  border: "#D9E2EC",
} as const

// -- The 3D sensor viewport's own dark, earthy world ---------------------
export const SURFACE = {
  appBg: "#1B2119",
  panelBg: "#0D141B",
  panelBorder: "rgba(255,255,255,0.08)",
} as const

export const TEXT = {
  primary: "#E7ECEE",
  secondary: "#96A3A8",
  muted: "#657278",
} as const

// The two semantic accents used everywhere -- in the 3D viewport AND in
// the light chrome (as the sole accent color there too, replacing any
// use of navy-as-decoration). Never reused decoratively -- teal means
// route/active-nav, red means hazard/danger, nothing else borrows either.
export const HAZARD_COLOR = "#C83C32"
export const PATH_COLOR = "#087E8B"

export const GRID_LINE_COLOR = "#536068"

// The UGV model's own material palette (Section 1 of the engineering-
// portal brief) -- kept separate from CLASS_COLOR since the vehicle is
// not a terrain classification.
export const VEHICLE_COLOR = {
  body: "#3E4648",
  dark: "#20272A",
  highlight: "#7C8789",
} as const

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

// Muted earthy/technical palette, remapped so index 8 (NEGATIVE_OBSTACLE)
// is the ONLY class using the hazard red -- reserved exclusively for
// LETHAL, per Ticket #52's own "Watch out". Every other class sits in a
// desaturated, recessive terrain family so nothing competes with the
// hazard/path accents for attention.
export const CLASS_COLOR: Record<DrishtiClassId, string> = {
  0: "#9AA5AB", // UNKNOWN / not yet observed
  1: "#64756A", // DRIVABLE
  2: "#8C8368", // CAUTION -- rough, passable (between drivable and obstacle)
  3: "#5B5548", // NON_TRAVERSABLE -- dark muted stone
  4: "#6B6255", // STATIC_OBSTACLE
  5: "#7D8F78", // VEGETATION
  6: "#5C6B70", // VEHICLE -- muted steel (another moving platform)
  7: "#A08F68", // PEDESTRIAN -- warm tan (attention-worthy, not full hazard)
  8: "#C83C32", // NEGATIVE_OBSTACLE -- HAZARD, reserved exclusively for danger
  9: "#6E6478", // OVERHANG -- muted violet-grey (overhead, distinct from ground obstacle)
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
