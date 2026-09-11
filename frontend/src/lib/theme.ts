/**
 * theme.ts -- the design system shared by every component: motion
 * tokens (per the motion-ui skill's own spec) and the DRISHTI class
 * color palette.
 *
 * Class colors follow Build Map Ticket #52's own rule: class is
 * CATEGORICAL (hue), height is CONTINUOUS (a separate ramp), confidence
 * is a separate channel (alpha) -- never three variables in one hue
 * ramp. Palette is Okabe-Ito derived (colorblind-safe), with red
 * reserved exclusively for LETHAL / NEGATIVE_OBSTACLE, matching the
 * same reservation this project's own Python backend makes for LETHAL
 * in planning/conservatism.py.
 */

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

// Okabe-Ito categorical palette, remapped so index 8 (NEGATIVE_OBSTACLE)
// is the ONLY class using red -- reserved exclusively for LETHAL, per
// Ticket #52's own "Watch out".
export const CLASS_COLOR: Record<DrishtiClassId, string> = {
  0: "#5b6472", // UNKNOWN -- neutral grey
  1: "#3fa7d6", // DRIVABLE -- blue
  2: "#e8b73a", // CAUTION -- amber
  3: "#8a5a44", // NON_TRAVERSABLE -- brown
  4: "#9d7ad1", // STATIC_OBSTACLE -- violet
  5: "#4fae5a", // VEGETATION -- green
  6: "#e07b39", // VEHICLE -- orange
  7: "#f2c14e", // PEDESTRIAN -- yellow
  8: "#e0342c", // NEGATIVE_OBSTACLE -- RED, reserved exclusively for LETHAL
  9: "#3ecfc0", // OVERHANG -- teal
}

export const OBSERVABILITY_COLOR = {
  UNOBSERVED: "#1b212c",
  FREE: "#1f6f5c33",
  OCCUPIED: "#8892a0",
  OCCLUDED: "#3a3f4d",
} as const

export const ACCENT = {
  cyan: "#4fd1ff",
  amber: "#ffb84f",
  red: "#ff5c5c",
  green: "#4fe0a0",
} as const
