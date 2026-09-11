/**
 * foveaMath.ts -- a direct TypeScript port of attention/fovea_controller.py
 * (Ticket #47). Same formulas, same constants, so this slider computes
 * REAL numbers today, on the frontend alone, and will match the Python
 * backend exactly once that gets wired in later (per the plan: frontend
 * first, backend later).
 *
 *     v_close(p) = v . p_hat
 *     TTC(p)     = ||p|| / max(v_close(p), v_min)
 *     c_ttc(p)   = c0 * (TTC(p) / tau0) ** gamma
 *     c(p)       = min(c_range(r), c_ttc(p))   -- Profile B, boundary term omitted client-side
 *
 * c_range here uses the SAME schedule formula as sensor.schedule
 * (r_l = c_l / d_theta), with HDL-64E's own real d_theta baked in as
 * the default sensor -- matching this project's own reference config.
 */

export const TAU0_S = 1.0
export const V_MIN_MS = 2.0
export const DEFAULT_GAMMA = 1.0
export const DEFAULT_C0_M = 0.05

// HDL-64E reference config (configs/sensor_hdl64e.yaml) -- real constant.
export const HDL64E_D_THETA_RAD = 0.0030159289474462015

export interface Point2D {
  x: number
  y: number
}

export const vClose = (v: Point2D, p: Point2D): number => {
  const r = Math.hypot(p.x, p.y)
  if (r === 0) return 0
  return (v.x * p.x + v.y * p.y) / r
}

export const ttcSeconds = (v: Point2D, p: Point2D, vMin = V_MIN_MS): number => {
  const r = Math.hypot(p.x, p.y)
  const closing = Math.max(vClose(v, p), vMin)
  return r / closing
}

export const cTtc = (
  v: Point2D,
  p: Point2D,
  gamma = DEFAULT_GAMMA,
  tau0 = TAU0_S,
  c0 = DEFAULT_C0_M,
  vMin = V_MIN_MS,
): number => {
  const t = ttcSeconds(v, p, vMin)
  return c0 * Math.pow(t / tau0, gamma)
}

export const cRange = (rMeters: number, c0 = DEFAULT_C0_M, dThetaRad = HDL64E_D_THETA_RAD, nLevels = 4): number => {
  for (let level = 0; level < nLevels; level++) {
    const cL = c0 * 2 ** level
    const nyquistRadius = cL / dThetaRad
    if (rMeters <= nyquistRadius) return cL
  }
  return c0 * 2 ** (nLevels - 1)
}

export type FoveaProfile = "A" | "B"

export const foveaCellSize = (
  p: Point2D,
  v: Point2D,
  profile: FoveaProfile,
  gamma = DEFAULT_GAMMA,
  c0 = DEFAULT_C0_M,
): number => {
  const r = Math.hypot(p.x, p.y)
  const floor = cRange(r, c0)
  if (profile === "A") return floor
  const ttcTerm = cTtc(v, p, gamma, TAU0_S, c0, V_MIN_MS)
  return Math.min(floor, ttcTerm)
}

/** A full sample grid of cell sizes for a given gamma/velocity, used to
 * drive the live 3D foveation preview -- mirrors eval/pareto.py's own
 * sample-grid pattern (evenly spaced points over a square extent). */
export const foveaSampleGrid = (
  extentM: number,
  nPerAxis: number,
  v: Point2D,
  profile: FoveaProfile,
  gamma: number,
): { x: number; y: number; cellSizeM: number }[] => {
  const out: { x: number; y: number; cellSizeM: number }[] = []
  const step = (2 * extentM) / (nPerAxis - 1)
  for (let i = 0; i < nPerAxis; i++) {
    for (let j = 0; j < nPerAxis; j++) {
      const x = -extentM + i * step
      const y = -extentM + j * step
      out.push({ x, y, cellSizeM: foveaCellSize({ x, y }, v, profile, gamma) })
    }
  }
  return out
}
