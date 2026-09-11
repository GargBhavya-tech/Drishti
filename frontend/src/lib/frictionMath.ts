/**
 * frictionMath.ts -- a direct TypeScript port of planning/friction.py
 * (the Semantic Friction Governor). Same CLASS_TO_MU table, same
 * "worst class along the lookahead path wins" reduction, same linear
 * a_max derating -- see the Python module's own docstring for the full
 * physical justification, citations, and the taxonomy-honesty caveat
 * (DRISHTI's 10-class taxonomy already merges finer materials like
 * mud/grass/sand for hazard-detection purposes; this can only modulate
 * friction at that same coarser granularity, not finer).
 *
 * CLASS_TO_MU values are cited from wheeled-vehicle terramechanics/
 * braking-friction literature, not measured on this vehicle:
 *   DRIVABLE (dry compacted dirt/gravel): 0.18-0.40 [Wong 2001;
 *     Samuelraj et al. 2018] -- top of range used, matching
 *     vehicle_ugv.yaml's own braking_a_ms2=4.0 (implies mu ~= 0.408).
 *   VEGETATION (wet grass, ice/snow-extrapolated proxy): 0.10-0.20
 *     [Salimi et al. 2015].
 *   CAUTION (mud/saturated soil/standing water): 0.05-0.08, measured
 *     braking mu at optimal slip [Samuelraj et al. 2018].
 */

import type { DrishtiClassId } from "./theme"

// Top of the cited dry-dirt/gravel range -- also nearly exactly what
// vehicle_ugv.yaml's own braking_a_ms2=4.0 already implies (mu ~= 0.408).
export const MU_DRY_REFERENCE = 0.4

// Mirrors planning/friction.py's CLASS_TO_MU exactly (DrishtiClass id -> mu).
export const CLASS_TO_MU: Record<DrishtiClassId, number> = {
  0: 0.15,  // UNKNOWN -- no better than the worst COMMON non-hazard terrain cited (vegetation)
  1: 0.4,   // DRIVABLE -- dry compacted dirt/gravel [Wong 2001; Samuelraj et al. 2018]
  2: 0.07,  // CAUTION -- mud/saturated soil/standing water, measured braking mu [Samuelraj et al. 2018]
  3: 0.07,  // NON_TRAVERSABLE -- hazard fallback, reuses CAUTION's cited value rather than an invented one
  4: 0.07,  // STATIC_OBSTACLE
  5: 0.15,  // VEGETATION -- wet grass, ice/snow-extrapolated proxy [Salimi et al. 2015]
  6: 0.07,  // VEHICLE
  7: 0.07,  // PEDESTRIAN
  8: 0.07,  // NEGATIVE_OBSTACLE
  9: 0.07,  // OVERHANG
}

export const muForClass = (classId: DrishtiClassId): number => CLASS_TO_MU[classId] ?? CLASS_TO_MU[0]

export interface BindingMu {
  mu: number
  bindingClass: DrishtiClassId | null
}

/** The worst (lowest) mu among `classesAlongPath` -- empty input keeps
 * the dry reference (no terrain evidence ahead is not a reason to
 * derate), matching planning/friction.py's own binding_mu(). */
export const bindingMu = (classesAlongPath: DrishtiClassId[]): BindingMu => {
  let mu = MU_DRY_REFERENCE
  let bindingClass: DrishtiClassId | null = null
  for (const c of classesAlongPath) {
    const m = muForClass(c)
    if (m < mu) {
      mu = m
      bindingClass = c
    }
  }
  return { mu, bindingClass }
}

/** braking_a_ms2 derated by mu/MU_DRY_REFERENCE -- see
 * friction_adjusted_vehicle()'s own docstring for the physical
 * justification (a_max = mu * g under the friction-limited-braking
 * model, so deceleration scales linearly with available friction). */
export const derateBrakingA = (brakingAMs2: number, mu: number): number => {
  if (mu <= 0) throw new Error(`mu must be positive, got ${mu}`)
  return brakingAMs2 * (mu / MU_DRY_REFERENCE)
}
