/**
 * frictionMath.ts -- a direct TypeScript port of planning/friction.py
 * (the Semantic Friction Governor). Same CLASS_TO_MU table, same
 * "worst class along the lookahead path wins" reduction, same linear
 * a_max derating -- see the Python module's own docstring for the full
 * physical justification and the taxonomy-honesty caveat (DRISHTI's
 * 10-class taxonomy already merges finer materials like mud/grass/sand
 * for hazard-detection purposes; this can only modulate friction at
 * that same coarser granularity, not finer).
 */

import type { DrishtiClassId } from "./theme"

export const MU_DRY_REFERENCE = 0.8

// Mirrors planning/friction.py's CLASS_TO_MU exactly (DrishtiClass id -> mu).
export const CLASS_TO_MU: Record<DrishtiClassId, number> = {
  0: 0.5,   // UNKNOWN -- conservative middle value, never best-case
  1: 0.8,   // DRIVABLE -- dry dirt/asphalt/concrete, the dry-reference baseline
  2: 0.35,  // CAUTION -- mud/puddle, the worst realistic drivable-adjacent surface
  3: 0.3,   // NON_TRAVERSABLE
  4: 0.3,   // STATIC_OBSTACLE
  5: 0.45,  // VEGETATION -- grass/bush/tree litter
  6: 0.3,   // VEHICLE
  7: 0.3,   // PEDESTRIAN
  8: 0.3,   // NEGATIVE_OBSTACLE
  9: 0.3,   // OVERHANG
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
