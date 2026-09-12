/**
 * VariableResGrid.tsx -- a single fullscreen-plane fragment shader
 * that draws grid lines whose spacing changes with distance from the
 * ego origin, using this project's own REAL resolution schedule
 * (sensor.schedule.generate_schedule, exported via
 * eval/export_frames.py's manifest.json) -- the PS's own headline
 * "variable resolution" claim, made literally visible for the first
 * time, entirely on the GPU: one draw call, zero CPU cost per frame,
 * no per-ring geometry or LOD switching.
 */

import { useMemo } from "react"
import type { LevelInfo } from "../lib/realData"
import { REAL_WORLD_SCALE } from "../lib/realScale"

const MAX_LEVELS = 4
const PLANE_HALF_EXTENT_M = 140 // covers the coarsest level's own ~130m Nyquist radius with margin
const UNUSED_LEVEL_RADIUS_M = 1e9 // stand-in for "infinite" that's still a safe finite GPU float

const VERTEX_SHADER = `
  varying vec2 vWorldXZ;
  void main() {
    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    vWorldXZ = worldPosition.xz;
    gl_Position = projectionMatrix * viewMatrix * worldPosition;
  }
`

const FRAGMENT_SHADER = `
  varying vec2 vWorldXZ;
  uniform float cellSizes[${MAX_LEVELS}];
  uniform float radii[${MAX_LEVELS}];
  uniform float worldScale;
  uniform int nLevels;

  float gridLine(vec2 coord, float spacing) {
    vec2 g = abs(fract(coord / spacing - 0.5) - 0.5) / fwidth(coord / spacing);
    return 1.0 - min(min(g.x, g.y), 1.0);
  }

  void main() {
    float r = length(vWorldXZ) / worldScale; // world units -> real metres
    float spacing = cellSizes[${MAX_LEVELS - 1}];
    for (int i = 0; i < ${MAX_LEVELS}; i++) {
      if (i >= nLevels) break;
      if (r <= radii[i]) { spacing = cellSizes[i]; break; }
    }
    float line = gridLine(vWorldXZ, spacing * worldScale);
    if (line < 0.05) discard;
    // Subordinate spatial reference only -- a single muted grey, never
    // competing with the terrain/hazard/path colours above it, fading
    // from ~10% near to fully transparent by the coarsest level's own
    // Nyquist radius (mission-control redesign's own grid spec).
    vec3 gridColor = vec3(0.325, 0.376, 0.407);
    float distFade = 1.0 - clamp(r / radii[${MAX_LEVELS - 1}], 0.0, 1.0);
    gl_FragColor = vec4(gridColor, line * 0.11 * distFade);
  }
`

export function VariableResGrid({ levels }: { levels: LevelInfo[] }) {
  const uniforms = useMemo(() => {
    const fallbackCellSize = levels[levels.length - 1]?.cellSizeM ?? 0.4
    const cellSizes = new Array(MAX_LEVELS).fill(fallbackCellSize)
    const radii = new Array(MAX_LEVELS).fill(UNUSED_LEVEL_RADIUS_M)
    levels.forEach((l, i) => {
      if (i < MAX_LEVELS) {
        cellSizes[i] = l.cellSizeM
        radii[i] = l.nyquistRadiusM
      }
    })
    return {
      cellSizes: { value: cellSizes },
      radii: { value: radii },
      worldScale: { value: REAL_WORLD_SCALE },
      nLevels: { value: Math.min(levels.length, MAX_LEVELS) },
    }
  }, [levels])

  const halfExtent = PLANE_HALF_EXTENT_M * REAL_WORLD_SCALE

  return (
    <mesh position={[0, -0.03, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[halfExtent * 2, halfExtent * 2, 1, 1]} />
      <shaderMaterial vertexShader={VERTEX_SHADER} fragmentShader={FRAGMENT_SHADER} uniforms={uniforms} transparent depthWrite={false} />
    </mesh>
  )
}
