/**
 * RealPointCloud.tsx -- renders the REAL exported LiDAR point cloud
 * (eval/export_frames.py) as a GPU point-sprite cloud: real x/y/z
 * returns, real FusionSegNet per-point class predictions. A
 * pre-allocated BufferGeometry sized to MAX_POINTS is reused across
 * frames (attribute .set() + setDrawRange(), never a new geometry per
 * frame) so scrubbing the timeline never triggers a GC-provoking
 * allocation churn -- the standard pattern for high-frequency buffer
 * updates in three.js.
 *
 * Per-point size attenuates with camera distance in the vertex shader
 * (GPU-only, no CPU sizing loop) so near returns read as a dense
 * surface and far returns don't vanish to sub-pixel dots.
 */

import { useEffect, useMemo, useRef } from "react"
import * as THREE from "three"
import type { RealFrame } from "../lib/realData"
import { REAL_WORLD_SCALE } from "../lib/realScale"
import { CLASS_COLOR } from "../lib/theme"

const MAX_POINTS = 120_000
const N_CLASSES = 10

function buildClassColorArray(): THREE.Vector3[] {
  const out: THREE.Vector3[] = []
  for (let c = 0; c < N_CLASSES; c++) {
    const color = new THREE.Color(CLASS_COLOR[c] ?? "#5b6472")
    out.push(new THREE.Vector3(color.r, color.g, color.b))
  }
  return out
}

const VERTEX_SHADER = `
  attribute float classId;
  varying vec3 vColor;
  uniform vec3 classColors[${N_CLASSES}];
  void main() {
    int idx = int(classId + 0.5);
    vColor = classColors[idx];
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    float dist = max(-mvPosition.z, 0.1);
    gl_PointSize = clamp(90.0 / dist, 1.2, 6.0);
  }
`

const FRAGMENT_SHADER = `
  varying vec3 vColor;
  void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    if (dot(c, c) > 0.25) discard;
    gl_FragColor = vec4(vColor, 1.0);
  }
`

export function RealPointCloud({ frame }: { frame: RealFrame }) {
  const geomRef = useRef<THREE.BufferGeometry>(null)
  const positionAttrRef = useRef<THREE.BufferAttribute>(null)
  const classAttrRef = useRef<THREE.BufferAttribute>(null)

  const positions = useMemo(() => new Float32Array(MAX_POINTS * 3), [])
  const classes = useMemo(() => new Float32Array(MAX_POINTS), [])
  const classColorArray = useMemo(() => buildClassColorArray(), [])
  const uniforms = useMemo(() => ({ classColors: { value: classColorArray } }), [classColorArray])

  useEffect(() => {
    const n = Math.min(frame.pointCount, MAX_POINTS)
    // Sensor frame is x-forward, y-left, z-up (perception.range_image's
    // own Sweep.xyz convention) -- map straight onto this app's
    // established world convention (x=forward, y=up/height, z=lateral),
    // the SAME mapping CellField/Scene.tsx already use for i/j/height.
    for (let i = 0; i < n; i++) {
      positions[i * 3] = frame.points[i * 4] * REAL_WORLD_SCALE
      positions[i * 3 + 1] = frame.points[i * 4 + 2] * REAL_WORLD_SCALE
      positions[i * 3 + 2] = frame.points[i * 4 + 1] * REAL_WORLD_SCALE
      classes[i] = frame.points[i * 4 + 3]
    }
    if (positionAttrRef.current) {
      positionAttrRef.current.set(positions)
      positionAttrRef.current.needsUpdate = true
    }
    if (classAttrRef.current) {
      classAttrRef.current.set(classes)
      classAttrRef.current.needsUpdate = true
    }
    if (geomRef.current) {
      geomRef.current.setDrawRange(0, n)
      geomRef.current.computeBoundingSphere()
    }
  }, [frame, positions, classes])

  return (
    <points frustumCulled={false}>
      <bufferGeometry ref={geomRef}>
        <bufferAttribute ref={positionAttrRef} attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute ref={classAttrRef} attach="attributes-classId" args={[classes, 1]} />
      </bufferGeometry>
      <shaderMaterial vertexShader={VERTEX_SHADER} fragmentShader={FRAGMENT_SHADER} uniforms={uniforms} />
    </points>
  )
}
