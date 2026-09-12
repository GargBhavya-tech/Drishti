/**
 * Scene.tsx -- the 3D map view. Terrain is rendered as ONE continuous
 * heightfield mesh (terrainMesh.ts), not a field of vertical bars --
 * mission-control redesign, 2026-09: a DRDO evaluator should read this
 * as ground with relief, not a bar chart. Real height is still
 * exaggerated for visual legibility (true DRISHTI heights are often
 * sub-metre; a flat-looking terrain would defeat the point of a 3D
 * view) -- only the geometry changed, not the underlying per-cell data.
 */

import { Html, Line, OrbitControls } from "@react-three/drei"
import { Canvas, useFrame } from "@react-three/fiber"
import { useEffect, useMemo, useRef } from "react"
import * as THREE from "three"
import { findGazeTarget, foveaSampleGrid } from "../lib/foveaMath"
import { gazeCandidatesFromFrame } from "../lib/mockData"
import type { DemoFrame } from "../lib/mockData"
import { costGridFromFrame, findPath } from "../lib/pathPlanner"
import { smoothPath } from "../lib/pathSmoothing"
import type { GridPoint } from "../lib/pathSmoothing"
import { buildHeightfieldGeometry } from "../lib/terrainMesh"
import { CLASS_COLOR, HAZARD_COLOR, OBSERVABILITY_COLOR, PATH_COLOR } from "../lib/theme"
import type { OverlayMode } from "../state/store"
import { useDashboardStore } from "../state/store"

const CELL_WORLD_SIZE = 0.32
const HEIGHT_EXAGGERATION = 2.2
const GRID_HALF = 34 // matches mockData.ts's own GRID_HALF -- kept in sync explicitly, not re-derived

const SPARSITY_COLOR: Record<string, string> = {
  NORMAL: "#3fa7d6",
  SPARSE_STRUCTURED: "#e8b73a",
  NOISE_SUPPRESSED: "#5b6472",
  FREE: "#1f2732",
  UNKNOWN: "#e0342c",
}

const HEIGHT_RAMP = ["#0d3b66", "#1f6f5c", "#4fae5a", "#e8b73a", "#e07b39", "#e0342c"]

function heightRampColor(h: number, min: number, max: number): string {
  const t = max > min ? (h - min) / (max - min) : 0.5
  const idx = Math.min(HEIGHT_RAMP.length - 1, Math.max(0, Math.floor(t * HEIGHT_RAMP.length)))
  return HEIGHT_RAMP[idx]
}

// Attention overlay: a dark-to-bright heat ramp (Grad-CAM-style), driven
// by CellSample.attentionScore -- SYNTHETIC in this mock-data demo (see
// that field's own doc comment in mockData.ts), not a live model.
const ATTENTION_RAMP = ["#0b0033", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"]

function attentionRampColor(score: number): string {
  const idx = Math.min(ATTENTION_RAMP.length - 1, Math.max(0, Math.floor(score * ATTENTION_RAMP.length)))
  return ATTENTION_RAMP[idx]
}

function colorForCell(cell: DemoFrame["cells"][number], mode: OverlayMode, hMin: number, hMax: number): THREE.Color {
  switch (mode) {
    case "observability":
      return new THREE.Color(OBSERVABILITY_COLOR[cell.observability] || "#333")
    case "sparsity":
      return new THREE.Color(SPARSITY_COLOR[cell.sparsityVerdict] || "#333")
    case "height":
      return new THREE.Color(heightRampColor(cell.heightM, hMin, hMax))
    case "motion":
      return new THREE.Color(cell.isMoving ? "#ff5c5c" : "#1b2735")
    case "attention":
      return new THREE.Color(attentionRampColor(cell.attentionScore))
    case "class":
    default:
      return new THREE.Color(CLASS_COLOR[cell.classId] ?? "#5b6472")
  }
}

/** The terrain surface: one continuous heightfield mesh built from the
 * frame's per-cell (i, j, height, class) data (terrainMesh.ts), replacing
 * the old per-cell InstancedMesh box field. Same source data, same
 * overlay-mode coloring -- only the geometry is now a shaded surface
 * instead of a bar chart, per the mission-control redesign's #1
 * priority. Moving cells (the "motion" overlay) still get a red pulse,
 * drawn as a thin marker ring above the surface rather than a color
 * change baked into the (now-shared, static-per-frame) vertex buffer. */
function Terrain({ frame }: { frame: DemoFrame }) {
  const meshRef = useRef<THREE.Mesh>(null)
  const overlayMode = useDashboardStore((s) => s.overlayMode)

  const { hMin, hMax } = useMemo(() => {
    let mn = Infinity
    let mx = -Infinity
    for (const c of frame.cells) {
      if (c.heightM < mn) mn = c.heightM
      if (c.heightM > mx) mx = c.heightM
    }
    return { hMin: mn, hMax: mx }
  }, [frame])

  const geometry = useMemo(() => {
    const cells = frame.cells.map((cell) => ({
      gx: cell.i,
      gy: cell.j,
      height: cell.heightM * HEIGHT_EXAGGERATION,
      color: colorForCell(cell, overlayMode, hMin, hMax),
    }))
    return buildHeightfieldGeometry(cells, {
      minGx: -GRID_HALF,
      maxGx: GRID_HALF,
      minGy: -GRID_HALF,
      maxGy: GRID_HALF,
      cellSize: CELL_WORLD_SIZE,
      centerOffset: 0,
      fallbackColor: new THREE.Color(CLASS_COLOR[0]),
    })
  }, [frame, overlayMode, hMin, hMax])

  useEffect(() => () => geometry.dispose(), [geometry])

  return (
    <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
      <meshStandardMaterial vertexColors roughness={0.9} metalness={0} />
    </mesh>
  )
}

/** A small, restrained pulse ring above every currently-moving cell
 * (the "motion" overlay) -- the one place motion still needs a live
 * visual cue now that the terrain mesh itself is static per frame. */
function MotionMarkers({ frame }: { frame: DemoFrame }) {
  const overlayMode = useDashboardStore((s) => s.overlayMode)
  const ringRefs = useRef<THREE.Mesh[]>([])
  const movingCells = useMemo(() => frame.cells.filter((c) => c.isMoving), [frame])

  useFrame((state) => {
    const pulse = 1 + 0.25 * Math.sin(state.clock.elapsedTime * 5)
    ringRefs.current.forEach((r) => r?.scale.setScalar(pulse))
  })

  if (overlayMode !== "motion" || movingCells.length === 0) return null

  return (
    <>
      {movingCells.map((cell, idx) => (
        <mesh
          key={`${cell.i}-${cell.j}`}
          ref={(el) => {
            if (el) ringRefs.current[idx] = el
          }}
          position={[cell.i * CELL_WORLD_SIZE, cell.heightM * HEIGHT_EXAGGERATION + 0.15, cell.j * CELL_WORLD_SIZE]}
          rotation={[-Math.PI / 2, 0, 0]}
        >
          <ringGeometry args={[0.14, 0.19, 20]} />
          <meshBasicMaterial color="#ff5c5c" transparent opacity={0.85} side={THREE.DoubleSide} />
        </mesh>
      ))}
    </>
  )
}

const HAZARD_CLASS_ID = 8 // NEGATIVE_OBSTACLE -- the only class this app treats as a hazard marker/label target

/** Professional warning symbology for a hazard region, not a game
 * marker: a restrained boundary ring in the hazard accent color, a
 * compact pylon, and a plain-language label (class + distance) --
 * never a giant glow. Position and distance are computed from the
 * SAME per-cell data the terrain mesh renders, never invented. */
function HazardMarker({ frame }: { frame: DemoFrame }) {
  const hazardCells = useMemo(() => frame.cells.filter((c) => c.classId === HAZARD_CLASS_ID), [frame])
  const pulseRef = useRef<THREE.Mesh>(null)

  useFrame((state) => {
    if (!pulseRef.current) return
    pulseRef.current.scale.setScalar(1 + 0.12 * Math.sin(state.clock.elapsedTime * 3))
  })

  if (hazardCells.length === 0) return null

  let sumI = 0
  let sumJ = 0
  for (const c of hazardCells) {
    sumI += c.i
    sumJ += c.j
  }
  const centerI = sumI / hazardCells.length
  const centerJ = sumJ / hazardCells.length
  const distanceM = Math.hypot(centerI, centerJ) // one grid-index unit == one metre, this codebase's own convention

  const worldX = centerI * CELL_WORLD_SIZE
  const worldZ = centerJ * CELL_WORLD_SIZE
  const worldY = 0.04

  return (
    <group position={[worldX, worldY, worldZ]}>
      <mesh ref={pulseRef} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.55, 0.68, 32]} />
        <meshBasicMaterial color={HAZARD_COLOR} transparent opacity={0.75} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, 0.55, 0]}>
        <coneGeometry args={[0.11, 0.32, 12]} />
        <meshStandardMaterial color={HAZARD_COLOR} emissive={HAZARD_COLOR} emissiveIntensity={0.35} roughness={0.5} />
      </mesh>
      <Html position={[0, 0.95, 0]} center distanceFactor={10} occlude={false}>
        <div className="pointer-events-none select-none rounded-md border border-[#E05245]/50 bg-[#0D141B]/90 px-2.5 py-1.5 text-center whitespace-nowrap">
          <div className="text-[10px] font-semibold tracking-widest text-[#E05245] uppercase">Trench &middot; High risk</div>
          <div className="text-[11px] font-mono-tech text-[#E7ECEE]">{distanceM.toFixed(0)} m</div>
        </div>
      </Html>
    </group>
  )
}

const FOVEA_EXTENT_M = 22
const FOVEA_GRID_N = 23
const FOVEA_MAX_MARKERS = FOVEA_GRID_N * FOVEA_GRID_N

/** The live foveation preview -- Bible Part 13's own "elongated
 * teardrop aligned with the velocity vector": small, dense markers
 * where the fovea controller would keep cells fine, larger sparse
 * markers where it lets them coarsen. Driven directly by gamma (the
 * store's live slider) and a fixed forward ego velocity, using the
 * exact ported formulas from foveaMath.ts -- real numbers, not a
 * decorative animation. */
function FoveaOverlay() {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const gamma = useDashboardStore((s) => s.gamma)
  const egoSpeedMs = useDashboardStore((s) => s.egoSpeedMs)
  const overlayMode = useDashboardStore((s) => s.overlayMode)
  const dummy = useMemo(() => new THREE.Object3D(), [])

  const samples = useMemo(
    () => foveaSampleGrid(FOVEA_EXTENT_M, FOVEA_GRID_N, { x: egoSpeedMs, y: 0 }, "B", gamma),
    [gamma, egoSpeedMs],
  )

  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return
    samples.forEach((s, idx) => {
      // World mapping matches CellField: i (x) is "forward", j (z) is lateral.
      const worldX = s.x * CELL_WORLD_SIZE
      const worldZ = s.y * CELL_WORLD_SIZE
      const markerRadius = 0.06 + s.cellSizeM * 0.9
      dummy.position.set(worldX, 0.04, worldZ)
      dummy.scale.set(markerRadius, 0.01, markerRadius)
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      mesh.setMatrixAt(idx, dummy.matrix)
      const t = Math.min(1, s.cellSizeM / 0.4)
      mesh.setColorAt(idx, new THREE.Color().lerpColors(new THREE.Color("#4fd1ff"), new THREE.Color("#ffb84f"), t))
    })
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [samples, dummy])

  if (overlayMode !== "class") return null

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, FOVEA_MAX_MARKERS]}>
      <cylinderGeometry args={[1, 1, 1, 12]} />
      <meshBasicMaterial transparent opacity={0.35} depthWrite={false} />
    </instancedMesh>
  )
}

const PATH_GRID_HALF = 34 // matches mockData.ts's own GRID_HALF -- kept in sync explicitly, not re-derived
const PATH_START: [number, number] = [PATH_GRID_HALF - 30, PATH_GRID_HALF + 17] // i=-30, j=17
const PATH_GOAL: [number, number] = [PATH_GRID_HALF + 30, PATH_GRID_HALF + 17] // i=+30, j=17 -- straight
// across the pedestrian's own j-range (14-20, see mockData.ts's pedestrianPositionAt), so the
// direct route genuinely crosses the moving hazard's path at some point in the sequence.

/** Ticket #49's real planner interface, made visible: an A* route
 * (planning/path_planner.py) recomputed EVERY frame from the CURRENT
 * cost grid, then smoothed with a Catmull-Rom curve
 * (planning/path_smoothing.py, ported to pathSmoothing.ts) so it no
 * longer shows the raw grid's artificial 45-degree kinks.
 *
 * Rendered as a single restrained cyan "safe route" -- the mission-
 * control redesign's own rule that cyan is reserved exclusively for
 * the active route/nav state, nothing else. The underlying friction +
 * curvature speed profile (planning/path_smoothing.py's own
 * curvatureSpeedProfile) is still computed here and handed to the
 * caller so GovernorPanel's "why would the vehicle slow down" readout
 * stays truthful -- only the per-segment colour-by-speed encoding on
 * the 3D line itself was removed, in favour of the plain-language
 * governor panel already reporting the binding constraint in words. */
function PlannedPath({ frame }: { frame: DemoFrame }) {
  const pulseRef = useRef<THREE.Mesh>(null)

  const pathResult = useMemo(() => {
    const { grid, size } = costGridFromFrame(frame, PATH_GRID_HALF)
    return { result: findPath(grid, size, size, PATH_START, PATH_GOAL), size }
  }, [frame])

  const curvePoints = useMemo(() => {
    const { result } = pathResult
    if (!result.path || result.path.length < 2) return [] as THREE.Vector3[]

    const smoothed = smoothPath(result.path as GridPoint[])
    const toWorld = ([r, c]: GridPoint) =>
      new THREE.Vector3((r - PATH_GRID_HALF) * CELL_WORLD_SIZE, 0.18, (c - PATH_GRID_HALF) * CELL_WORLD_SIZE)
    return smoothed.map(toWorld)
  }, [pathResult])

  useFrame((state) => {
    if (!pulseRef.current || curvePoints.length < 2) return
    // A restrained pulse travelling along the route -- communicates
    // "this is the active, live-updating route," not decoration.
    const t = (state.clock.elapsedTime * 0.18) % 1
    const idx = Math.min(curvePoints.length - 1, Math.floor(t * curvePoints.length))
    pulseRef.current.position.copy(curvePoints[idx])
    pulseRef.current.position.y += 0.02
  })

  if (curvePoints.length < 2) return null

  return (
    <>
      {/* Subtle outer glow beneath the crisp core line. */}
      <Line points={curvePoints} color={PATH_COLOR} lineWidth={9} transparent opacity={0.16} />
      <Line points={curvePoints} color={PATH_COLOR} lineWidth={2.5} />
      <mesh ref={pulseRef} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.05, 0.09, 20]} />
        <meshBasicMaterial color={PATH_COLOR} transparent opacity={0.9} side={THREE.DoubleSide} />
      </mesh>
      <VehicleMarker position={curvePoints[0]} heading={curvePoints[1]} />
    </>
  )
}

/** A compact top-down vehicle chevron -- position + heading, nothing
 * more. Replaces the old plain glowing sphere; sits at the current
 * (first) point of the planned route, oriented toward the next point. */
function VehicleMarker({ position, heading }: { position: THREE.Vector3; heading: THREE.Vector3 }) {
  const yaw = Math.atan2(heading.x - position.x, heading.z - position.z)
  return (
    <group position={[position.x, position.y + 0.02, position.z]} rotation={[0, yaw, 0]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <shapeGeometry
          args={[
            (() => {
              const s = new THREE.Shape()
              s.moveTo(0, 0.22)
              s.lineTo(0.14, -0.16)
              s.lineTo(0, -0.06)
              s.lineTo(-0.14, -0.16)
              s.closePath()
              return s
            })(),
          ]}
        />
        <meshStandardMaterial color="#E7ECEE" roughness={0.6} metalness={0} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0.001]}>
        <ringGeometry args={[0.22, 0.25, 24]} />
        <meshBasicMaterial color={PATH_COLOR} transparent opacity={0.8} side={THREE.DoubleSide} />
      </mesh>
    </group>
  )
}

const GAZE_URGENCY_THRESHOLD_S = 6 // don't saccade onto a hazard many seconds away -- urgency IS time-to-contact

/** Saccadic gaze steering, made visible: attention/fovea_controller.py's
 * find_gaze_target() (ported to foveaMath.ts) picks the single most
 * urgent (soonest-TTC) candidate hazard cell each frame -- a beam and a
 * pulsing ring lock onto it, dramatising "look where you'll be soon,"
 * Bible Part 13's own framing, rather than the always-dead-ahead gaze a
 * typical demo defaults to. */
function GazeBeam({ frame }: { frame: DemoFrame }) {
  const egoSpeedMs = useDashboardStore((s) => s.egoSpeedMs)
  const ringRef = useRef<THREE.Mesh>(null)

  const target = useMemo(() => {
    const candidates = gazeCandidatesFromFrame(frame)
    return findGazeTarget(candidates, { x: egoSpeedMs, y: 0 })
  }, [frame, egoSpeedMs])

  useFrame((state) => {
    if (!ringRef.current) return
    const pulse = 1 + 0.18 * Math.sin(state.clock.elapsedTime * 4)
    ringRef.current.scale.setScalar(pulse)
  })

  if (!target || target.ttcS > GAZE_URGENCY_THRESHOLD_S) return null

  const origin = new THREE.Vector3(0, 0.05, 0)
  const targetWorld = new THREE.Vector3(target.point.x * CELL_WORLD_SIZE, 0.05, target.point.y * CELL_WORLD_SIZE)

  return (
    <>
      <Line points={[origin, targetWorld]} color="#ffe27a" lineWidth={1.5} transparent opacity={0.75} />
      <mesh ref={ringRef} position={targetWorld} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.16, 0.22, 24]} />
        <meshBasicMaterial color="#ffe27a" transparent opacity={0.85} side={THREE.DoubleSide} />
      </mesh>
    </>
  )
}

/** Subordinate spatial-reference grid -- thin, low-opacity, dark grey,
 * never competing with terrain (mission-control redesign's own grid
 * spec). gridHelper's default material isn't transparent by default,
 * so opacity is set explicitly once the material exists. */
function GroundGrid() {
  const ref = useRef<THREE.GridHelper>(null)
  useEffect(() => {
    const mat = ref.current?.material as THREE.Material | THREE.Material[] | undefined
    const apply = (m: THREE.Material) => {
      m.transparent = true
      m.opacity = 0.1
    }
    if (Array.isArray(mat)) mat.forEach(apply)
    else if (mat) apply(mat)
  }, [])
  return <gridHelper ref={ref} args={[24, 48, "#536068", "#536068"]} position={[0, -0.02, 0]} />
}

function Rig() {
  useFrame((state) => {
    state.camera.lookAt(0, 0, 0)
  })
  return null
}

export function Scene({ frame }: { frame: DemoFrame }) {
  return (
    <Canvas
      shadows
      camera={{ position: [14, 12, 14], fov: 42 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.25 }}
    >
      <color attach="background" args={["#080D12"]} />
      <fogExp2 attach="fog" args={["#080D12", 0.032]} />
      <hemisphereLight args={["#2a3540", "#0a0d10", 0.55]} />
      <ambientLight intensity={0.25} />
      <directionalLight
        position={[-9, 18, 10]}
        intensity={1.8}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <GroundGrid />
      <Terrain frame={frame} />
      <MotionMarkers frame={frame} />
      <HazardMarker frame={frame} />
      <FoveaOverlay />
      <PlannedPath frame={frame} />
      <GazeBeam frame={frame} />
      <Rig />
      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        minDistance={6}
        maxDistance={40}
        maxPolarAngle={Math.PI / 2.05}
      />
    </Canvas>
  )
}
