/**
 * Scene.tsx -- the 3D map view. Cells are rendered as an InstancedMesh
 * (not one mesh per cell -- thousands of cells would tank performance
 * otherwise), colored by whichever overlay mode is active, with real
 * height exaggerated for visual legibility (true DRISHTI heights are
 * often sub-metre; a flat-looking terrain would defeat the point of a
 * 3D view).
 */

import { Line, OrbitControls } from "@react-three/drei"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { useEffect, useMemo, useRef } from "react"
import * as THREE from "three"
import { findGazeTarget, foveaSampleGrid } from "../lib/foveaMath"
import { gazeCandidatesFromFrame } from "../lib/mockData"
import type { DemoFrame } from "../lib/mockData"
import { classGridFromFrame, costGridFromFrame, findPath } from "../lib/pathPlanner"
import { curvatureSpeedProfile, smoothPath } from "../lib/pathSmoothing"
import type { GridPoint } from "../lib/pathSmoothing"
import { CLASS_COLOR, OBSERVABILITY_COLOR } from "../lib/theme"
import type { OverlayMode } from "../state/store"
import { useDashboardStore } from "../state/store"

const CELL_WORLD_SIZE = 0.32
const HEIGHT_EXAGGERATION = 2.2
const MAX_INSTANCES = 4761 // (2*34+1)^2, the worst-case (undecimated) cell count

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

function CellField({ frame }: { frame: DemoFrame }) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const overlayMode = useDashboardStore((s) => s.overlayMode)
  const pulseRef = useRef(0)

  const { hMin, hMax } = useMemo(() => {
    let mn = Infinity
    let mx = -Infinity
    for (const c of frame.cells) {
      if (c.heightM < mn) mn = c.heightM
      if (c.heightM > mx) mx = c.heightM
    }
    return { hMin: mn, hMax: mx }
  }, [frame])

  const dummy = useMemo(() => new THREE.Object3D(), [])

  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return

    frame.cells.forEach((cell, idx) => {
      const x = cell.i * CELL_WORLD_SIZE
      const z = cell.j * CELL_WORLD_SIZE
      const y = cell.heightM * HEIGHT_EXAGGERATION * 0.5
      const heightScale = Math.max(0.02, cell.heightM * HEIGHT_EXAGGERATION + 0.06)

      dummy.position.set(x, y, z)
      dummy.scale.set(CELL_WORLD_SIZE * 0.92, heightScale, CELL_WORLD_SIZE * 0.92)
      dummy.updateMatrix()
      mesh.setMatrixAt(idx, dummy.matrix)
      mesh.setColorAt(idx, colorForCell(cell, overlayMode, hMin, hMax))
    })

    // Hide unused instance slots (frame count varies due to decimation).
    for (let idx = frame.cells.length; idx < MAX_INSTANCES; idx++) {
      dummy.position.set(0, -9999, 0)
      dummy.scale.set(0.0001, 0.0001, 0.0001)
      dummy.updateMatrix()
      mesh.setMatrixAt(idx, dummy.matrix)
    }

    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [frame, overlayMode, hMin, hMax, dummy])

  // A gentle pulse on the "motion" overlay's moving cells -- communicates
  // state (something here is moving RIGHT NOW), not decoration.
  useFrame((_, delta) => {
    if (overlayMode !== "motion") return
    pulseRef.current += delta
  })

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, MAX_INSTANCES]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial roughness={0.7} metalness={0.05} />
    </instancedMesh>
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

// Perceptual cap for the path's speed-limit colour ramp: green means
// "no worse than the ~43km/h dry-hazard baseline SpeedGauge's own gauge
// tops out near," red means the kinodynamic governor has cut speed
// hard. Not a real vehicle limit by itself -- just this colour ramp's
// own reference point, kept in one place.
const SPEED_COLOR_CAP_MS = 12

function speedLimitColor(vMs: number): THREE.Color {
  const t = Number.isFinite(vMs) ? Math.min(1, Math.max(0, vMs / SPEED_COLOR_CAP_MS)) : 1
  return new THREE.Color().lerpColors(new THREE.Color("#e0342c"), new THREE.Color("#4fe0a0"), t)
}

/** Ticket #49's real planner interface, made visible AND made
 * kinodynamically honest: an A* route (planning/path_planner.py)
 * recomputed EVERY frame from the CURRENT cost grid, then smoothed
 * with a Catmull-Rom curve (planning/path_smoothing.py, ported to
 * pathSmoothing.ts) so it no longer shows the raw grid's artificial
 * 45-degree kinks, and coloured along its length by the SAME module's
 * friction-and-curvature-limited cornering speed (green = fast, red =
 * "the governor is slowing down for this stretch") -- Bible Part 14's
 * own framing ("a path re-routing around a pedestrian shows
 * CONSEQUENCE") extended to show WHY the speed varies along the route,
 * not just where the route goes. */
function PlannedPath({ frame }: { frame: DemoFrame }) {
  const pathResult = useMemo(() => {
    const { grid, size } = costGridFromFrame(frame, PATH_GRID_HALF)
    return { result: findPath(grid, size, size, PATH_START, PATH_GOAL), size }
  }, [frame])

  const { curvePoints, curveColors } = useMemo(() => {
    const { result, size } = pathResult
    if (!result.path || result.path.length < 2) {
      return { curvePoints: [] as THREE.Vector3[], curveColors: [] as THREE.Color[] }
    }

    const { classGrid } = classGridFromFrame(frame, PATH_GRID_HALF)
    const smoothed = smoothPath(result.path as GridPoint[])
    const classAt = (p: GridPoint): number => {
      const r = Math.round(Math.max(0, Math.min(size - 1, p[0])))
      const c = Math.round(Math.max(0, Math.min(size - 1, p[1])))
      return classGrid[r * size + c]
    }
    // cellSizeM=1.0 -- this demo's own established convention (see
    // Scene.tsx's FoveaOverlay/foveaSampleGrid): one grid-index unit is
    // treated as one metre for every physics formula here, with
    // CELL_WORLD_SIZE applied ONLY at the final Three.js world-position
    // step below, identically for every other renderer in this file.
    const profile = curvatureSpeedProfile(smoothed, 1.0, classAt)

    const toWorld = ([r, c]: GridPoint) =>
      new THREE.Vector3((r - PATH_GRID_HALF) * CELL_WORLD_SIZE, 0.18, (c - PATH_GRID_HALF) * CELL_WORLD_SIZE)

    const curvePoints = smoothed.map(toWorld)
    const curveColors = smoothed.map((_, idx) => {
      // profile[] covers only the curve's interior points (indices
      // 1..N-2 of `smoothed`); endpoints borrow their nearest interior
      // sample. An empty profile (path too short to have curvature)
      // defaults to "no limit" green rather than an arbitrary colour.
      if (profile.length === 0) return speedLimitColor(Infinity)
      const profileIdx = Math.min(profile.length - 1, Math.max(0, idx - 1))
      return speedLimitColor(profile[profileIdx].vMaxMs)
    })

    return { curvePoints, curveColors }
  }, [frame, pathResult])

  if (curvePoints.length < 2) return null

  return (
    <>
      <Line points={curvePoints} vertexColors={curveColors} lineWidth={3} />
      <mesh position={curvePoints[0]}>
        <sphereGeometry args={[0.12, 12, 12]} />
        <meshBasicMaterial color="#4fd1ff" />
      </mesh>
      <mesh position={curvePoints[curvePoints.length - 1]}>
        <sphereGeometry args={[0.12, 12, 12]} />
        <meshBasicMaterial color="#4fe0a0" />
      </mesh>
    </>
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

function GroundGrid() {
  return (
    <gridHelper args={[24, 48, "#1c2430", "#131a24"]} position={[0, -0.02, 0]} />
  )
}

function CameraPresetController() {
  const cameraPreset = useDashboardStore((s) => s.cameraPreset)
  const camera = useThree((state) => state.camera)

  useEffect(() => {
    const presets = {
      driver: { position: [12, 2.8, 0] as const, target: [0, 0.2, 0] as const },
      tactical: { position: [14, 12, 14] as const, target: [0, 0, 0] as const },
      hazard: { position: [4, 4.5, -8] as const, target: [-0.3, 0.1, -3] as const },
    }
    const preset = presets[cameraPreset]
    camera.position.set(preset.position[0], preset.position[1], preset.position[2])
    camera.lookAt(preset.target[0], preset.target[1], preset.target[2])
  }, [camera, cameraPreset])

  return null
}

export function Scene({ frame }: { frame: DemoFrame }) {
  return (
    <Canvas
      camera={{ position: [14, 12, 14], fov: 42 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true }}
    >
      <color attach="background" args={["#05070c"]} />
      <fog attach="fog" args={["#05070c", 18, 42]} />
      <ambientLight intensity={0.35} />
      <directionalLight position={[10, 16, 6]} intensity={1.15} />
      <pointLight position={[-8, 6, -8]} intensity={0.25} color="#4fd1ff" />
      <GroundGrid />
      <CellField frame={frame} />
      <FoveaOverlay />
      <PlannedPath frame={frame} />
      <GazeBeam frame={frame} />
      <CameraPresetController />
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
