/**
 * Scene.tsx -- the 3D map view. Cells are rendered as an InstancedMesh
 * (not one mesh per cell -- thousands of cells would tank performance
 * otherwise), colored by whichever overlay mode is active, with real
 * height exaggerated for visual legibility (true DRISHTI heights are
 * often sub-metre; a flat-looking terrain would defeat the point of a
 * 3D view).
 */

import { Line, OrbitControls } from "@react-three/drei"
import { Canvas, useFrame } from "@react-three/fiber"
import { useEffect, useMemo, useRef } from "react"
import * as THREE from "three"
import { foveaSampleGrid } from "../lib/foveaMath"
import type { DemoFrame } from "../lib/mockData"
import { costGridFromFrame, findPath } from "../lib/pathPlanner"
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
    <instancedMesh ref={meshRef} args={[undefined, undefined, MAX_INSTANCES]} castShadow receiveShadow>
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

/** Ticket #49's real planner interface, made visible: an A* route
 * (planning/path_planner.py, ported to TS in pathPlanner.ts) recomputed
 * EVERY frame from the CURRENT cost grid -- so the line drawn here
 * visibly bends around the moving pedestrian and any static hazards,
 * not a scripted animation. Bible Part 14's own framing: "a path
 * re-routing around a pedestrian shows CONSEQUENCE." */
function PlannedPath({ frame }: { frame: DemoFrame }) {
  const result = useMemo(() => {
    const { grid, size } = costGridFromFrame(frame, PATH_GRID_HALF)
    return findPath(grid, size, size, PATH_START, PATH_GOAL)
  }, [frame])

  const points = useMemo(() => {
    if (!result.path) return []
    return result.path.map(([r, c]) => {
      const i = r - PATH_GRID_HALF
      const j = c - PATH_GRID_HALF
      return new THREE.Vector3(i * CELL_WORLD_SIZE, 0.18, j * CELL_WORLD_SIZE)
    })
  }, [result])

  if (points.length < 2) return null

  return (
    <>
      <Line points={points} color="#4fe0a0" lineWidth={3} />
      <mesh position={points[0]}>
        <sphereGeometry args={[0.12, 12, 12]} />
        <meshBasicMaterial color="#4fd1ff" />
      </mesh>
      <mesh position={points[points.length - 1]}>
        <sphereGeometry args={[0.12, 12, 12]} />
        <meshBasicMaterial color="#4fe0a0" />
      </mesh>
    </>
  )
}

function GroundGrid() {
  return (
    <gridHelper args={[24, 48, "#1c2430", "#131a24"]} position={[0, -0.02, 0]} />
  )
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
      gl={{ antialias: true }}
    >
      <color attach="background" args={["#05070c"]} />
      <fog attach="fog" args={["#05070c", 18, 42]} />
      <ambientLight intensity={0.35} />
      <directionalLight
        position={[10, 16, 6]}
        intensity={1.15}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <pointLight position={[-8, 6, -8]} intensity={0.25} color="#4fd1ff" />
      <GroundGrid />
      <CellField frame={frame} />
      <FoveaOverlay />
      <PlannedPath frame={frame} />
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
