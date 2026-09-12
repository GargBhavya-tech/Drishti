/**
 * Scene.tsx -- the 3D map view. Terrain is ONE continuous, smoothed,
 * edge-faded heightfield mesh (terrainMesh.ts) -- GIS-portal redesign,
 * 2026-09: this should read as calm, low-relief ground seen from an
 * elevated operator camera, not a bar chart and not a spiky mountain
 * range. The path and the UGV both sample that SAME rendered surface's
 * height (via the heightfield's own `sampleHeight`), so nothing floats
 * above or clips through the ground the viewer can see.
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
import type { Heightfield } from "../lib/terrainMesh"
import { buildHeightfieldGeometry } from "../lib/terrainMesh"
import { BASE_TERRAIN_COLOR, CLASS_COLOR, HAZARD_COLOR, OBSERVABILITY_COLOR, PATH_COLOR, SURFACE, terrainFillColor } from "../lib/theme"
import type { OverlayMode } from "../state/store"
import { FRAME_COUNT_EXPORT, useDashboardStore } from "../state/store"
import { UgvModel } from "./UgvModel"

const CELL_WORLD_SIZE = 0.32
// heightM is real metres; CELL_WORLD_SIZE is the world-unit size of one
// HORIZONTAL metre. Multiplying height by CELL_WORLD_SIZE first keeps
// vertical and horizontal on the SAME physical scale -- true-scale
// terrain, before any deliberate exaggeration -- and TERRAIN_RELIEF then
// applies a modest, restrained bump (well under the "2-3x for meaningful
// features" allowance) purely for visual legibility. The previous
// version multiplied raw metres directly without the CELL_WORLD_SIZE
// factor, which (independent of any exaggeration constant) made terrain
// read roughly 3x steeper than it physically is -- the dominant reason
// it still looked like a spiky graph rather than gentle ground.
const TERRAIN_RELIEF = 1.8
const HEIGHT_WORLD_SCALE = CELL_WORLD_SIZE * TERRAIN_RELIEF
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
      return new THREE.Color(terrainFillColor(cell.classId))
  }
}

/** Builds the terrain heightfield ONCE per frame/overlay change, shared
 * by the terrain mesh itself AND by the path/vehicle/hazard markers
 * (via `sampleHeight`) so everything in the scene sits on the exact
 * same ground surface the viewer sees -- never a separately-computed
 * "raw" height that could drift from the rendered mesh. */
function useTerrainHeightfield(frame: DemoFrame, overlayMode: OverlayMode): Heightfield {
  const { hMin, hMax } = useMemo(() => {
    let mn = Infinity
    let mx = -Infinity
    for (const c of frame.cells) {
      if (c.heightM < mn) mn = c.heightM
      if (c.heightM > mx) mx = c.heightM
    }
    return { hMin: mn, hMax: mx }
  }, [frame])

  return useMemo(() => {
    const cells = frame.cells.map((cell) => ({
      gx: cell.i,
      gy: cell.j,
      height: cell.heightM * HEIGHT_WORLD_SCALE,
      color: colorForCell(cell, overlayMode, hMin, hMax),
    }))
    return buildHeightfieldGeometry(cells, {
      minGx: -GRID_HALF,
      maxGx: GRID_HALF,
      minGy: -GRID_HALF,
      maxGy: GRID_HALF,
      cellSize: CELL_WORLD_SIZE,
      centerOffset: 0,
      fallbackColor: new THREE.Color(BASE_TERRAIN_COLOR),
      smoothPasses: 3,
      edgeFadeCells: 6,
    })
  }, [frame, overlayMode, hMin, hMax])
}

/** The terrain surface: one continuous, smoothed heightfield mesh. */
function Terrain({ heightfield }: { heightfield: Heightfield }) {
  useEffect(() => () => heightfield.geometry.dispose(), [heightfield])

  return (
    <mesh geometry={heightfield.geometry} castShadow receiveShadow>
      <meshStandardMaterial vertexColors roughness={0.92} metalness={0} />
    </mesh>
  )
}

/** A small, restrained pulse ring above every currently-moving cell
 * (the "motion" overlay) -- the one place motion still needs a live
 * visual cue now that the terrain mesh itself is static per frame. */
function MotionMarkers({ frame, heightfield }: { frame: DemoFrame; heightfield: Heightfield }) {
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
          position={[cell.i * CELL_WORLD_SIZE, heightfield.sampleHeight(cell.i, cell.j) + 0.1, cell.j * CELL_WORLD_SIZE]}
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
 * marker: a boundary ring sized to the hazard region's OWN extent,
 * sitting at the actual (depressed) terrain height there, plus a
 * compact pylon and a plain-language label. The depression itself is
 * the terrain mesh's own doing (the trench cells' real, lower height,
 * softened but not erased by terrainMesh.ts's smoothing) -- this marker
 * only adds the boundary + label on top of it, never a flat red box. */
function HazardMarker({ frame, heightfield }: { frame: DemoFrame; heightfield: Heightfield }) {
  const hazardCells = useMemo(() => frame.cells.filter((c) => c.classId === HAZARD_CLASS_ID), [frame])
  const pulseRef = useRef<THREE.Mesh>(null)

  useFrame((state) => {
    if (!pulseRef.current) return
    pulseRef.current.scale.setScalar(1 + 0.08 * Math.sin(state.clock.elapsedTime * 2.5))
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

  let maxSpread = 1
  for (const c of hazardCells) maxSpread = Math.max(maxSpread, Math.hypot(c.i - centerI, c.j - centerJ))
  const ringOuter = Math.max(0.35, maxSpread * CELL_WORLD_SIZE * 0.75)
  const ringInner = ringOuter * 0.88

  const worldX = centerI * CELL_WORLD_SIZE
  const worldZ = centerJ * CELL_WORLD_SIZE
  const worldY = heightfield.sampleHeight(centerI, centerJ) + 0.03

  return (
    <group position={[worldX, worldY, worldZ]}>
      <mesh ref={pulseRef} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[ringInner, ringOuter, 32]} />
        <meshBasicMaterial color={HAZARD_COLOR} transparent opacity={0.7} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, 0.42, 0]}>
        <coneGeometry args={[0.09, 0.26, 12]} />
        <meshStandardMaterial color={HAZARD_COLOR} emissive={HAZARD_COLOR} emissiveIntensity={0.3} roughness={0.5} />
      </mesh>
      <Html position={[0, 0.78, 0]} center distanceFactor={10} occlude={false}>
        <div className="pointer-events-none select-none rounded-md border border-[#C83C32]/50 bg-[#0D141B]/90 px-2.5 py-1.5 text-center whitespace-nowrap">
          <div className="text-[10px] font-semibold tracking-widest text-[#C83C32] uppercase">Trench &middot; High risk</div>
          <div className="text-[11px] font-mono-tech text-[#E7ECEE]">{distanceM.toFixed(0)} m</div>
        </div>
      </Html>
    </group>
  )
}

const STATIC_OBSTACLE_CLASS_ID = 4
// Fixed, deterministic scatter offsets (fraction of the cluster's own
// bounding box, in [-1, 1]) -- NOT Math.random(), so the layout is
// stable across re-renders instead of jittering every frame.
const ROCK_SCATTER: [number, number, number][] = [
  [-0.5, -0.4, 0.14],
  [0.3, 0.5, 0.1],
  [-0.15, 0.55, 0.08],
  [0.5, -0.35, 0.11],
  [0.05, -0.05, 0.16],
]

/** Obstacles are physical objects, not colored terrain: a small cluster
 * of low-poly rock-like forms (irregular icosahedra) sitting directly
 * on the (unbumped, physically accurate) terrain height at each rock's
 * own position -- the terrain-architecture redesign's own "obstacles
 * should be physical objects... rock, boulder... simple low-poly
 * geometry with proper lighting," never a flat colored patch of ground. */
function ObstacleMarkers({ frame, heightfield }: { frame: DemoFrame; heightfield: Heightfield }) {
  const obstacleCells = useMemo(() => frame.cells.filter((c) => c.classId === STATIC_OBSTACLE_CLASS_ID), [frame])
  if (obstacleCells.length === 0) return null

  let minI = Infinity
  let maxI = -Infinity
  let minJ = Infinity
  let maxJ = -Infinity
  for (const c of obstacleCells) {
    minI = Math.min(minI, c.i)
    maxI = Math.max(maxI, c.i)
    minJ = Math.min(minJ, c.j)
    maxJ = Math.max(maxJ, c.j)
  }
  const centerI = (minI + maxI) / 2
  const centerJ = (minJ + maxJ) / 2
  const halfSpanI = Math.max(1.5, (maxI - minI) / 2)
  const halfSpanJ = Math.max(1.5, (maxJ - minJ) / 2)

  return (
    <>
      {ROCK_SCATTER.map(([fi, fj, size], idx) => {
        const i = centerI + fi * halfSpanI
        const j = centerJ + fj * halfSpanJ
        const worldX = i * CELL_WORLD_SIZE
        const worldZ = j * CELL_WORLD_SIZE
        const worldY = heightfield.sampleHeight(i, j) + size * 0.5
        return (
          <mesh
            key={idx}
            position={[worldX, worldY, worldZ]}
            rotation={[idx * 0.7, idx * 1.3, idx * 0.4]}
            scale={[size, size * 0.8, size]}
            castShadow
            receiveShadow
          >
            <icosahedronGeometry args={[1, 0]} />
            <meshStandardMaterial color={CLASS_COLOR[STATIC_OBSTACLE_CLASS_ID]} roughness={0.95} metalness={0} flatShading />
          </mesh>
        )
      })}
    </>
  )
}

const OVERHANG_CLASS_ID = 9
const OVERHANG_CLEARANCE = 0.55 // world units of headroom the ledge sits above local ground

/** An overhang is a ceiling/clearance hazard ABOVE the vehicle, not
 * raised ground -- represented as a real physical rock face (a wall)
 * with a ledge jutting out from it at headroom height, never as a bump
 * in the terrain's own height field (see mockData.ts's own comment on
 * why that earlier encoding was removed). */
function OverhangMarkers({ frame, heightfield }: { frame: DemoFrame; heightfield: Heightfield }) {
  const overhangCells = useMemo(() => frame.cells.filter((c) => c.classId === OVERHANG_CLASS_ID), [frame])
  if (overhangCells.length === 0) return null

  let minI = Infinity
  let maxI = -Infinity
  let minJ = Infinity
  let maxJ = -Infinity
  for (const c of overhangCells) {
    minI = Math.min(minI, c.i)
    maxI = Math.max(maxI, c.i)
    minJ = Math.min(minJ, c.j)
    maxJ = Math.max(maxJ, c.j)
  }
  const centerJ = (minJ + maxJ) / 2
  const spanJ = Math.max(1, maxJ - minJ) * CELL_WORLD_SIZE
  const spanI = Math.max(1, maxI - minI) * CELL_WORLD_SIZE
  const groundY = heightfield.sampleHeight((minI + maxI) / 2, centerJ)

  const wallX = minI * CELL_WORLD_SIZE
  const ledgeCenterX = ((minI + maxI) / 2) * CELL_WORLD_SIZE
  const worldZ = centerJ * CELL_WORLD_SIZE

  return (
    <group>
      {/* The rock face the ledge protrudes from. */}
      <mesh position={[wallX, groundY + OVERHANG_CLEARANCE, worldZ]} castShadow receiveShadow>
        <boxGeometry args={[0.12, OVERHANG_CLEARANCE * 2 + 0.3, spanJ * 0.96]} />
        <meshStandardMaterial color={CLASS_COLOR[OVERHANG_CLASS_ID]} roughness={0.9} metalness={0} />
      </mesh>
      {/* The horizontal ledge itself, jutting out at headroom height. */}
      <mesh position={[ledgeCenterX, groundY + OVERHANG_CLEARANCE, worldZ]} castShadow receiveShadow>
        <boxGeometry args={[spanI, 0.1, spanJ * 0.9]} />
        <meshStandardMaterial color={CLASS_COLOR[OVERHANG_CLASS_ID]} roughness={0.9} metalness={0} />
      </mesh>
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
      // World mapping matches Terrain: i (x) is "forward", j (z) is lateral.
      const worldX = s.x * CELL_WORLD_SIZE
      const worldZ = s.y * CELL_WORLD_SIZE
      const markerRadius = 0.06 + s.cellSizeM * 0.9
      dummy.position.set(worldX, 0.04, worldZ)
      dummy.scale.set(markerRadius, 0.01, markerRadius)
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      mesh.setMatrixAt(idx, dummy.matrix)
      const t = Math.min(1, s.cellSizeM / 0.4)
      mesh.setColorAt(idx, new THREE.Color().lerpColors(new THREE.Color(PATH_COLOR), new THREE.Color("#ffb84f"), t))
    })
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [samples, dummy])

  if (overlayMode !== "class") return null

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, FOVEA_MAX_MARKERS]}>
      <cylinderGeometry args={[1, 1, 1, 12]} />
      <meshBasicMaterial transparent opacity={0.28} depthWrite={false} />
    </instancedMesh>
  )
}

const PATH_GRID_HALF = 34 // matches mockData.ts's own GRID_HALF -- kept in sync explicitly, not re-derived
const PATH_START: [number, number] = [PATH_GRID_HALF - 30, PATH_GRID_HALF + 17] // i=-30, j=17
const PATH_GOAL: [number, number] = [PATH_GRID_HALF + 30, PATH_GRID_HALF + 17] // i=+30, j=17 -- straight
// across the pedestrian's own j-range (14-20, see mockData.ts's pedestrianPositionAt), so the
// direct route genuinely crosses the moving hazard's path at some point in the sequence.

// Exaggerated for legibility at this dashboard's default camera distance,
// the same reasoning HEIGHT_WORLD_SCALE already applies to terrain relief:
// a true-scale ~0.4m body would read as an unreadable dot from this camera.
const UGV_SCENE_SCALE = 5.5

function lerpAngle(a: number, b: number, t: number): number {
  let diff = b - a
  while (diff > Math.PI) diff -= Math.PI * 2
  while (diff < -Math.PI) diff += Math.PI * 2
  return a + diff * t
}

/** Ticket #49's real planner interface, made visible: an A* route
 * (planning/path_planner.py) recomputed EVERY frame from the CURRENT
 * cost grid, then smoothed with a Catmull-Rom curve
 * (planning/path_smoothing.py, ported to pathSmoothing.ts). Every point
 * on the curve sits at the SAME terrain height the mesh renders
 * (`heightfield.sampleHeight`) plus a small, fixed clearance -- never a
 * flat absolute Y -- so the route visibly follows the ground instead of
 * floating over it.
 *
 * The UGV does not sit statically at the route's start: its position is
 * the replay timeline's own `frameIndex` (the store's real, existing
 * playback state) mapped to a fraction of progress along THIS frame's
 * route, smoothly damped frame-to-frame rather than snapped -- see this
 * function's own useFrame loop. There is no recorded GPS trajectory
 * anywhere in this codebase to play back instead; this is an honest
 * "the vehicle is advancing along its currently-computed safe route as
 * the replay plays" interpretation of the real timeline state, not a
 * fabricated arbitrary loop. */
function RouteAndVehicle({ frame, heightfield }: { frame: DemoFrame; heightfield: Heightfield }) {
  const frameIndex = useDashboardStore((s) => s.frameIndex)
  const vehicleGroupRef = useRef<THREE.Group>(null)
  const pulseRef = useRef<THREE.Mesh>(null)
  const currentPos = useRef(new THREE.Vector3())
  const currentYaw = useRef(0)
  const initialized = useRef(false)

  const pathResult = useMemo(() => {
    const { grid, size } = costGridFromFrame(frame, PATH_GRID_HALF)
    return { result: findPath(grid, size, size, PATH_START, PATH_GOAL), size }
  }, [frame])

  const curvePoints = useMemo(() => {
    const { result } = pathResult
    if (!result.path || result.path.length < 2) return [] as THREE.Vector3[]

    const smoothed = smoothPath(result.path as GridPoint[])
    const toWorld = ([r, c]: GridPoint) => {
      const i = r - PATH_GRID_HALF
      const j = c - PATH_GRID_HALF
      const groundY = heightfield.sampleHeight(i, j)
      return new THREE.Vector3(i * CELL_WORLD_SIZE, groundY + 0.07, j * CELL_WORLD_SIZE)
    }
    return smoothed.map(toWorld)
  }, [pathResult, heightfield])

  useFrame((state, delta) => {
    if (curvePoints.length < 2) return

    const progress = FRAME_COUNT_EXPORT > 1 ? frameIndex / (FRAME_COUNT_EXPORT - 1) : 0
    const targetIdxF = progress * (curvePoints.length - 1)
    const i0 = Math.max(0, Math.min(curvePoints.length - 2, Math.floor(targetIdxF)))
    const i1 = i0 + 1
    const localT = targetIdxF - i0
    const target = curvePoints[i0].clone().lerp(curvePoints[i1], localT)

    if (!initialized.current) {
      currentPos.current.copy(target)
      const tangent0 = curvePoints[i1].clone().sub(curvePoints[i0])
      currentYaw.current = Math.atan2(tangent0.x, tangent0.z)
      initialized.current = true
    } else {
      const damp = 1 - Math.exp(-delta * 4)
      currentPos.current.lerp(target, damp)
      const tangent = curvePoints[i1].clone().sub(curvePoints[i0])
      if (tangent.lengthSq() > 1e-6) {
        const targetYaw = Math.atan2(tangent.x, tangent.z)
        currentYaw.current = lerpAngle(currentYaw.current, targetYaw, damp)
      }
    }

    if (vehicleGroupRef.current) {
      vehicleGroupRef.current.position.copy(currentPos.current)
      vehicleGroupRef.current.rotation.y = currentYaw.current
    }

    // A restrained pulse travelling along the FULL route -- communicates
    // "this is the live-updating planned path," independent of the
    // vehicle's own current progress along it.
    const pulseT = (state.clock.elapsedTime * 0.18) % 1
    const pulseIdx = Math.min(curvePoints.length - 1, Math.floor(pulseT * curvePoints.length))
    if (pulseRef.current) {
      pulseRef.current.position.copy(curvePoints[pulseIdx])
      pulseRef.current.position.y += 0.02
    }
  })

  if (curvePoints.length < 2) return null

  return (
    <>
      {/* Thin core with only a subtle glow -- no giant neon tube. */}
      <Line points={curvePoints} color={PATH_COLOR} lineWidth={5} transparent opacity={0.13} />
      <Line points={curvePoints} color={PATH_COLOR} lineWidth={1.6} />
      <mesh ref={pulseRef} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.04, 0.07, 20]} />
        <meshBasicMaterial color={PATH_COLOR} transparent opacity={0.85} side={THREE.DoubleSide} />
      </mesh>

      <group ref={vehicleGroupRef}>
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.008, 0]}>
          <ringGeometry args={[0.42, 0.48, 32]} />
          <meshBasicMaterial color={PATH_COLOR} transparent opacity={0.6} side={THREE.DoubleSide} />
        </mesh>
        <group scale={UGV_SCENE_SCALE}>
          <group rotation={[0, -Math.PI / 2, 0]}>
            <UgvModel />
          </group>
        </group>
      </group>
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
      <Line points={[origin, targetWorld]} color="#ffe27a" lineWidth={1.5} transparent opacity={0.6} />
      <mesh ref={ringRef} position={targetWorld} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.16, 0.22, 24]} />
        <meshBasicMaterial color="#ffe27a" transparent opacity={0.75} side={THREE.DoubleSide} />
      </mesh>
    </>
  )
}

/** Subordinate spatial-reference grid -- thin, low-opacity, dark grey,
 * pushed below the terrain's own lowest point so it can never poke
 * through the surface (the earlier "graph paper cutting through the
 * terrain" artifact was exactly this plane sitting too close to a
 * heightfield that could dip below it). Reference information, not
 * terrain -- the viewer should see ground first, grid only on a second
 * look. */
function GroundGrid({ minHeight }: { minHeight: number }) {
  const ref = useRef<THREE.GridHelper>(null)
  useEffect(() => {
    const mat = ref.current?.material as THREE.Material | THREE.Material[] | undefined
    const apply = (m: THREE.Material) => {
      m.transparent = true
      m.opacity = 0.045
    }
    if (Array.isArray(mat)) mat.forEach(apply)
    else if (mat) apply(mat)
  }, [])
  return <gridHelper ref={ref} args={[26, 26, "#52606D", "#52606D"]} position={[0, minHeight - 0.25, 0]} />
}

function Rig() {
  useFrame((state) => {
    state.camera.lookAt(0, 0, 0)
  })
  return null
}

export function Scene({ frame }: { frame: DemoFrame }) {
  const overlayMode = useDashboardStore((s) => s.overlayMode)
  const heightfield = useTerrainHeightfield(frame, overlayMode)

  return (
    <Canvas
      shadows
      camera={{ position: [7, 20, 7], fov: 34 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.2 }}
    >
      <color attach="background" args={[SURFACE.appBg]} />
      <fogExp2 attach="fog" args={[SURFACE.appBg, 0.028]} />
      <hemisphereLight args={["#3a4234", "#0a0d10", 0.55]} />
      <ambientLight intensity={0.3} />
      <directionalLight
        position={[-9, 18, 10]}
        intensity={1.7}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      {/* Reference-only: hidden on the default "terrain type" view so the
          clean physical surface is what a first look actually shows;
          available the moment any technical overlay is selected. */}
      {overlayMode !== "class" && <GroundGrid minHeight={heightfield.minHeight} />}
      <Terrain heightfield={heightfield} />
      <ObstacleMarkers frame={frame} heightfield={heightfield} />
      <OverhangMarkers frame={frame} heightfield={heightfield} />
      <MotionMarkers frame={frame} heightfield={heightfield} />
      <HazardMarker frame={frame} heightfield={heightfield} />
      <FoveaOverlay />
      <RouteAndVehicle frame={frame} heightfield={heightfield} />
      <GazeBeam frame={frame} />
      <Rig />
      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        minDistance={8}
        maxDistance={28}
        minPolarAngle={0.26}
        maxPolarAngle={0.55}
      />
    </Canvas>
  )
}
