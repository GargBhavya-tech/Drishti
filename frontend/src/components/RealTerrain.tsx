/**
 * RealTerrain.tsx -- renders the REAL exported multi-resolution grid
 * (eval/export_frames.py's cells_{i}.bin, or accumulated_cells_{i}.bin
 * when `useAccumulated` is set) as one InstancedMesh PER REAL SCHEDULE
 * LEVEL -- four separately-sized box fields, each using that level's
 * own real cell_size_m, real height, real majority class. This is the
 * first time this project's variable-resolution claim is actually
 * VISIBLE: all four real resolution tiers rendered simultaneously, at
 * their real relative sizes, from real data -- not the single uniform
 * synthetic grid the mock demo (Scene.tsx's CellField) has shown until
 * now.
 *
 * `useAccumulated`: when true, sources cells from `frame.accumulatedCells`
 * (export_frames.py's `_WorldCellMemory`, real multi-frame world-frame
 * memory reprojected into this frame's own sensor-local coordinates)
 * instead of `frame.cells` (this frame's own single-sweep snapshot) --
 * the SAME rendering code either way, since both arrays share the
 * identical 5-float schema; only the DATA differs, not the component.
 */

import { useEffect, useMemo, useRef } from "react"
import * as THREE from "three"
import type { LevelInfo, RealFrame } from "../lib/realData"
import { REAL_WORLD_SCALE } from "../lib/realScale"
import { CLASS_COLOR } from "../lib/theme"

// Real observed per-level cell counts on this project's own exported
// demo sequence: level 0 (5cm, <=16.3m) ~44k, level 1 (10cm) ~4k,
// level 2 (20cm) ~0.4k, level 3 (40cm) rarely populated at all --
// RELLIS-3D's off-road returns thin out well before 65-130m (see
// configs/sensor_ouster_os1_64.yaml's own usable_range_m note). Caps
// below carry real headroom over that observed distribution.
const MAX_INSTANCES_PER_LEVEL = [50_000, 6_000, 1_200, 300]
const HEIGHT_EXAGGERATION = 2.5

function LevelField({
  frame,
  level,
  levelInfo,
  useAccumulated,
}: {
  frame: RealFrame
  level: number
  levelInfo: LevelInfo
  useAccumulated: boolean
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const maxInstances = MAX_INSTANCES_PER_LEVEL[level] ?? 2000
  const cellWorldSize = levelInfo.cellSizeM * REAL_WORLD_SCALE

  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return
    const cellData = useAccumulated ? frame.accumulatedCells : frame.cells
    const nCells = useAccumulated ? frame.accumulatedCellCount : frame.cellCount
    let count = 0
    for (let i = 0; i < nCells && count < maxInstances; i++) {
      const off = i * 5
      const cellLevel = cellData[off]
      if (Math.round(cellLevel) !== level) continue
      const gx = cellData[off + 1]
      const gy = cellData[off + 2]
      const heightM = cellData[off + 3]
      const classId = cellData[off + 4]

      const worldX = (gx + 0.5) * levelInfo.cellSizeM * REAL_WORLD_SCALE
      const worldZ = (gy + 0.5) * levelInfo.cellSizeM * REAL_WORLD_SCALE
      const worldY = heightM * REAL_WORLD_SCALE * HEIGHT_EXAGGERATION * 0.5
      const scaleY = Math.max(0.02, Math.abs(heightM) * REAL_WORLD_SCALE * HEIGHT_EXAGGERATION + 0.02)

      dummy.position.set(worldX, worldY, worldZ)
      dummy.scale.set(cellWorldSize * 0.94, scaleY, cellWorldSize * 0.94)
      dummy.updateMatrix()
      mesh.setMatrixAt(count, dummy.matrix)
      mesh.setColorAt(count, new THREE.Color(CLASS_COLOR[Math.round(classId)] ?? "#5b6472"))
      count++
    }
    for (let i = count; i < maxInstances; i++) {
      dummy.position.set(0, -9999, 0)
      dummy.scale.set(0.0001, 0.0001, 0.0001)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
    }
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [frame, level, levelInfo, dummy, maxInstances, cellWorldSize, useAccumulated])

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, maxInstances]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial roughness={0.78} metalness={0.02} />
    </instancedMesh>
  )
}

export function RealTerrain({
  frame,
  levels,
  useAccumulated = false,
}: {
  frame: RealFrame
  levels: LevelInfo[]
  useAccumulated?: boolean
}) {
  return (
    <>
      {levels.map((levelInfo) => (
        <LevelField key={levelInfo.level} frame={frame} level={levelInfo.level} levelInfo={levelInfo} useAccumulated={useAccumulated} />
      ))}
    </>
  )
}
