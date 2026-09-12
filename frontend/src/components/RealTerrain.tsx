/**
 * RealTerrain.tsx -- renders the REAL exported multi-resolution grid
 * (eval/export_frames.py's cells_{i}.bin) as one CONTINUOUS heightfield
 * mesh PER REAL SCHEDULE LEVEL (terrainMesh.ts), each using that
 * level's own real cell_size_m, real height, real majority class --
 * mission-control redesign, 2026-09: a field of vertical bars reads as
 * a debug visualization, not terrain, so every level now builds one
 * shaded surface instead. This is still the first time this project's
 * variable-resolution claim is actually VISIBLE: all four real
 * resolution tiers rendered simultaneously, at their real relative
 * sizes, from real data -- not the single uniform synthetic grid the
 * mock demo (Scene.tsx) shows.
 *
 * Scope note (see export_frames.py's own docstring): each level's
 * cells come from ONE sweep's real points binned once by the real
 * Nyquist-radius schedule rule, not a live temporally-accumulated
 * clipmap -- an honest single-frame snapshot, not a scrolling map.
 */

import { useEffect, useMemo } from "react"
import * as THREE from "three"
import type { LevelInfo, RealFrame } from "../lib/realData"
import { REAL_WORLD_SCALE } from "../lib/realScale"
import { buildHeightfieldGeometry } from "../lib/terrainMesh"
import { CLASS_COLOR } from "../lib/theme"

const HEIGHT_EXAGGERATION = 2.5
const UNKNOWN_COLOR = new THREE.Color(CLASS_COLOR[0])

function LevelField({ frame, level, levelInfo }: { frame: RealFrame; level: number; levelInfo: LevelInfo }) {
  const geometry = useMemo(() => {
    const cellWorldSize = levelInfo.cellSizeM * REAL_WORLD_SCALE
    const cells: { gx: number; gy: number; height: number; color: THREE.Color }[] = []
    let minGx = Infinity
    let maxGx = -Infinity
    let minGy = Infinity
    let maxGy = -Infinity

    const nCells = frame.cellCount
    for (let i = 0; i < nCells; i++) {
      const off = i * 5
      if (Math.round(frame.cells[off]) !== level) continue
      const gx = Math.round(frame.cells[off + 1])
      const gy = Math.round(frame.cells[off + 2])
      const heightM = frame.cells[off + 3]
      const classId = Math.round(frame.cells[off + 4])

      if (gx < minGx) minGx = gx
      if (gx > maxGx) maxGx = gx
      if (gy < minGy) minGy = gy
      if (gy > maxGy) maxGy = gy

      cells.push({
        gx,
        gy,
        height: heightM * REAL_WORLD_SCALE * HEIGHT_EXAGGERATION,
        color: new THREE.Color(CLASS_COLOR[classId] ?? "#5b6472"),
      })
    }

    // Safety clamp: a real Nyquist-radius level should never span more
    // than a couple hundred cells across, but a corrupt/unexpected
    // export shouldn't be allowed to allocate a runaway vertex grid --
    // fail safe to an empty surface rather than hang the tab.
    const spanTooLarge = maxGx - minGx > 2000 || maxGy - minGy > 2000

    // No cells at this level this frame -- an empty, degenerate
    // geometry rather than a crash (a real level, e.g. the coarsest
    // one, is often unpopulated on a short-range indoor-ish frame).
    if (cells.length === 0 || spanTooLarge) {
      return buildHeightfieldGeometry([], {
        minGx: 0,
        maxGx: 0,
        minGy: 0,
        maxGy: 0,
        cellSize: cellWorldSize,
        centerOffset: 0.5,
        fallbackColor: UNKNOWN_COLOR,
      })
    }

    return buildHeightfieldGeometry(cells, {
      minGx,
      maxGx,
      minGy,
      maxGy,
      cellSize: cellWorldSize,
      centerOffset: 0.5,
      fallbackColor: UNKNOWN_COLOR,
    })
  }, [frame, level, levelInfo])

  useEffect(() => () => geometry.dispose(), [geometry])

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial vertexColors roughness={0.92} metalness={0} />
    </mesh>
  )
}

export function RealTerrain({ frame, levels }: { frame: RealFrame; levels: LevelInfo[] }) {
  return (
    <>
      {levels.map((levelInfo) => (
        <LevelField key={levelInfo.level} frame={frame} level={levelInfo.level} levelInfo={levelInfo} />
      ))}
    </>
  )
}
