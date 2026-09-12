/**
 * terrainMesh.ts -- builds a single continuous heightfield mesh from a
 * sparse grid of (gx, gy, height, color) cells, replacing the old
 * "one InstancedMesh box per cell" terrain rendering (mission-control
 * redesign, 2026-09). The underlying data structure is UNCHANGED --
 * still a flat per-cell list, still addressed by integer grid indices --
 * only the visual presentation becomes a shaded, continuous surface
 * instead of a field of vertical bars.
 *
 * A cell with no data this frame (decimated far cells in the mock demo,
 * an unpopulated address in a real sparse level) gets `fallbackColor`
 * and height 0 rather than being omitted -- an actual hole in a
 * heightfield mesh reads as a rendering bug, not "unobserved", so it is
 * filled in as flat "not yet seen" ground instead. This never invents a
 * hazard or a height that wasn't in the source data; it only fills the
 * gaps a real occupancy grid would otherwise leave undefined.
 *
 * Normals are computed with BufferGeometry.computeVertexNormals() (a
 * built-in three.js method) rather than hand-rolled -- this is what
 * turns a grid of flat quads into something that reads as continuous
 * terrain once real lighting hits it (see Scene.tsx / RealTerrain.tsx).
 */

import * as THREE from "three"

export interface HeightCell {
  gx: number
  gy: number
  /** Final world-space Y (already scaled/exaggerated by the caller). */
  height: number
  color: THREE.Color
}

export interface HeightfieldOptions {
  minGx: number
  maxGx: number
  minGy: number
  maxGy: number
  /** World size of one grid step along x/z. */
  cellSize: number
  /** 0 for a grid already centered on integer indices (mock demo's i/j),
   * 0.5 for a grid whose indices address a cell's corner, not its
   * center (the real exported gx/gy convention). */
  centerOffset: number
  fallbackColor: THREE.Color
}

/** Builds one continuous heightfield BufferGeometry covering
 * [minGx..maxGx] x [minGy..maxGy]. Cells not present in `cells` are
 * filled flat with `fallbackColor` so the surface has no holes. */
export function buildHeightfieldGeometry(cells: HeightCell[], opts: HeightfieldOptions): THREE.BufferGeometry {
  const width = opts.maxGx - opts.minGx + 1
  const depth = opts.maxGy - opts.minGy + 1
  const nVerts = width * depth

  const heights = new Float32Array(nVerts)
  const colors = new Float32Array(nVerts * 3)
  for (let i = 0; i < nVerts; i++) {
    colors[i * 3] = opts.fallbackColor.r
    colors[i * 3 + 1] = opts.fallbackColor.g
    colors[i * 3 + 2] = opts.fallbackColor.b
  }

  for (const cell of cells) {
    const ix = cell.gx - opts.minGx
    const iz = cell.gy - opts.minGy
    if (ix < 0 || ix >= width || iz < 0 || iz >= depth) continue
    const idx = iz * width + ix
    heights[idx] = cell.height
    colors[idx * 3] = cell.color.r
    colors[idx * 3 + 1] = cell.color.g
    colors[idx * 3 + 2] = cell.color.b
  }

  const positions = new Float32Array(nVerts * 3)
  for (let iz = 0; iz < depth; iz++) {
    for (let ix = 0; ix < width; ix++) {
      const idx = iz * width + ix
      const worldX = (ix + opts.minGx + opts.centerOffset) * opts.cellSize
      const worldZ = (iz + opts.minGy + opts.centerOffset) * opts.cellSize
      positions[idx * 3] = worldX
      positions[idx * 3 + 1] = heights[idx]
      positions[idx * 3 + 2] = worldZ
    }
  }

  const nQuads = (width - 1) * (depth - 1)
  const indices = new Uint32Array(nQuads * 6)
  let ii = 0
  for (let iz = 0; iz < depth - 1; iz++) {
    for (let ix = 0; ix < width - 1; ix++) {
      const a = iz * width + ix
      const b = iz * width + (ix + 1)
      const c = (iz + 1) * width + ix
      const d = (iz + 1) * width + (ix + 1)
      indices[ii++] = a
      indices[ii++] = c
      indices[ii++] = b
      indices[ii++] = b
      indices[ii++] = c
      indices[ii++] = d
    }
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3))
  geometry.setIndex(new THREE.BufferAttribute(indices, 1))
  geometry.computeVertexNormals()
  return geometry
}
