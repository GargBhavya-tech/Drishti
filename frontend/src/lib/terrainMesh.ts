/**
 * terrainMesh.ts -- builds a single continuous, LOW-RELIEF heightfield
 * mesh from a sparse grid of (gx, gy, height, color) cells (GIS-portal
 * redesign, 2026-09, replacing the first heightfield pass that still
 * read as a spiky "height graph"). The underlying data structure is
 * UNCHANGED -- still a flat per-cell list, still addressed by integer
 * grid indices -- only the visual presentation changes, in three
 * concrete steps, in order:
 *
 * 1. `fillUnsetHeights` -- a decimated/missing cell is filled from its
 *    populated neighbours' average, never left at a hard flat 0 next to
 *    a real sample (which is what turned isolated real cells into
 *    spikes in the first pass).
 * 2. `smoothHeights` -- a small box-blur pass over the WHOLE field. This
 *    is the actual fix for "looks like a bar chart": raw per-cell LiDAR
 *    height noise is real data, but presenting every single sample at
 *    full amplitude is what produced a jagged mountain range instead of
 *    rolling ground. A contiguous multi-cell feature (a trench, a
 *    vegetation mass) survives a gentle blur as a softened depression/
 *    rise; only single-cell noise gets smoothed away, which is exactly
 *    what should happen to noise.
 * 3. `fadeEdgesToZero` -- the outermost rows/columns of the grid are
 *    where decimation is heaviest (see mockData.ts's own "decimate
 *    coarser rings" comment) and therefore the least reliable; a hard,
 *    finite-extent grid also has no business pretending it knows the
 *    ground truth right up to its own boundary. Both are handled by
 *    fading height to 0 over the last few cells, giving the mesh a
 *    clean, deliberate boundary instead of a wall or a spike field.
 *
 * The returned `sampleHeight(gx, gy)` is the SAME final (filled +
 * smoothed + faded) height a caller would see baked into the mesh --
 * used by Scene.tsx so the planned path and the UGV marker sit flush on
 * the surface that is actually rendered, not on the raw noisy samples.
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
  /** Box-blur passes applied to the whole height field. Default 2 --
   * enough to turn single-cell noise into rolling ground while a
   * multi-cell feature (a trench, a rise) survives as a softened
   * version of itself. */
  smoothPasses?: number
  /** How many cells at the grid's outer edge fade linearly to height 0.
   * Default 5. */
  edgeFadeCells?: number
}

export interface Heightfield {
  geometry: THREE.BufferGeometry
  /** The exact height baked into the mesh at grid cell (gx, gy),
   * clamped to the field's own bounds. Use this for anything that must
   * sit ON the rendered surface (a path, a vehicle marker) rather than
   * re-deriving height from raw per-cell data, which would drift from
   * what is actually on screen once smoothing/fading are applied. */
  sampleHeight: (gx: number, gy: number) => number
  minHeight: number
}

function fillUnsetHeights(heights: Float32Array, set: Uint8Array, width: number, depth: number): void {
  const MAX_PASSES = 3
  const remaining = new Set<number>()
  for (let i = 0; i < set.length; i++) if (!set[i]) remaining.add(i)

  for (let pass = 0; pass < MAX_PASSES && remaining.size > 0; pass++) {
    const resolvedThisPass: [number, number][] = []
    for (const idx of remaining) {
      const ix = idx % width
      const iz = Math.floor(idx / width)
      let sum = 0
      let count = 0
      const neighbors: [number, number][] = [
        [ix - 1, iz], [ix + 1, iz], [ix, iz - 1], [ix, iz + 1],
      ]
      for (const [nx, nz] of neighbors) {
        if (nx < 0 || nx >= width || nz < 0 || nz >= depth) continue
        const nIdx = nz * width + nx
        if (set[nIdx]) {
          sum += heights[nIdx]
          count++
        }
      }
      if (count > 0) resolvedThisPass.push([idx, sum / count])
    }
    for (const [idx, avgHeight] of resolvedThisPass) {
      heights[idx] = avgHeight
      set[idx] = 1
      remaining.delete(idx)
    }
    if (resolvedThisPass.length === 0) break
  }
}

/** A separable 3x3 box blur, `passes` times, edge-clamped (a boundary
 * cell blurs with itself standing in for the missing neighbour, rather
 * than wrapping or darkening toward 0). */
function smoothHeights(
  heights: Float32Array<ArrayBufferLike>,
  width: number,
  depth: number,
  passes: number,
): Float32Array<ArrayBufferLike> {
  let src = heights
  for (let p = 0; p < passes; p++) {
    const dst = new Float32Array(src.length)
    for (let iz = 0; iz < depth; iz++) {
      for (let ix = 0; ix < width; ix++) {
        let sum = 0
        for (let dz = -1; dz <= 1; dz++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = Math.min(width - 1, Math.max(0, ix + dx))
            const nz = Math.min(depth - 1, Math.max(0, iz + dz))
            sum += src[nz * width + nx]
          }
        }
        dst[iz * width + ix] = sum / 9
      }
    }
    src = dst
  }
  return src
}

/** Linearly fades height to 0 over the outer `fadeCells` rings of the
 * grid, so the mesh always ends in a clean, flat boundary regardless of
 * what the (heaviest-decimated, least-reliable) edge samples said. */
function fadeEdgesToZero(heights: Float32Array, width: number, depth: number, fadeCells: number): void {
  if (fadeCells <= 0) return
  for (let iz = 0; iz < depth; iz++) {
    for (let ix = 0; ix < width; ix++) {
      const distToEdge = Math.min(ix, width - 1 - ix, iz, depth - 1 - iz)
      if (distToEdge >= fadeCells) continue
      const t = distToEdge / fadeCells // 0 at the very edge, 1 at fadeCells-in
      heights[iz * width + ix] *= t
    }
  }
}

/** Builds one continuous, smoothed, edge-faded heightfield covering
 * [minGx..maxGx] x [minGy..maxGy]. See this module's own doc comment
 * for the three-step process (fill -> smooth -> fade) that replaces the
 * earlier raw-heightfield approach. */
export function buildHeightfieldGeometry(cells: HeightCell[], opts: HeightfieldOptions): Heightfield {
  const smoothPasses = opts.smoothPasses ?? 2
  const edgeFadeCells = opts.edgeFadeCells ?? 5

  const width = opts.maxGx - opts.minGx + 1
  const depth = opts.maxGy - opts.minGy + 1
  const nVerts = width * depth

  let heights: Float32Array<ArrayBufferLike> = new Float32Array(nVerts)
  const colors = new Float32Array(nVerts * 3)
  const set = new Uint8Array(nVerts)
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
    set[idx] = 1
  }

  fillUnsetHeights(heights, set, width, depth)
  heights = smoothHeights(heights, width, depth, smoothPasses)
  fadeEdgesToZero(heights, width, depth, edgeFadeCells)

  const positions = new Float32Array(nVerts * 3)
  let minHeight = Infinity
  for (let iz = 0; iz < depth; iz++) {
    for (let ix = 0; ix < width; ix++) {
      const idx = iz * width + ix
      const worldX = (ix + opts.minGx + opts.centerOffset) * opts.cellSize
      const worldZ = (iz + opts.minGy + opts.centerOffset) * opts.cellSize
      positions[idx * 3] = worldX
      positions[idx * 3 + 1] = heights[idx]
      positions[idx * 3 + 2] = worldZ
      if (heights[idx] < minHeight) minHeight = heights[idx]
    }
  }
  if (!Number.isFinite(minHeight)) minHeight = 0

  const nQuads = (width - 1) * (depth - 1)
  const indices = new Uint32Array(Math.max(0, nQuads) * 6)
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

  const finalHeights = heights
  const sampleHeight = (gx: number, gy: number): number => {
    const ix = Math.min(width - 1, Math.max(0, Math.round(gx - opts.minGx)))
    const iz = Math.min(depth - 1, Math.max(0, Math.round(gy - opts.minGy)))
    return finalHeights[iz * width + ix]
  }

  return { geometry, sampleHeight, minHeight }
}
