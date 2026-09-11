/**
 * pathPlanner.ts -- a direct TypeScript port of planning/path_planner.py's
 * A* (same algorithm: 8-connected, Euclidean admissible heuristic,
 * LETHAL cells never expanded, +1 step-cost floor so FREE cells still
 * cost real distance). Ported rather than called live, matching this
 * frontend's own established pattern (foveaMath.ts) of porting the
 * real backend formulas so the demo computes genuine numbers locally.
 *
 * Class -> cost mapping mirrors planning/conservatism.py's spirit:
 * NEGATIVE_OBSTACLE is LETHAL (never crossed), moving hazards and
 * confirmed obstacles are heavily penalised (still technically
 * crossable so A* never silently fails to find AN answer, but never
 * the cheapest one), UNKNOWN/CAUTION cost more than DRIVABLE.
 */

import type { DemoFrame } from "./mockData"

export type GridCell = [number, number] // [row, col]

export interface PathResult {
  path: GridCell[] | null
  totalCost: number
  cellsExpanded: number
}

const SQRT2 = Math.SQRT2
const NEIGHBOR_OFFSETS: [number, number, number][] = [
  [-1, -1, SQRT2], [-1, 0, 1], [-1, 1, SQRT2],
  [0, -1, 1], [0, 1, 1],
  [1, -1, SQRT2], [1, 0, 1], [1, 1, SQRT2],
]

class MinHeap {
  private items: { f: number; cell: GridCell }[] = []
  push(f: number, cell: GridCell) {
    this.items.push({ f, cell })
    this.items.sort((a, b) => a.f - b.f) // small grids here -- clarity over a real binary heap
  }
  pop() {
    return this.items.shift()
  }
  get length() {
    return this.items.length
  }
}

function heuristic(a: GridCell, b: GridCell): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1])
}

export function findPath(costGrid: Float64Array, h: number, w: number, start: GridCell, goal: GridCell): PathResult {
  const key = (c: GridCell) => c[0] * w + c[1]
  const inBounds = (r: number, c: number) => r >= 0 && r < h && c >= 0 && c < w

  const gScore = new Map<number, number>()
  const cameFrom = new Map<number, GridCell>()
  const visited = new Set<number>()
  gScore.set(key(start), 0)

  const open = new MinHeap()
  open.push(0, start)
  let cellsExpanded = 0

  while (open.length > 0) {
    const top = open.pop()
    if (!top) break
    const current = top.cell
    const currentKey = key(current)
    if (visited.has(currentKey)) continue
    visited.add(currentKey)
    cellsExpanded++

    if (current[0] === goal[0] && current[1] === goal[1]) {
      const path: GridCell[] = [current]
      let k = currentKey
      while (cameFrom.has(k)) {
        const prev = cameFrom.get(k)!
        path.push(prev)
        k = key(prev)
      }
      path.reverse()
      return { path, totalCost: gScore.get(currentKey) ?? Infinity, cellsExpanded }
    }

    for (const [di, dj, dist] of NEIGHBOR_OFFSETS) {
      const ni = current[0] + di
      const nj = current[1] + dj
      if (!inBounds(ni, nj)) continue
      const neighbor: GridCell = [ni, nj]
      const nKey = key(neighbor)
      if (visited.has(nKey)) continue
      const cost = costGrid[ni * w + nj]
      if (!Number.isFinite(cost)) continue // LETHAL -- a hard wall

      const stepCost = dist * (1 + cost)
      const tentativeG = (gScore.get(currentKey) ?? Infinity) + stepCost
      if (tentativeG < (gScore.get(nKey) ?? Infinity)) {
        gScore.set(nKey, tentativeG)
        cameFrom.set(nKey, current)
        open.push(tentativeG + heuristic(neighbor, goal), neighbor)
      }
    }
  }

  return { path: null, totalCost: Infinity, cellsExpanded }
}

const LETHAL = Infinity
const CLASS_COST: Record<number, number> = {
  0: 40, // UNKNOWN -- cautious, matches UNKNOWN_COST's spirit
  1: 0, // DRIVABLE
  2: 12, // CAUTION
  3: 60, // NON_TRAVERSABLE
  4: 200, // STATIC_OBSTACLE
  5: 3, // VEGETATION
  6: 200, // VEHICLE
  7: 150, // PEDESTRIAN (also used for the moving hazard cell)
  8: LETHAL, // NEGATIVE_OBSTACLE
  9: 80, // OVERHANG
}

/** Build a (rows x cols) cost grid from one demo frame's sparse cell
 * list, covering the full [-half, half] extent -- cells with no data
 * this frame (decimated, per Ticket #51's own display convention) get
 * UNKNOWN's own cost, never a silent zero. */
export function costGridFromFrame(frame: DemoFrame, half: number): { grid: Float64Array; size: number } {
  const size = half * 2 + 1
  const grid = new Float64Array(size * size).fill(CLASS_COST[0])
  for (const cell of frame.cells) {
    const r = cell.i + half
    const c = cell.j + half
    if (r < 0 || r >= size || c < 0 || c >= size) continue
    grid[r * size + c] = CLASS_COST[cell.classId] ?? CLASS_COST[0]
  }
  return { grid, size }
}

/** The SAME dense grid costGridFromFrame builds, but of raw DRISHTI
 * class ids (UNKNOWN=0 for a cell with no data this frame) rather than
 * cost -- used by the kinodynamic path-smoothing pass to look up "what
 * terrain is under this point on the curve" at the SAME indexing
 * costGridFromFrame already established, so both consumers agree on
 * cell (row, col) <-> (i, j) mapping by construction, not by
 * convention two functions have to independently get right. */
export function classGridFromFrame(frame: DemoFrame, half: number): { classGrid: Int16Array; size: number } {
  const size = half * 2 + 1
  const classGrid = new Int16Array(size * size).fill(0) // 0 = UNKNOWN
  for (const cell of frame.cells) {
    const r = cell.i + half
    const c = cell.j + half
    if (r < 0 || r >= size || c < 0 || c >= size) continue
    classGrid[r * size + c] = cell.classId
  }
  return { classGrid, size }
}
