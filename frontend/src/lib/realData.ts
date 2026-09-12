/**
 * realData.ts -- loads REAL exported RELLIS-3D + FusionSegNet data
 * (eval/export_frames.py) as raw binary, zero-parse-overhead typed
 * arrays -- `new Float32Array(buffer)` directly, no JSON. Binary
 * layout matches that script's own docstring exactly:
 *
 *   points_{i}.bin: 4 floats/point [x, y, z, classId]   (sensor frame, metres)
 *   cells_{i}.bin:  5 floats/cell  [level, gx, gy, heightM, classId]
 *
 * See export_frames.py's own module docstring for the honest scope
 * note: each frame is a self-contained single-sweep snapshot (real
 * geometry, real predictions, real resolution-schedule binning), not a
 * live temporally-accumulated clipmap replay.
 */

export interface LevelInfo {
  level: number
  cellSizeM: number
  nyquistRadiusM: number
}

export interface RealManifest {
  checkpointPath: string
  trainedEpoch: number
  nFrames: number
  classNames: string[]
  levels: LevelInfo[]
  pointCounts: number[]
  cellCounts: number[]
}

export interface RealFrame {
  frameIndex: number
  points: Float32Array // 4 floats/point
  cells: Float32Array // 5 floats/cell
  pointCount: number
  cellCount: number
}

const DEFAULT_BASE_URL = "/data"

let _manifestPromise: Promise<RealManifest> | null = null

export function loadManifest(baseUrl: string = DEFAULT_BASE_URL): Promise<RealManifest> {
  if (!_manifestPromise) {
    _manifestPromise = fetch(`${baseUrl}/manifest.json`)
      .then((r) => r.json())
      .then(
        (raw): RealManifest => ({
          checkpointPath: raw.checkpoint_path,
          trainedEpoch: raw.trained_epoch,
          nFrames: raw.n_frames,
          classNames: raw.class_names,
          levels: raw.levels.map((l: { level: number; cell_size_m: number; nyquist_radius_m: number }) => ({
            level: l.level,
            cellSizeM: l.cell_size_m,
            nyquistRadiusM: l.nyquist_radius_m,
          })),
          pointCounts: raw.point_counts,
          cellCounts: raw.cell_counts,
        }),
      )
  }
  return _manifestPromise
}

const _frameCache = new Map<number, RealFrame>()
const _framePromises = new Map<number, Promise<RealFrame>>()

export function loadFrame(index: number, baseUrl: string = DEFAULT_BASE_URL): Promise<RealFrame> {
  const cached = _frameCache.get(index)
  if (cached) return Promise.resolve(cached)
  const pending = _framePromises.get(index)
  if (pending) return pending

  const idx = String(index).padStart(3, "0")
  const promise = Promise.all([
    fetch(`${baseUrl}/points_${idx}.bin`).then((r) => r.arrayBuffer()),
    fetch(`${baseUrl}/cells_${idx}.bin`).then((r) => r.arrayBuffer()),
  ]).then(([pointsBuf, cellsBuf]) => {
    const points = new Float32Array(pointsBuf)
    const cells = new Float32Array(cellsBuf)
    const frame: RealFrame = { frameIndex: index, points, cells, pointCount: points.length / 4, cellCount: cells.length / 5 }
    _frameCache.set(index, frame)
    _framePromises.delete(index)
    return frame
  })
  _framePromises.set(index, promise)
  return promise
}

/** Kicks off a fetch for every frame in the manifest without waiting --
 * call once on mount so scrubbing the timeline doesn't stall on
 * network. Individual `loadFrame` calls elsewhere share the same
 * cache/in-flight-promise map, so this never double-fetches. */
export async function preloadAllFrames(baseUrl: string = DEFAULT_BASE_URL): Promise<void> {
  const manifest = await loadManifest(baseUrl)
  await Promise.all(Array.from({ length: manifest.nFrames }, (_, i) => loadFrame(i, baseUrl)))
}
