/**
 * realData.ts -- loads REAL exported RELLIS-3D + FusionSegNet data
 * (eval/export_frames.py) as raw binary, zero-parse-overhead typed
 * arrays -- `new Float32Array(buffer)` directly, no JSON. Binary
 * layout matches that script's own docstring exactly:
 *
 *   points_{i}.bin:            4 floats/point [x, y, z, classId]   (sensor frame, metres)
 *   cells_{i}.bin:              5 floats/cell  [level, gx, gy, heightM, classId] (single-sweep snapshot)
 *   accumulated_cells_{i}.bin:  SAME 5-float schema, but real multi-frame world-frame
 *                                memory reprojected into frame i's own sensor-local frame
 *                                (see export_frames.py's `_WorldCellMemory`) -- a cell
 *                                observed several frames ago but not touched since still
 *                                appears here, which is what makes this genuinely different
 *                                from `cells` (the map has real memory, not just this
 *                                frame's own fresh binning).
 *   detections_{i}.bin:         6 floats/detection [x, y, z, classId, footprintAreaM2, heightM]
 *                                (perception.geometric_instance_detector.detect_instances,
 *                                sensor frame)
 *
 * `accumulatedCells`/`detections` are fetched with a graceful fallback
 * to an empty array on 404 -- an OLDER export directory (generated
 * before these two files existed) still loads and renders exactly as
 * it did before this addition, just without those two layers.
 */

export interface LevelInfo {
  level: number
  cellSizeM: number
  nyquistRadiusM: number
}

export interface RealManifest {
  checkpointPath: string
  trainedEpoch: number
  sequenceDir: string
  rellisFrameIndices: number[]
  nFrames: number
  classNames: string[]
  levels: LevelInfo[]
  pointCounts: number[]
  cellCounts: number[]
  accumulatedCellCounts: number[]
  detectionCounts: number[]
  hasAccumulatedCells: boolean
  hasDetections: boolean
}

export interface RealFrame {
  frameIndex: number
  points: Float32Array // 4 floats/point
  cells: Float32Array // 5 floats/cell
  accumulatedCells: Float32Array // 5 floats/cell -- real multi-frame world-frame memory, see module docstring
  detections: Float32Array // 6 floats/detection
  pointCount: number
  cellCount: number
  accumulatedCellCount: number
  detectionCount: number
}

/** Fetches a binary file, returning an empty Float32Array on any
 * failure (404, network error) rather than throwing -- an older export
 * directory without accumulated_cells_*.bin / detections_*.bin still
 * loads cleanly, just with those two layers empty. */
async function _fetchOptionalBinary(url: string): Promise<Float32Array> {
  try {
    const r = await fetch(url)
    if (!r.ok) return new Float32Array(0)
    return new Float32Array(await r.arrayBuffer())
  } catch {
    return new Float32Array(0)
  }
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
          sequenceDir: raw.sequence_dir ?? "RELLIS-3D export",
          rellisFrameIndices: raw.rellis_frame_indices ?? [],
          nFrames: raw.n_frames,
          classNames: raw.class_names,
          levels: raw.levels.map((l: { level: number; cell_size_m: number; nyquist_radius_m: number }) => ({
            level: l.level,
            cellSizeM: l.cell_size_m,
            nyquistRadiusM: l.nyquist_radius_m,
          })),
          pointCounts: raw.point_counts,
          cellCounts: raw.cell_counts,
          accumulatedCellCounts: raw.accumulated_cell_counts ?? raw.cell_counts,
          detectionCounts: raw.detection_counts ?? [],
          hasAccumulatedCells: raw.has_accumulated_cells ?? false,
          hasDetections: raw.has_detections ?? false,
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
    _fetchOptionalBinary(`${baseUrl}/accumulated_cells_${idx}.bin`),
    _fetchOptionalBinary(`${baseUrl}/detections_${idx}.bin`),
  ]).then(([pointsBuf, cellsBuf, accumulatedCells, detections]) => {
    const points = new Float32Array(pointsBuf)
    const cells = new Float32Array(cellsBuf)
    const frame: RealFrame = {
      frameIndex: index,
      points,
      cells,
      accumulatedCells,
      detections,
      pointCount: points.length / 4,
      cellCount: cells.length / 5,
      accumulatedCellCount: accumulatedCells.length / 5,
      detectionCount: detections.length / 6,
    }
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
