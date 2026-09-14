/**
 * Renders the locally exported RELLIS-3D run. The scene has no
 * invented predictions: it shows only labels and instances present in
 * the export, then reports its exact metadata to the dashboard shell.
 */

import { OrbitControls } from "@react-three/drei"
import { Bloom, EffectComposer } from "@react-three/postprocessing"
import { Canvas, useThree } from "@react-three/fiber"
import { useEffect, useState } from "react"
import * as THREE from "three"
import type { RealFrame, RealManifest } from "../lib/realData"
import { loadAccumulatedCells, loadFrame, loadManifest } from "../lib/realData"
import { useDashboardStore } from "../state/store"
import { DetectionMarkers } from "./DetectionMarkers"
import { RealPointCloud } from "./RealPointCloud"
import { RealTerrain } from "./RealTerrain"
import { ResolutionLegend } from "./ResolutionLegend"
import { VariableResGrid } from "./VariableResGrid"

export interface RealSceneStatus {
  manifest: RealManifest
  frame: RealFrame
  frameIndex: number
  useAccumulated: boolean
}

function LoadingOverlay({ message, error = false }: { message: string; error?: boolean }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-[#07101a]" role={error ? "alert" : "status"} aria-live="polite">
      <div className="max-w-sm rounded-xl border border-white/10 bg-white/[0.035] px-5 py-4 text-center shadow-xl">
        <p className={`font-mono-tech text-xs ${error ? "text-amber-200" : "text-slate-300"}`}>{message}</p>
        {error && <p className="mt-2 text-[11px] leading-4 text-slate-500">Check that the frontend export is available at <code>/data/manifest.json</code>, then re-open real export mode.</p>}
      </div>
    </div>
  )
}

function CameraPresetController() {
  const cameraPreset = useDashboardStore((s) => s.cameraPreset)
  const camera = useThree((state) => state.camera)

  useEffect(() => {
    const presets = {
      driver: { position: [12, 3.2, 0] as const, target: [0, 0.35, 0] as const },
      tactical: { position: [8, 10, 8] as const, target: [0, 0, 0] as const },
      hazard: { position: [5.5, 3.5, -7] as const, target: [0.8, 0.25, 0] as const },
    }
    const preset = presets[cameraPreset]
    camera.position.set(preset.position[0], preset.position[1], preset.position[2])
    camera.lookAt(preset.target[0], preset.target[1], preset.target[2])
  }, [camera, cameraPreset])

  return null
}

function RealLegend({ manifest, frameIndex, detectionCount }: { manifest: RealManifest; frameIndex: number; detectionCount: number }) {
  const sourceFrame = manifest.rellisFrameIndices[frameIndex]
  return (
    <div className="max-w-sm rounded-xl border border-white/10 bg-[#09101b]/80 px-3 py-2.5 shadow-lg backdrop-blur-md">
      <p className="text-[9px] font-semibold uppercase tracking-[0.16em] text-cyan-300/85">Measured export</p>
      <p className="mt-1 text-[11px] leading-4 text-slate-300">
        RELLIS-3D {sourceFrame === undefined ? `export frame ${frameIndex + 1}` : `frame ${sourceFrame}`} · epoch {manifest.trainedEpoch} · semantic points + 2.5D cells
      </p>
      <p className="mt-1 text-[10px] text-amber-100/90">{detectionCount} geometric object detection{detectionCount === 1 ? "" : "s"} in this frame</p>
    </div>
  )
}

export function RealScene({ onStatusChange }: { onStatusChange?: (status: RealSceneStatus | null) => void }) {
  const [manifest, setManifest] = useState<RealManifest | null>(null)
  // `frame` holds the LAST SUCCESSFULLY LOADED frame, which can lag
  // behind `realFrameIndex` while the next one is still fetching --
  // deliberately NOT cleared on every frame-index change (it used to
  // be, which blanked the whole 3D view to a "Loading export frame N..."
  // message every time playback's fixed-interval timer (RealTimeline,
  // Controls.tsx) outran a frame's network fetch -- a real, reported
  // "it takes time to load and see stuff" bug, not a one-off). The
  // scene now keeps rendering whatever it last had while the next frame
  // loads in the background, exactly like a video player holds its last
  // decoded frame during a network stall instead of going black.
  const [frame, setFrame] = useState<RealFrame | null>(null)
  const [isFetchingFrame, setIsFetchingFrame] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const realFrameIndex = useDashboardStore((s) => s.realFrameIndex)
  const realUseAccumulated = useDashboardStore((s) => s.realUseAccumulated)
  const showRealTerrain = useDashboardStore((s) => s.showRealTerrain)
  const showRealPoints = useDashboardStore((s) => s.showRealPoints)
  const showRealDetections = useDashboardStore((s) => s.showRealDetections)
  const showResolutionGrid = useDashboardStore((s) => s.showResolutionGrid)
  const setRealFrameCount = useDashboardStore((s) => s.setRealFrameCount)

  useEffect(() => {
    let mounted = true
    loadManifest()
      .then((loadedManifest) => {
        if (!mounted) return
        setManifest(loadedManifest)
        setRealFrameCount(loadedManifest.nFrames)
      })
      .catch(() => {
        if (mounted) setError("Could not load the real-data manifest.")
      })
    return () => {
      mounted = false
    }
  }, [setRealFrameCount])

  useEffect(() => {
    if (!manifest) return
    let cancelled = false
    setIsFetchingFrame(true)
    loadFrame(realFrameIndex)
      .then((loadedFrame) => {
        if (cancelled) return
        setFrame(loadedFrame)
        setIsFetchingFrame(false)
      })
      .catch(() => {
        if (cancelled) return
        // A failed fetch only surfaces as a hard error when there is no
        // prior frame to keep showing -- otherwise keep the last good
        // frame on screen and just stop the (silent) fetching indicator,
        // matching this effect's own "hold the last frame" design.
        setIsFetchingFrame(false)
        setFrame((prev) => {
          if (!prev) setError(`Could not load exported frame ${realFrameIndex + 1}.`)
          return prev
        })
      })
    return () => {
      cancelled = true
    }
  }, [manifest, realFrameIndex])

  // accumulatedCells is fetched lazily (realData.ts's own doc comment
  // explains why: ~30% of a frame's payload, only needed once the user
  // turns accumulation on). Merge it into `frame` once it arrives,
  // guarded by frameIndex so a fetch that resolves after the user has
  // already scrubbed to a different frame doesn't clobber it.
  useEffect(() => {
    if (!realUseAccumulated || !frame || frame.accumulatedCellCount > 0) return
    const targetIndex = frame.frameIndex
    let cancelled = false
    loadAccumulatedCells(targetIndex).then((accumulatedCells) => {
      if (cancelled) return
      setFrame((prev) => (prev && prev.frameIndex === targetIndex ? { ...prev, accumulatedCells, accumulatedCellCount: accumulatedCells.length / 5 } : prev))
    })
    return () => {
      cancelled = true
    }
  }, [realUseAccumulated, frame])

  // Reports the LAST LOADED frame's status even while a newer one is
  // still fetching -- `frameIndex` in the reported status is the frame
  // actually being shown (frame.frameIndex), not necessarily
  // `realFrameIndex`, so consumers (DecisionStack, RunStatusBar) stay in
  // sync with what's genuinely on screen instead of flashing to a null/
  // loading state on every playback tick.
  useEffect(() => {
    if (!manifest || !frame) {
      onStatusChange?.(null)
      return
    }
    onStatusChange?.({ manifest, frame, frameIndex: frame.frameIndex, useAccumulated: realUseAccumulated })
  }, [frame, manifest, onStatusChange, realUseAccumulated])

  useEffect(() => () => onStatusChange?.(null), [onStatusChange])

  if (error) return <LoadingOverlay message={error} error />
  if (!manifest) return <LoadingOverlay message="Loading real-data manifest…" />
  // Only the very FIRST load (nothing has ever loaded yet) blanks the
  // scene -- once any frame has loaded, later frame changes keep it on
  // screen (see the frame-loading effect's own comment) rather than
  // repeatedly blanking to this overlay.
  if (!frame) return <LoadingOverlay message={`Loading export frame ${realFrameIndex + 1}…`} />

  return (
    <div className="relative h-full w-full">
      <Canvas
        shadows={false}
        camera={{ position: [8, 10, 8], fov: 45 }}
        dpr={[1, 1.75]}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.1 }}
      >
        <color attach="background" args={["#07101a"]} />
        <fog attach="fog" args={["#07101a", 14, 45]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 16, 6]} intensity={1.2} />
        <pointLight position={[-8, 6, -8]} intensity={0.2} color="#4fd1ff" />

        {showResolutionGrid && <VariableResGrid levels={manifest.levels} />}
        {showRealTerrain && <RealTerrain frame={frame} levels={manifest.levels} useAccumulated={realUseAccumulated} />}
        {showRealPoints && <RealPointCloud frame={frame} />}
        {showRealDetections && <DetectionMarkers frame={frame} />}
        <CameraPresetController />
        <OrbitControls enableDamping dampingFactor={0.08} minDistance={2} maxDistance={35} maxPolarAngle={Math.PI / 2.05} />

        <EffectComposer multisampling={0}>
          <Bloom luminanceThreshold={0.7} luminanceSmoothing={0.2} intensity={0.35} mipmapBlur />
        </EffectComposer>
      </Canvas>

      <div className="pointer-events-none absolute left-3 top-3 flex max-w-[calc(100%-1.5rem)] flex-col gap-2 sm:max-w-sm">
        {/* frame.frameIndex, not realFrameIndex -- shows the frame actually
            on screen, which can lag the scrubber position by one frame
            while the next one fetches (see the frame-loading effect). */}
        <RealLegend manifest={manifest} frameIndex={frame.frameIndex} detectionCount={frame.detectionCount} />
        <ResolutionLegend levels={manifest.levels} />
      </div>
      {isFetchingFrame && (
        <div className="pointer-events-none absolute right-3 bottom-3 flex items-center gap-1.5 rounded-md border border-white/10 bg-[#09101b]/80 px-2.5 py-1.5 font-mono-tech text-[10px] text-slate-400 backdrop-blur-md">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-300" aria-hidden="true" />
          Loading next frame…
        </div>
      )}
      <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border border-white/10 bg-[#09101b]/75 px-2.5 py-1.5 font-mono-tech text-[10px] text-slate-400 backdrop-blur-md">
        Drag to orbit · scroll to zoom · camera presets above
      </div>
    </div>
  )
}
