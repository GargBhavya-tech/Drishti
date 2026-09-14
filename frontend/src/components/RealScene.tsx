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
import { loadFrame, loadManifest } from "../lib/realData"
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
  const [frame, setFrame] = useState<RealFrame | null>(null)
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
    loadFrame(realFrameIndex)
      .then((loadedFrame) => {
        if (!cancelled) setFrame(loadedFrame)
      })
      .catch(() => {
        if (!cancelled) setError(`Could not load exported frame ${realFrameIndex + 1}.`)
      })
    return () => {
      cancelled = true
    }
  }, [manifest, realFrameIndex])

  useEffect(() => {
    if (!manifest || !frame || frame.frameIndex !== realFrameIndex) {
      onStatusChange?.(null)
      return
    }
    onStatusChange?.({ manifest, frame, frameIndex: realFrameIndex, useAccumulated: realUseAccumulated })
  }, [frame, manifest, onStatusChange, realFrameIndex, realUseAccumulated])

  useEffect(() => () => onStatusChange?.(null), [onStatusChange])

  if (error) return <LoadingOverlay message={error} error />
  if (!manifest) return <LoadingOverlay message="Loading real-data manifest…" />
  if (!frame || frame.frameIndex !== realFrameIndex) return <LoadingOverlay message={`Loading export frame ${realFrameIndex + 1}…`} />

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
        <RealLegend manifest={manifest} frameIndex={realFrameIndex} detectionCount={frame.detectionCount} />
        <ResolutionLegend levels={manifest.levels} />
      </div>
      <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border border-white/10 bg-[#09101b]/75 px-2.5 py-1.5 font-mono-tech text-[10px] text-slate-400 backdrop-blur-md">
        Drag to orbit · scroll to zoom · camera presets above
      </div>
    </div>
  )
}
