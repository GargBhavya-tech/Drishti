/**
 * RealScene.tsx -- the REAL-data 3D view: real RELLIS-3D LiDAR points,
 * real FusionSegNet predictions, real multi-resolution grid, all from
 * eval/export_frames.py's exported binaries (see that script's own
 * docstring for exactly what "real" means here and its one honest
 * simplification: a self-contained per-sweep snapshot, not a live
 * temporally-accumulated map).
 *
 * Deliberately self-contained (its own play/pause/scrub state) rather
 * than wired into the existing Zustand store's frameIndex -- the mock
 * demo sequence (48 frames) and this real export (however many frames
 * were exported) are independent timelines with different lengths, and
 * unifying them was a materially bigger refactor than this pass's
 * scope. Toggled in from App.tsx as an alternate top-level view.
 */

import { Bloom, EffectComposer } from "@react-three/postprocessing"
import { OrbitControls } from "@react-three/drei"
import { Canvas } from "@react-three/fiber"
import { useEffect, useRef, useState } from "react"
import * as THREE from "three"
import type { RealFrame, RealManifest } from "../lib/realData"
import { loadFrame, loadManifest } from "../lib/realData"
import { DetectionMarkers } from "./DetectionMarkers"
import { RealPointCloud } from "./RealPointCloud"
import { RealTerrain } from "./RealTerrain"
import { VariableResGrid } from "./VariableResGrid"

function LoadingOverlay({ message }: { message: string }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-[#05070c]">
      <div className="text-slate-400 font-mono-tech text-sm">{message}</div>
    </div>
  )
}

function RealPlaybackBar({
  frameIndex,
  frameCount,
  isPlaying,
  onScrub,
  onTogglePlay,
}: {
  frameIndex: number
  frameCount: number
  isPlaying: boolean
  onScrub: (i: number) => void
  onTogglePlay: () => void
}) {
  return (
    <div className="absolute bottom-3 left-3 right-3 flex items-center gap-3 rounded-xl border border-white/10 bg-black/40 backdrop-blur-md px-4 py-2.5">
      <button
        onClick={onTogglePlay}
        className="h-8 w-8 flex items-center justify-center rounded-full border border-white/10 bg-white/5 text-slate-200"
      >
        {isPlaying ? "||" : ">"}
      </button>
      <input
        type="range"
        min={0}
        max={Math.max(0, frameCount - 1)}
        value={frameIndex}
        onChange={(e) => onScrub(Number(e.target.value))}
        className="flex-1 accent-cyan-400"
      />
      <span className="font-mono-tech text-xs text-slate-400 w-16 text-right">
        {String(frameIndex + 1).padStart(2, "0")}/{frameCount}
      </span>
    </div>
  )
}

function RealLegend({ manifest, detectionCount }: { manifest: RealManifest; detectionCount: number }) {
  return (
    <div className="absolute top-3 left-3 rounded-xl border border-white/10 bg-black/40 backdrop-blur-md px-3 py-2 max-w-xs">
      <div className="text-[11px] uppercase tracking-widest text-cyan-400/80 mb-1">Real data</div>
      <div className="text-[11px] text-slate-400 leading-relaxed">
        Real RELLIS-3D LiDAR + real FusionSegNet (checkpoint epoch {manifest.trainedEpoch}). Points, terrain, and
        resolution rings are computed from a real trained model on real off-road data -- not synthetic.
      </div>
      <div className="text-[11px] text-amber-300/90 mt-1">
        {detectionCount} real object detection{detectionCount === 1 ? "" : "s"} this frame (wireframe boxes, geometric detector)
      </div>
    </div>
  )
}

export function RealScene() {
  const [manifest, setManifest] = useState<RealManifest | null>(null)
  const [frame, setFrame] = useState<RealFrame | null>(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(true)
  const [error, setError] = useState<string | null>(null)
  // Toggles between this frame's own single-sweep cells and the real
  // multi-frame world-frame memory (export_frames.py's _WorldCellMemory)
  // -- off by default so the existing single-sweep view is unchanged
  // unless a viewer explicitly asks to see accumulation.
  const [useAccumulated, setUseAccumulated] = useState(false)
  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)

  useEffect(() => {
    loadManifest()
      .then(setManifest)
      .catch(() => setError("Could not load /data/manifest.json -- run eval/export_frames.py first."))
  }, [])

  useEffect(() => {
    if (!manifest) return
    let cancelled = false
    loadFrame(frameIndex).then((f) => {
      if (!cancelled) setFrame(f)
    })
    return () => {
      cancelled = true
    }
  }, [manifest, frameIndex])

  useEffect(() => {
    if (!isPlaying || !manifest) return
    const FRAME_MS = 260
    const tick = (t: number) => {
      if (t - lastTickRef.current >= FRAME_MS) {
        setFrameIndex((i) => (i + 1) % manifest.nFrames)
        lastTickRef.current = t
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [isPlaying, manifest])

  if (error) return <LoadingOverlay message={error} />
  if (!manifest) return <LoadingOverlay message="Loading real data manifest..." />
  if (!frame) return <LoadingOverlay message={`Loading frame ${frameIndex + 1}...`} />

  return (
    <div className="relative w-full h-full">
      <Canvas
        shadows={false}
        camera={{ position: [8, 7, 8], fov: 45 }}
        dpr={[1, 1.75]}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.1 }}
      >
        <color attach="background" args={["#05070c"]} />
        <fog attach="fog" args={["#05070c", 14, 45]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 16, 6]} intensity={1.2} />
        <pointLight position={[-8, 6, -8]} intensity={0.2} color="#4fd1ff" />

        <VariableResGrid levels={manifest.levels} />
        <RealTerrain frame={frame} levels={manifest.levels} useAccumulated={useAccumulated} />
        <RealPointCloud frame={frame} />
        <DetectionMarkers frame={frame} />

        <OrbitControls enableDamping dampingFactor={0.08} minDistance={2} maxDistance={35} maxPolarAngle={Math.PI / 2.05} />

        <EffectComposer multisampling={0}>
          <Bloom luminanceThreshold={0.7} luminanceSmoothing={0.2} intensity={0.35} mipmapBlur />
        </EffectComposer>
      </Canvas>
      <button
        onClick={() => setUseAccumulated((v) => !v)}
        className="absolute top-3 right-3 rounded-lg border border-white/10 bg-black/40 backdrop-blur-md px-3 py-1.5 text-[11px] font-mono-tech text-slate-200"
        title="Toggle between this frame's own single-sweep cells and the real multi-frame world-frame memory (export_frames.py's _WorldCellMemory)"
      >
        {useAccumulated ? "● Live memory (accumulated)" : "○ Single-sweep snapshot"}
      </button>
      <RealLegend manifest={manifest} detectionCount={frame.detectionCount} />
      <RealPlaybackBar
        frameIndex={frameIndex}
        frameCount={manifest.nFrames}
        isPlaying={isPlaying}
        onScrub={(i) => {
          setIsPlaying(false)
          setFrameIndex(i)
        }}
        onTogglePlay={() => setIsPlaying((p) => !p)}
      />
    </div>
  )
}
