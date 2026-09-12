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
import { PATH_COLOR, SURFACE } from "../lib/theme"
import { RealPointCloud } from "./RealPointCloud"
import { RealTerrain } from "./RealTerrain"
import { UgvModel } from "./UgvModel"
import { VariableResGrid } from "./VariableResGrid"

const UGV_REAL_SCALE = 3.5 // real-world scale (REAL_WORLD_SCALE) reads the UGV's 0.42m body too small at this camera distance otherwise

/** The UGV at the ego origin -- every exported real frame is in the
 * sensor/ego frame (see realData.ts's own doc comment), so the vehicle
 * is, by construction, always at (0, 0, 0), facing UgvModel's own
 * native +X-forward direction (this codebase's established sensor-
 * frame convention -- see RealPointCloud.tsx's own comment). */
function RealVehicleMarker() {
  return (
    <group position={[0, 0, 0]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
        <ringGeometry args={[0.42, 0.48, 32]} />
        <meshBasicMaterial color={PATH_COLOR} transparent opacity={0.7} side={THREE.DoubleSide} />
      </mesh>
      <group scale={UGV_REAL_SCALE}>
        <UgvModel />
      </group>
    </group>
  )
}

function LoadingOverlay({ message }: { message: string }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-[#1B2119]">
      <div className="text-[#96A3A8] text-sm max-w-sm text-center px-6">{message}</div>
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
    <div className="panel-glass absolute bottom-3 left-3 right-3 flex items-center gap-3 px-4 py-2.5">
      <button
        onClick={onTogglePlay}
        className="h-8 w-8 flex items-center justify-center rounded-full border border-white/10 bg-white/5 text-[#E7ECEE]"
      >
        {isPlaying ? "||" : ">"}
      </button>
      <input
        type="range"
        min={0}
        max={Math.max(0, frameCount - 1)}
        value={frameIndex}
        onChange={(e) => onScrub(Number(e.target.value))}
        className="flex-1 accent-[#55D6E8]"
      />
      <span className="font-mono-tech text-xs text-[#96A3A8] w-16 text-right">
        {String(frameIndex + 1).padStart(2, "0")}/{frameCount}
      </span>
    </div>
  )
}

function RealLegend({ manifest }: { manifest: RealManifest }) {
  return (
    <div className="panel-glass absolute top-3 left-3 px-3 py-2 max-w-xs">
      <div className="text-[11px] uppercase tracking-widest text-[#96A3A8] mb-1 font-medium">Live sensors</div>
      <div className="text-[11px] text-[#96A3A8] leading-relaxed">
        Real RELLIS-3D LiDAR + real FusionSegNet (checkpoint epoch {manifest.trainedEpoch}). Points, terrain, and
        resolution rings are computed from a real trained model on real off-road data -- not synthetic.
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
        camera={{ position: [5, 14, 5], fov: 36 }}
        dpr={[1, 1.75]}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.2 }}
      >
        <color attach="background" args={[SURFACE.appBg]} />
        <fogExp2 attach="fog" args={[SURFACE.appBg, 0.055]} />
        <hemisphereLight args={["#2a3540", "#0a0d10", 0.5]} />
        <ambientLight intensity={0.22} />
        <directionalLight position={[-6, 12, 7]} intensity={1.75} />

        <VariableResGrid levels={manifest.levels} />
        <RealTerrain frame={frame} levels={manifest.levels} />
        <RealPointCloud frame={frame} />
        <RealVehicleMarker />

        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          minDistance={6}
          maxDistance={22}
          minPolarAngle={0.26}
          maxPolarAngle={0.55}
        />

        <EffectComposer multisampling={0}>
          <Bloom luminanceThreshold={0.7} luminanceSmoothing={0.2} intensity={0.35} mipmapBlur />
        </EffectComposer>
      </Canvas>
      <RealLegend manifest={manifest} />
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
