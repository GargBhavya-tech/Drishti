/**
 * store.ts -- shared UI state (Zustand). Kept intentionally small:
 * playback position, which overlay layer is active, split-screen mode,
 * and the live gamma/velocity driving the fovea preview.
 */

import { create } from "zustand"
import { DEMO_SEQUENCE } from "../lib/mockData"

export type OverlayMode = "class" | "observability" | "sparsity" | "height" | "motion" | "attention"
export type CameraPreset = "driver" | "tactical" | "hazard"

interface DashboardState {
  frameIndex: number
  isPlaying: boolean
  overlayMode: OverlayMode
  splitScreen: boolean
  compareWipe: boolean
  realDataMode: boolean
  realFrameIndex: number
  realFrameCount: number
  realIsPlaying: boolean
  realUseAccumulated: boolean
  showRealTerrain: boolean
  showRealPoints: boolean
  showRealDetections: boolean
  showResolutionGrid: boolean
  cameraPreset: CameraPreset
  evidenceOpen: boolean
  gamma: number
  egoSpeedMs: number

  setFrameIndex: (i: number) => void
  stepFrame: (delta: number) => void
  togglePlaying: () => void
  setPlaying: (v: boolean) => void
  setOverlayMode: (m: OverlayMode) => void
  toggleSplitScreen: () => void
  toggleCompareWipe: () => void
  toggleRealDataMode: () => void
  setRealFrameIndex: (i: number) => void
  stepRealFrame: (delta: number) => void
  setRealFrameCount: (count: number) => void
  toggleRealPlaying: () => void
  setRealPlaying: (v: boolean) => void
  toggleRealAccumulated: () => void
  toggleRealTerrain: () => void
  toggleRealPoints: () => void
  toggleRealDetections: () => void
  toggleResolutionGrid: () => void
  setCameraPreset: (preset: CameraPreset) => void
  toggleEvidence: () => void
  setEvidenceOpen: (open: boolean) => void
  setGamma: (g: number) => void
  setEgoSpeedMs: (v: number) => void
}

const FRAME_COUNT = DEMO_SEQUENCE.length

export const useDashboardStore = create<DashboardState>((set) => ({
  frameIndex: 0,
  isPlaying: true,
  overlayMode: "class",
  splitScreen: false,
  compareWipe: false,
  realDataMode: false,
  realFrameIndex: 0,
  realFrameCount: 0,
  realIsPlaying: true,
  realUseAccumulated: false,
  showRealTerrain: true,
  showRealPoints: true,
  showRealDetections: true,
  showResolutionGrid: true,
  cameraPreset: "tactical",
  evidenceOpen: false,
  gamma: 1.0,
  egoSpeedMs: 15,

  setFrameIndex: (i) => set({ frameIndex: ((i % FRAME_COUNT) + FRAME_COUNT) % FRAME_COUNT }),
  stepFrame: (delta) =>
    set((s) => ({ frameIndex: ((s.frameIndex + delta) % FRAME_COUNT + FRAME_COUNT) % FRAME_COUNT })),
  togglePlaying: () => set((s) => ({ isPlaying: !s.isPlaying })),
  setPlaying: (v) => set({ isPlaying: v }),
  setOverlayMode: (m) => set({ overlayMode: m }),
  toggleSplitScreen: () => set((s) => ({ splitScreen: !s.splitScreen, compareWipe: false })),
  toggleCompareWipe: () => set((s) => ({ compareWipe: !s.compareWipe, splitScreen: false })),
  toggleRealDataMode: () => set((s) => ({ realDataMode: !s.realDataMode, splitScreen: false, compareWipe: false })),
  setRealFrameIndex: (i) =>
    set((s) => {
      const count = s.realFrameCount
      return { realFrameIndex: count > 0 ? ((i % count) + count) % count : 0 }
    }),
  stepRealFrame: (delta) =>
    set((s) => {
      const count = s.realFrameCount
      return { realFrameIndex: count > 0 ? ((s.realFrameIndex + delta) % count + count) % count : 0 }
    }),
  setRealFrameCount: (count) => set((s) => ({ realFrameCount: count, realFrameIndex: Math.min(s.realFrameIndex, Math.max(0, count - 1)) })),
  toggleRealPlaying: () => set((s) => ({ realIsPlaying: !s.realIsPlaying })),
  setRealPlaying: (v) => set({ realIsPlaying: v }),
  toggleRealAccumulated: () => set((s) => ({ realUseAccumulated: !s.realUseAccumulated })),
  toggleRealTerrain: () => set((s) => ({ showRealTerrain: !s.showRealTerrain })),
  toggleRealPoints: () => set((s) => ({ showRealPoints: !s.showRealPoints })),
  toggleRealDetections: () => set((s) => ({ showRealDetections: !s.showRealDetections })),
  toggleResolutionGrid: () => set((s) => ({ showResolutionGrid: !s.showResolutionGrid })),
  setCameraPreset: (cameraPreset) => set({ cameraPreset }),
  toggleEvidence: () => set((s) => ({ evidenceOpen: !s.evidenceOpen })),
  setEvidenceOpen: (evidenceOpen) => set({ evidenceOpen }),
  setGamma: (g) => set({ gamma: g }),
  setEgoSpeedMs: (v) => set({ egoSpeedMs: v }),
}))

export const FRAME_COUNT_EXPORT = FRAME_COUNT
