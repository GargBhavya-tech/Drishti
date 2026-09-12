/**
 * store.ts -- shared UI state (Zustand). Kept intentionally small:
 * playback position, which overlay layer is active, split-screen mode,
 * and the live gamma/velocity driving the fovea preview.
 */

import { create } from "zustand"
import { DEMO_SEQUENCE } from "../lib/mockData"

export type OverlayMode = "class" | "observability" | "sparsity" | "height" | "motion" | "attention"

interface DashboardState {
  frameIndex: number
  isPlaying: boolean
  overlayMode: OverlayMode
  splitScreen: boolean
  compareWipe: boolean
  realDataMode: boolean
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
  toggleRealDataMode: () => set((s) => ({ realDataMode: !s.realDataMode })),
  setGamma: (g) => set({ gamma: g }),
  setEgoSpeedMs: (v) => set({ egoSpeedMs: v }),
}))

export const FRAME_COUNT_EXPORT = FRAME_COUNT
