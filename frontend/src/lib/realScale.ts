/**
 * realScale.ts -- the world-unit scale shared by every REAL-data
 * renderer (RealPointCloud, RealTerrain, VariableResGrid). Separate
 * from mockData/Scene.tsx's own CELL_WORLD_SIZE (0.32): the mock demo's
 * grid indices only ever span +-34 cells, but real exported points
 * span real metres out to ~130m (the coarsest resolution level's own
 * Nyquist radius) -- a materially larger physical extent that needs
 * its own, smaller scale to sit comfortably inside the same camera/
 * orbit-control setup (OrbitControls maxDistance=40 in Scene.tsx).
 */
export const REAL_WORLD_SCALE = 0.15
