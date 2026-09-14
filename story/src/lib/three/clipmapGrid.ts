import { Color, DynamicDrawUsage, InstancedMesh, MeshBasicMaterial, Object3D, PlaneGeometry } from 'three';

/**
 * The real 3D -> 2.5D conversion (Bible Layers 4-5), visualized directly
 * from real loaded point data: bins points into cells whose SIZE is derived
 * from each point's own range via the real sensor-schedule formula
 * (sensor/schedule.py's c(r) = c0 * 2^ceil(log2(r*d_theta/c0))) -- fine near
 * the vehicle, coarser far away. Each cell stores the min/max height of the
 * real points landing in it, the multi-layer-cell idea (Part C.9) simplified
 * to one layer for this visual. Real Ouster OS1-64 d_theta (measured from
 * real data, Bible Part D.1), not the datasheet placeholder.
 */

export const MAX_CELLS = 6000;
const D_THETA_RAD = (0.17578125 * Math.PI) / 180;
const C0 = 0.05;

export function cellSizeForRange(r: number): number {
	if (r < 0.15) return C0;
	const level = Math.max(0, Math.ceil(Math.log2((r * D_THETA_RAD) / C0)));
	return C0 * Math.pow(2, Math.min(level, 6));
}

export function createClipmapGrid(): InstancedMesh {
	const geometry = new PlaneGeometry(1, 1);
	geometry.rotateX(-Math.PI / 2);
	const material = new MeshBasicMaterial({ transparent: true, opacity: 0.82 });
	const mesh = new InstancedMesh(geometry, material, MAX_CELLS);
	mesh.instanceMatrix.setUsage(DynamicDrawUsage);
	mesh.count = 0;
	mesh.frustumCulled = false;
	return mesh;
}

export function buildGridFromPoints(grid: InstancedMesh, positions: ArrayLike<number>, count: number): number {
	const cells = new Map<
		string,
		{ x: number; z: number; size: number; hMin: number; hMax: number }
	>();

	for (let i = 0; i < count; i++) {
		const x = positions[i * 3];
		const y = positions[i * 3 + 1];
		const z = positions[i * 3 + 2];
		const r = Math.sqrt(x * x + z * z);
		if (r < 0.3 || r > 20) continue; // skip the sensor mount itself and far sparse returns
		const size = cellSizeForRange(r);
		const cx = Math.floor(x / size);
		const cz = Math.floor(z / size);
		const key = `${cx}_${cz}_${size.toFixed(3)}`;
		let cell = cells.get(key);
		if (!cell) {
			if (cells.size >= MAX_CELLS) continue;
			cell = { x: (cx + 0.5) * size, z: (cz + 0.5) * size, size, hMin: y, hMax: y };
			cells.set(key, cell);
		}
		if (y < cell.hMin) cell.hMin = y;
		if (y > cell.hMax) cell.hMax = y;
	}

	const dummy = new Object3D();
	const color = new Color();
	let i = 0;
	for (const cell of cells.values()) {
		if (i >= MAX_CELLS) break;
		dummy.position.set(cell.x, (cell.hMin + cell.hMax) / 2, cell.z);
		dummy.scale.set(cell.size * 0.92, cell.size * 0.92, 1);
		dummy.updateMatrix();
		grid.setMatrixAt(i, dummy.matrix);

		const clearance = cell.hMax - cell.hMin;
		const t = Math.min(clearance / 1.5, 1);
		color.setRGB(0.15 + t * 0.6, 0.45 - t * 0.25, 0.55 - t * 0.35);
		grid.setColorAt(i, color);
		i++;
	}

	grid.count = i;
	grid.instanceMatrix.needsUpdate = true;
	if (grid.instanceColor) grid.instanceColor.needsUpdate = true;
	return i;
}
