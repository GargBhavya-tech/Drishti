<script lang="ts">
	import { T } from '@threlte/core';
	import * as THREE from 'three';

	// A stylized, clearly-not-sensor-data environment layer -- muted, unlit,
	// low-poly. Corrected to match REAL RELLIS-3D terrain (Bible Part D.5:
	// off-road trail, ~85% grass/soil ground, heavy vegetation occlusion) --
	// no paved road, no fence, no urban traffic, none of which RELLIS-3D
	// actually contains. Dirt trail + scattered trees only.

	function makeDirtTexture(): THREE.CanvasTexture {
		const size = 512;
		const canvas = document.createElement('canvas');
		canvas.width = canvas.height = size;
		const ctx = canvas.getContext('2d')!;
		ctx.fillStyle = '#2b2620';
		ctx.fillRect(0, 0, size, size);
		// Mottled dirt/gravel speckle, not lane markings -- an off-road trail
		// has no painted lines.
		for (let i = 0; i < 900; i++) {
			const x = Math.random() * size;
			const y = Math.random() * size;
			const r = 1 + Math.random() * 2.5;
			const shade = 30 + Math.random() * 40;
			ctx.fillStyle = `rgba(${shade + 20},${shade + 12},${shade},0.5)`;
			ctx.beginPath();
			ctx.arc(x, y, r, 0, Math.PI * 2);
			ctx.fill();
		}
		const tex = new THREE.CanvasTexture(canvas);
		tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
		tex.repeat.set(3, 6);
		return tex;
	}

	const dirtTexture = makeDirtTexture();

	// A single low-poly tree: cylinder trunk + cone canopy.
	function treeAt(x: number, z: number, scale: number): [number, number, number] {
		return [x, 0, z];
	}

	// Scattered along both sides of the trail, denser near the vehicle's own
	// loop (radius ~0-16) -- a stand-in silhouette for RELLIS-3D's own
	// dominant ground truth (grass/tree/bush -> VEGETATION), never claimed
	// as real per-tree sensor detections.
	const TREES: { pos: [number, number, number]; scale: number }[] = Array.from(
		{ length: 34 },
		(_, i) => {
			const angle = (i / 34) * Math.PI * 2 + Math.sin(i) * 0.4;
			const radius = 11 + Math.random() * 9;
			return {
				pos: treeAt(7 + Math.cos(angle) * radius, Math.sin(angle) * radius, 1),
				scale: 0.7 + Math.random() * 0.8
			};
		}
	);
</script>

<!-- Dirt trail surface -->
<T.Mesh rotation.x={-Math.PI / 2} position={[7, -0.03, 3]}>
	<T.PlaneGeometry args={[46, 46]} />
	<T.MeshBasicMaterial map={dirtTexture} transparent opacity={0.6} />
</T.Mesh>

<!-- Trailside vegetation -->
{#each TREES as t, i (i)}
	<T.Group position={t.pos} scale={t.scale}>
		<T.Mesh position={[0, 0.6, 0]}>
			<T.CylinderGeometry args={[0.08, 0.12, 1.2, 6]} />
			<T.MeshBasicMaterial color="#2e241a" transparent opacity={0.75} />
		</T.Mesh>
		<T.Mesh position={[0, 1.6, 0]}>
			<T.ConeGeometry args={[0.9, 1.9, 7]} />
			<T.MeshBasicMaterial color="#233a24" transparent opacity={0.7} />
		</T.Mesh>
	</T.Group>
{/each}
