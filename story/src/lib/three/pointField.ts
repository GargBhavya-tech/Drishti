import {
	BufferGeometry,
	CanvasTexture,
	Float32BufferAttribute,
	NormalBlending,
	Points,
	ShaderMaterial,
	Vector3
} from 'three';

/**
 * Ceiling for a single frame's point count. Buffers below are allocated
 * ONCE at this size and reused for the life of the app -- mutate in place,
 * restrict the draw range, flag needsUpdate. This is the exact pattern
 * real per-frame LiDAR playback needs (build order step 2); establishing
 * it now with synthetic data means the streaming code later is a drop-in.
 */
export const MAX_POINTS = 140_000;

function makeSpriteTexture(): CanvasTexture {
	const size = 64;
	const canvas = document.createElement('canvas');
	canvas.width = canvas.height = size;
	const ctx = canvas.getContext('2d')!;
	const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
	grad.addColorStop(0, 'rgba(255,255,255,1)');
	grad.addColorStop(0.4, 'rgba(255,255,255,0.55)');
	grad.addColorStop(1, 'rgba(255,255,255,0)');
	ctx.fillStyle = grad;
	ctx.fillRect(0, 0, size, size);
	return new CanvasTexture(canvas);
}

export function createPointField(): Points {
	const positions = new Float32Array(MAX_POINTS * 3);
	const colors = new Float32Array(MAX_POINTS * 3);
	const alphas = new Float32Array(MAX_POINTS).fill(1);

	const geometry = new BufferGeometry();
	geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
	geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
	geometry.setAttribute('alpha', new Float32BufferAttribute(alphas, 1));
	geometry.setDrawRange(0, 0);

	const material = new ShaderMaterial({
		uniforms: {
			pointTexture: { value: makeSpriteTexture() },
			size: { value: 2.2 },
			opacity: { value: 0.85 },
			// Classification-pulse (Bible-adjacent "trick" beat): a shockwave
			// expanding from pulseCenter, radius = pulseElapsed * speed.
			// pulseElapsed < 0 means inactive -- no pulse contribution at all.
			pulseCenter: { value: new Vector3(0, 0, 0) },
			pulseElapsed: { value: -999 }
		},
		transparent: true,
		depthWrite: false,
		blending: NormalBlending,
		vertexShader: `
			attribute vec3 color;
			attribute float alpha;
			varying vec3 vColor;
			varying float vAlpha;
			varying float vPulse;
			uniform float size;
			uniform vec3 pulseCenter;
			uniform float pulseElapsed;
			void main() {
				vColor = color;
				vAlpha = alpha;

				vPulse = 0.0;
				if (pulseElapsed >= 0.0) {
					float d = length(position - pulseCenter);
					float front = pulseElapsed * 9.0;
					float diff = abs(d - front);
					vPulse = smoothstep(2.2, 0.0, diff) * exp(-pulseElapsed * 0.5);
				}

				vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
				gl_PointSize = size * (120.0 / -mvPosition.z) * (1.0 + vPulse * 1.8);
				gl_Position = projectionMatrix * mvPosition;
			}
		`,
		fragmentShader: `
			uniform sampler2D pointTexture;
			uniform float opacity;
			varying vec3 vColor;
			varying float vAlpha;
			varying float vPulse;
			void main() {
				vec4 tex = texture2D(pointTexture, gl_PointCoord);
				if (tex.a < 0.05) discard;
				vec3 finalColor = vColor + vec3(1.0, 0.92, 0.6) * vPulse * 1.3;
				gl_FragColor = vec4(finalColor, tex.a * opacity * vAlpha);
			}
		`
	});

	const points = new Points(geometry, material);
	points.frustumCulled = false;
	return points;
}

/** Updates the classification-pulse uniforms every frame (see Scene.svelte). */
export function setPulseUniforms(points: Points, center: Vector3, elapsedSincePulse: number) {
	const material = points.material as ShaderMaterial;
	material.uniforms.pulseCenter.value.copy(center);
	material.uniforms.pulseElapsed.value = elapsedSincePulse;
}

/**
 * Carves a real sparsity void into the loaded point buffer -- points within
 * `radius` of `center` are hidden (alpha -> 0) to stand in for a real
 * negative-obstacle dropout (Bible Part C.10/C.11: a ditch doesn't emit a
 * "hazard" signal, it emits an absence of returns). This project's own eval
 * scripts (eval/checkpoint_trench.png) already establish carving a synthetic
 * hazard into a sweep as the honest way to demo this without a hand-picked
 * real negative-obstacle frame. `active=false` restores full density.
 */
export function setVoidCarve(points: Points, center: Vector3, radius: number, active: boolean) {
	const geometry = points.geometry;
	const posAttr = geometry.getAttribute('position') as Float32BufferAttribute;
	const alphaAttr = geometry.getAttribute('alpha') as Float32BufferAttribute;
	const count = geometry.drawRange.count;
	const r2 = radius * radius;

	for (let i = 0; i < count; i++) {
		if (!active) {
			alphaAttr.setX(i, 1);
			continue;
		}
		const dx = posAttr.getX(i) - center.x;
		const dy = posAttr.getY(i) - center.y;
		const dz = posAttr.getZ(i) - center.z;
		if (dx * dx + dy * dy + dz * dz < r2) {
			alphaAttr.setX(i, 0);
		}
	}
	alphaAttr.needsUpdate = true;
}

/**
 * Synthetic scatter around the origin so the scene isn't empty in step 1.
 * Replaced by real exported LiDAR frames in step 2 -- this function has
 * the exact shape a real per-frame updater will have (mutate + drawRange).
 */
export function fillSyntheticScatter(points: Points, count: number) {
	const geometry = points.geometry;
	const posAttr = geometry.getAttribute('position') as Float32BufferAttribute;
	const colorAttr = geometry.getAttribute('color') as Float32BufferAttribute;
	const n = Math.min(count, MAX_POINTS);

	for (let i = 0; i < n; i++) {
		const angle = Math.random() * Math.PI * 2;
		const radius = 3 + Math.random() * 55;
		const x = Math.cos(angle) * radius + (Math.random() - 0.5) * 10;
		const z = Math.sin(angle) * radius + (Math.random() - 0.5) * 10;
		// Mostly ground-hugging returns, with occasional taller clutter --
		// a stand-in silhouette for "mostly flat ground, a few obstacles"
		// until real per-point height (from a real sweep) replaces this.
		const isTall = Math.random() < 0.06;
		const height = isTall ? 1 + Math.random() * 3.5 : Math.random() * 0.25;
		posAttr.setXYZ(i, x, height, z);

		const t = Math.min(height / 3, 1);
		// Dim teal ground rising to a warmer, brighter tone for tall clutter --
		// avoids uniform brightness so overlapping points don't wash to white.
		colorAttr.setXYZ(i, 0.08 + t * 0.55, 0.35 + t * 0.4, 0.42 + t * 0.3);
	}

	posAttr.needsUpdate = true;
	colorAttr.needsUpdate = true;
	geometry.setDrawRange(0, n);
	geometry.computeBoundingSphere();
}

/**
 * Loads a real exported RELLIS-3D frame (raw interleaved float32 x,y,z,intensity,
 * written by eval/export_story_frames.py) and fills the shared point buffer with
 * it -- the exact mutate-in-place + drawRange pattern fillSyntheticScatter used,
 * now with real sensor data instead of a procedural stand-in.
 *
 * KITTI-format RELLIS-3D .bin frames are a fixed-size range-image dump
 * (64 x 2048 = 131,072 slots); azimuth columns with no real return are
 * zero-padded (x=y=z=0). Those are filtered out here rather than rendered
 * as a false cluster of points at the sensor origin.
 */
export async function loadRealFrame(points: Points, binUrl: string): Promise<number> {
	const buf = await fetch(binUrl).then((r) => r.arrayBuffer());
	const data = new Float32Array(buf);
	const totalSlots = data.length / 4;

	const geometry = points.geometry;
	const posAttr = geometry.getAttribute('position') as Float32BufferAttribute;
	const colorAttr = geometry.getAttribute('color') as Float32BufferAttribute;

	let n = 0;
	for (let i = 0; i < totalSlots && n < MAX_POINTS; i++) {
		const x = data[i * 4];
		const y = data[i * 4 + 1];
		const z = data[i * 4 + 2];
		const intensity = data[i * 4 + 3];

		// Zero-padded "no return" slot -- not a real point at the origin.
		if (x === 0 && y === 0 && z === 0) continue;

		// RELLIS-3D sensor frame: x forward, y left, z up. Re-map to the
		// Three.js scene's convention used elsewhere in this app (y up).
		posAttr.setXYZ(n, x, z, -y);

		const t = Math.min(Math.max(intensity, 0), 1);
		colorAttr.setXYZ(n, 0.1 + t * 0.5, 0.4 + t * 0.4, 0.5 + t * 0.3);
		n++;
	}

	posAttr.needsUpdate = true;
	colorAttr.needsUpdate = true;
	geometry.setDrawRange(0, n);
	geometry.computeBoundingSphere();
	return n;
}
