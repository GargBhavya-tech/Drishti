import {
	BufferGeometry,
	CatmullRomCurve3,
	Float32BufferAttribute,
	Line,
	LineBasicMaterial,
	Vector3
} from 'three';

export function createPathLine(): Line {
	const geometry = new BufferGeometry();
	const samples = 160;
	geometry.setAttribute('position', new Float32BufferAttribute(new Float32Array(samples * 3), 3));
	geometry.setAttribute('color', new Float32BufferAttribute(new Float32Array(samples * 3), 3));
	const material = new LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 });
	const line = new Line(geometry, material);
	line.frustumCulled = false;
	return line;
}

/**
 * Direct port of planning/speed_envelope.py's v_max(R): the fastest speed at
 * which stopping distance still fits inside detection range R (meters).
 * a=4 m/s^2 braking, t_react=0.3s -- the same constants used throughout the
 * Bible's own worked examples (Part C.15).
 */
export function speedEnvelopeKmh(rangeMeters: number): number {
	const a = 4;
	const tReact = 0.3;
	const vMs = -a * tReact + Math.sqrt(a * a * tReact * tReact + 2 * a * Math.max(rangeMeters, 0));
	return Math.max(vMs, 0) * 3.6;
}

/**
 * Updates the path line's geometry every frame: samples the base curve,
 * laterally pushes points away from `hazardCenter` within `influenceRadius`
 * by up to `bendStrength` (0 = the real recorded path, 1 = fully dodging --
 * the "Proof" beat's path-bend), and colors each point by the real
 * perception-limited speed envelope for that point's own distance to the
 * hazard (red = slow, green = fast) -- a real formula reacting to real
 * geometry, not a scripted color ramp.
 */
export function updatePathLine(
	line: Line,
	curve: CatmullRomCurve3,
	hazardCenter: Vector3,
	influenceRadius: number,
	bendStrength: number
) {
	const geometry = line.geometry;
	const posAttr = geometry.getAttribute('position') as Float32BufferAttribute;
	const colorAttr = geometry.getAttribute('color') as Float32BufferAttribute;
	const samples = posAttr.count;
	const toHazard = new Vector3();

	for (let i = 0; i < samples; i++) {
		const u = i / (samples - 1);
		const p = curve.getPointAt(u);

		toHazard.subVectors(p, hazardCenter);
		toHazard.y = 0;
		const dist = toHazard.length();

		let x = p.x;
		let z = p.z;
		if (dist < influenceRadius && dist > 1e-4) {
			const push = (1 - dist / influenceRadius) * bendStrength * 3.0;
			const awayX = toHazard.x / dist;
			const awayZ = toHazard.z / dist;
			x += awayX * push;
			z += awayZ * push;
		}
		posAttr.setXYZ(i, x, 0.15, z);

		const speed = speedEnvelopeKmh(dist);
		const t = Math.min(speed / 45, 1); // 45 km/h treated as "fast/clear" for this color ramp
		colorAttr.setXYZ(i, 1 - t, 0.25 + t * 0.6, 0.25);
	}

	posAttr.needsUpdate = true;
	colorAttr.needsUpdate = true;
	geometry.computeBoundingSphere();
}
