import { CatmullRomCurve3, Vector3 } from 'three';

/**
 * A synthetic OPEN (non-looping) trail path -- an off-road trail doesn't
 * circle back on itself, and a visibly-circling vehicle reads as an obvious
 * animation loop rather than a real drive. Traversed ping-pong (forward to
 * the end, then back) via samplePath's own continuous-time mapping below,
 * so the vehicle drives the SAME real trail both ways rather than warping.
 */
const WAYPOINTS: Vector3[] = [
	[0, 0, 0],
	[3, 0, -3],
	[7, 0, -5],
	[11, 0, -4],
	[14, 0, -1],
	[15, 0, 4],
	[13, 0, 9],
	[9, 0, 11]
].map(([x, y, z]) => new Vector3(x, y, z));

export const vehiclePathCurve = new CatmullRomCurve3(WAYPOINTS, false, 'catmullrom', 0.5);

/** Triangle wave: 0->1->0 over each period-length span of continuous time. */
function pingPong(t: number, period: number): { u: number; forward: boolean } {
	const phase = ((t % period) + period) % period;
	const half = period / 2;
	return phase <= half ? { u: phase / half, forward: true } : { u: 2 - phase / half, forward: false };
}

export function samplePath(t: number, period = 26) {
	const { u, forward } = pingPong(t, period);
	const clamped = Math.min(Math.max(u, 0), 1);
	const position = vehiclePathCurve.getPointAt(clamped);
	const rawTangent = vehiclePathCurve.getTangentAt(clamped);
	const tangent = forward ? rawTangent : rawTangent.clone().multiplyScalar(-1);
	return { position, tangent };
}
