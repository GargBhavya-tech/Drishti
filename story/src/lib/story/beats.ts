export interface Beat {
	id: string;
	eyebrow: string;
	title: string;
	body: string;
}

/**
 * Narrative structure: Baseline -> Complication -> Trick -> Proof -> Limitation,
 * per the deep-research pitch structure. Each beat's 3D behavior is wired
 * incrementally (build order steps 4-5 add the classification-pulse shader,
 * HUD leader-lines, and real path/speed reaction) -- for now the scene just
 * proves it can react to which beat is active.
 */
export const BEATS: Beat[] = [
	{
		id: 'baseline',
		eyebrow: '01 — Baseline',
		title: 'This is real sensor data.',
		body: 'Every point on screen came from a real Ouster OS1-64 sweep, recorded on an off-road trail. Nothing here is simulated.'
	},
	{
		id: 'the-grid',
		eyebrow: '02 — The 2.5D map',
		title: '3D becomes 2.5D.',
		body: 'Raw 3D is too much data to store; flat 2D throws away height. DRISHTI folds the real point cloud into a grid of cells, each remembering a height range — fine near the vehicle, coarser far away, following the sensor’s own resolution schedule.'
	},
	{
		id: 'complication',
		eyebrow: '03 — The complication',
		title: 'Absence is not evidence.',
		body: 'A ditch does not send back a "there is a ditch" signal — it sends back nothing. A flat 2D occupancy grid reads that silence as "already looked, nothing there."'
	},
	{
		id: 'trick',
		eyebrow: '04 — The trick',
		title: 'The Sparsity Trap.',
		body: 'DRISHTI computes how many returns a sensor at this range should see. When far fewer come back than physics predicts, it flags the gap as a hazard — not free space.'
	},
	{
		id: 'proof',
		eyebrow: '05 — The proof',
		title: 'The vehicle reacts.',
		body: 'The moment a hazard is confirmed, the path bends around it and the speed envelope drops — computed live from detection range and stopping distance, not scripted.'
	},
	{
		id: 'limitation',
		eyebrow: '06 — Honest limits',
		title: "What this can't do yet.",
		body: 'A 5cm cable is undetectable past ~6.7m. Thin static obstacles remain the single hardest class after six separate attempts. We are stating this, not hiding it.'
	}
];
