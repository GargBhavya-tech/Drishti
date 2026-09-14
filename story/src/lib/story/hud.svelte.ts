/**
 * Shared reactive HUD state, written every frame by Scene.svelte (which owns
 * the chase camera and can project the hazard's 3D position to 2D screen
 * space) and read by +page.svelte (which renders the HTML/SVG leader-line
 * and numeric readouts OUTSIDE the Canvas). This is the "2D reacts to 3D"
 * mechanism from the research -- the leader line and the speed number both
 * come from the SAME underlying real geometry every frame.
 */
export const hudState = $state({
	visible: false,
	screenX: 0,
	screenY: 0,
	cardX: 0,
	cardY: 0,
	speedKmh: 0,
	distanceM: 0
});
