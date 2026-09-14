<script lang="ts">
	import { T, useTask, useThrelte } from '@threlte/core';
	import * as THREE from 'three';
	import { samplePath, vehiclePathCurve } from '$lib/config/vehiclePath';
	import { createPathLine, updatePathLine, speedEnvelopeKmh } from '$lib/three/pathLine';
	import { hudState } from '$lib/story/hud.svelte';
	import {
		createPointField,
		fillSyntheticScatter,
		loadRealFrame,
		setPulseUniforms,
		setVoidCarve
	} from '$lib/three/pointField';
	import { createClipmapGrid, buildGridFromPoints } from '$lib/three/clipmapGrid';
	import HazardMarker from './HazardMarker.svelte';
	import EnvironmentContext from './EnvironmentContext.svelte';
	import Pedestrian from './Pedestrian.svelte';

	interface Props {
		activeBeatIndex?: number;
	}
	let { activeBeatIndex = 0 }: Props = $props();

	// Beat order: 0 baseline, 1 the-grid, 2 complication, 3 trick, 4 proof,
	// 5 limitation (beats.ts). Per-beat vehicle tint + chase distance proves
	// scroll state reaches the 3D scene; richer per-beat content (grid,
	// pulse, void, HUD) is layered on top of this same index.
	const BEAT_VEHICLE_COLOR = ['#ff5533', '#4fa8ff', '#ffb020', '#ff3060', '#30d0a0', '#8090a0'];
	const BEAT_CHASE_DISTANCE = [6, 8, 5, 4.5, 5.5, 8];

	const { renderer, scene, size, renderStage } = useThrelte();

	// Step 2: a real exported RELLIS-3D frame (data/rellis/00000, frame 50)
	// replaces the step 1 synthetic scatter. Falls back to synthetic only
	// if the real export is missing (e.g. before running the exporter).
	const pointField = createPointField();
	fillSyntheticScatter(pointField, 4_000);
	const clipmapGrid = createClipmapGrid();
	let usingRealFrame = $state(false);
	loadRealFrame(pointField, '/data/baseline_frame.bin')
		.then((n) => {
			usingRealFrame = true;
			console.log(`[story] loaded real frame: ${n} points`);
			const positions = pointField.geometry.getAttribute('position').array as Float32Array;
			const cellCount = buildGridFromPoints(clipmapGrid, positions, n);
			console.log(`[story] built 2.5D grid: ${cellCount} real cells`);
		})
		.catch((err) => {
			console.warn('[story] real frame not found, keeping synthetic scatter', err);
		});

	let chaseCam: THREE.PerspectiveCamera | undefined = $state();
	let topCam: THREE.OrthographicCamera | undefined = $state();
	let vehicle: THREE.Group | undefined = $state();

	const camPos = new THREE.Vector3(0, 6, 14);
	const lookAtPos = new THREE.Vector3();
	let initialized = false;

	const LOOP_SECONDS = 24;
	let elapsed = 0;

	// The "trick" beat's hazard: a real sparsity void carved into the real
	// point cloud (build order step 4), placed along the vehicle's own loop
	// so the marker and the pulse are both grounded in real geometry, not
	// floating in empty space.
	const HAZARD_CENTER = new THREE.Vector3(9, 0.05, -6);
	const HAZARD_RADIUS = 2.2;

	// A detected pedestrian along the trail (PEDESTRIAN is DRISHTI's highest
	// protection-priority class, Bible Part C.6) -- the ego vehicle visibly
	// swerves to a different lateral position on the SAME real driven path
	// once it's within avoidance range, not just a rendered line change.
	const PEDESTRIAN_POS = new THREE.Vector3(13, 0, 8);
	const PEDESTRIAN_DETECT_RADIUS = 8;
	const PEDESTRIAN_AVOID_RADIUS = 5;
	let pedestrianDetected = $state(false);

	let carveActive = false;
	let pulseStartTime = -9999;
	let lastPulseBeat = -1;
	let markerPulse = $state(0);
	let bendStrength = 0;

	const pathLine = createPathLine();
	const projected = new THREE.Vector3();

	$effect(() => {
		const shouldCarve = activeBeatIndex >= 2;
		if (shouldCarve !== carveActive) {
			setVoidCarve(pointField, HAZARD_CENTER, HAZARD_RADIUS, shouldCarve);
			carveActive = shouldCarve;
		}
		if (activeBeatIndex === 3 && lastPulseBeat !== 3) {
			pulseStartTime = elapsed;
		}
		lastPulseBeat = activeBeatIndex;
	});

	// Drive the vehicle along the smoothed path and damp the chase camera
	// toward it -- exponential decay (frame-rate independent), never a raw
	// snap-to-position, per the camera-choreography research.
	useTask((delta) => {
		elapsed += delta;
		const { position, tangent } = samplePath(elapsed, LOOP_SECONDS);

		// Pedestrian avoidance: within range, the vehicle's REAL driven
		// position shifts laterally (perpendicular to travel direction) --
		// a genuine lane-change on the same recorded trail, not a cosmetic
		// line redraw. Distance-graded, so it eases in/out rather than snapping.
		const toPed = new THREE.Vector3(position.x - PEDESTRIAN_POS.x, 0, position.z - PEDESTRIAN_POS.z);
		const distToPed = toPed.length();
		pedestrianDetected = distToPed < PEDESTRIAN_DETECT_RADIUS;
		const drivenPosition = position.clone();
		if (distToPed < PEDESTRIAN_AVOID_RADIUS && distToPed > 1e-4) {
			const side = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize();
			const amount = (1 - distToPed / PEDESTRIAN_AVOID_RADIUS) * 2.6;
			drivenPosition.x += side.x * amount;
			drivenPosition.z += side.z * amount;
		}

		if (vehicle) {
			vehicle.position.copy(drivenPosition);
			vehicle.position.y = 0.4;
			const lookTarget = drivenPosition.clone().add(tangent);
			vehicle.lookAt(lookTarget.x, 0.4, lookTarget.z);
		}

		const dist = BEAT_CHASE_DISTANCE[activeBeatIndex] ?? 6;
		const behind = tangent.clone().setY(0).normalize().multiplyScalar(-dist);
		const desiredCamPos = drivenPosition
			.clone()
			.add(behind)
			.add(new THREE.Vector3(0, dist * 0.58, 0));

		if (!initialized) {
			camPos.copy(desiredCamPos);
			lookAtPos.copy(drivenPosition);
			initialized = true;
		} else {
			const posDamp = 1 - Math.pow(0.0001, delta);
			const lookDamp = 1 - Math.pow(0.00005, delta);
			camPos.lerp(desiredCamPos, posDamp);
			lookAtPos.lerp(drivenPosition, lookDamp);
		}

		if (chaseCam) {
			chaseCam.position.copy(camPos);
			chaseCam.lookAt(lookAtPos);
		}

		if (topCam && vehicle) {
			topCam.position.set(vehicle.position.x, 60, vehicle.position.z);
			topCam.lookAt(vehicle.position.x, 0, vehicle.position.z);
		}

		setPulseUniforms(pointField, HAZARD_CENTER, elapsed - pulseStartTime);
		markerPulse = 0.5 + 0.5 * Math.sin(elapsed * 4);

		// Dim the raw points during "The 2.5D map" beat so the real cell
		// conversion (clipmapGrid) reads clearly instead of competing with them.
		(pointField.material as THREE.ShaderMaterial).uniforms.opacity.value =
			activeBeatIndex === 1 ? 0.28 : 0.85;

		// "Proof" beat (index 3): the rendered path visibly bends around the
		// hazard, and the HUD speed number is the REAL speed-envelope formula
		// evaluated at the vehicle's current real distance to the hazard --
		// both driven by the same underlying geometry every frame.
		const bendTarget = activeBeatIndex === 4 ? 1 : 0;
		bendStrength += (bendTarget - bendStrength) * Math.min(1, delta * 2.5);
		updatePathLine(pathLine, vehiclePathCurve, HAZARD_CENTER, 6, bendStrength);

		if (vehicle && chaseCam) {
			const distanceToHazard = vehicle.position.distanceTo(
				new THREE.Vector3(HAZARD_CENTER.x, vehicle.position.y, HAZARD_CENTER.z)
			);
			hudState.speedKmh = speedEnvelopeKmh(distanceToHazard);
			hudState.distanceM = distanceToHazard;
			hudState.visible = activeBeatIndex >= 3;

			projected.copy(HAZARD_CENTER).project(chaseCam);
			const w = size.current.width;
			const h = size.current.height;
			hudState.screenX = (projected.x * 0.5 + 0.5) * w;
			hudState.screenY = (-projected.y * 0.5 + 0.5) * h;
			// Card position is clamped separately from the dot/line so the
			// leader-line still points at the real screen position while the
			// ~180x140px card itself never runs off the viewport edge.
			hudState.cardX = Math.min(Math.max(hudState.screenX, 20), Math.max(w - 190, 20));
			hudState.cardY = Math.min(Math.max(hudState.screenY, 130), Math.max(h - 30, 130));
			const onScreen = projected.z < 1 && Math.abs(projected.x) < 0.92 && Math.abs(projected.y) < 0.92;
			hudState.visible = hudState.visible && onScreen;
		}
	});

	// Manual split-screen render: chase view fills the canvas, a top-down
	// tactical minimap renders into a scissored inset -- one renderer, two
	// passes, per the split-screen architecture from the research.
	useTask(
		() => {
			const w = size.current.width;
			const h = size.current.height;
			if (!w || !h || !chaseCam || !topCam) return;

			renderer.setScissorTest(true);

			renderer.setViewport(0, 0, w, h);
			renderer.setScissor(0, 0, w, h);
			renderer.render(scene, chaseCam);

			const mapSize = Math.min(w, h) * 0.28;
			const pad = 16;
			const mapX = w - mapSize - pad;
			const mapY = pad;
			renderer.setViewport(mapX, mapY, mapSize, mapSize);
			renderer.setScissor(mapX, mapY, mapSize, mapSize);
			renderer.clearDepth();
			renderer.render(scene, topCam);

			renderer.setScissorTest(false);
			renderer.setViewport(0, 0, w, h);
		},
		{ stage: renderStage }
	);

	const aspect = $derived(
		size.current.width && size.current.height ? size.current.width / size.current.height : 1
	);
</script>

<T.PerspectiveCamera bind:ref={chaseCam} args={[50, aspect, 0.1, 500]} position={[0, 6, 14]} />

<T.OrthographicCamera
	bind:ref={topCam}
	manual
	args={[-30, 30, 30, -30, 0.1, 200]}
	position={[0, 60, 0]}
	up={[0, 0, -1]}
/>

<EnvironmentContext />

<T.AmbientLight intensity={0.6} />
<T.DirectionalLight position={[20, 30, 10]} intensity={1.2} />

<T.Group bind:ref={vehicle}>
	<T.Mesh position={[0, 0, 0.6]}>
		<T.BoxGeometry args={[1.6, 0.8, 3]} />
		<T.MeshStandardMaterial color={BEAT_VEHICLE_COLOR[activeBeatIndex] ?? '#ff5533'} />
	</T.Mesh>
</T.Group>

<T is={pathLine} />

<T is={pointField} />

<T is={clipmapGrid} visible={activeBeatIndex === 1} />

<HazardMarker
	position={[HAZARD_CENTER.x, HAZARD_CENTER.y, HAZARD_CENTER.z]}
	visible={activeBeatIndex >= 2}
	pulse={markerPulse}
/>

<Pedestrian
	position={[PEDESTRIAN_POS.x, PEDESTRIAN_POS.y, PEDESTRIAN_POS.z]}
	detected={pedestrianDetected}
/>
