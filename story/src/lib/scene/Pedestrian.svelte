<script lang="ts">
	import { T } from '@threlte/core';

	interface Props {
		position: [number, number, number];
		detected?: boolean;
	}
	let { position, detected = false }: Props = $props();

	// PEDESTRIAN is DRISHTI's highest-protection-priority class (Bible Part
	// C.6, taxonomy id 7) -- rendered here as a simple capsule proxy that
	// switches from a muted "undetected" grey to a bright, high-confidence
	// highlight the moment the ego vehicle's own detection radius reaches it.
</script>

<T.Group {position}>
	<T.Mesh position={[0, 0.75, 0]}>
		<T.CapsuleGeometry args={[0.22, 0.9, 4, 8]} />
		<T.MeshBasicMaterial color={detected ? '#ffcf40' : '#5a5f66'} transparent opacity={0.95} />
	</T.Mesh>
	<T.Mesh position={[0, 1.5, 0]}>
		<T.SphereGeometry args={[0.16, 10, 10]} />
		<T.MeshBasicMaterial color={detected ? '#ffcf40' : '#5a5f66'} transparent opacity={0.95} />
	</T.Mesh>
	{#if detected}
		<T.Mesh position={[0, 0.75, 0]}>
			<T.CylinderGeometry args={[0.6, 0.6, 1.8, 16, 1, true]} />
			<T.MeshBasicMaterial color="#ffcf40" wireframe transparent opacity={0.35} />
		</T.Mesh>
	{/if}
</T.Group>
