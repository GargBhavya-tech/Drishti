<script lang="ts">
	import { T } from '@threlte/core';

	interface Props {
		position: [number, number, number];
		visible?: boolean;
		pulse?: number;
	}
	let { position, visible = true, pulse = 0 }: Props = $props();
</script>

{#if visible}
	<T.Group {position}>
		<!-- Void-fill: gives the negative-obstacle void a tangible, visible form
			 (Bible Part C.10) instead of leaving it as an unremarkable gap in the
			 point cloud a first-time viewer would never notice. -->
		<T.Mesh>
			<T.BoxGeometry args={[2.6, 1.2, 2.6]} />
			<T.MeshBasicMaterial color="#ff3b3b" wireframe transparent opacity={0.5 + pulse * 0.45} />
		</T.Mesh>
		<T.Mesh position={[0, 0.05, 0]}>
			<T.SphereGeometry args={[0.3, 12, 12]} />
			<T.MeshBasicMaterial color="#ffdd55" transparent opacity={0.4 + pulse * 0.6} />
		</T.Mesh>
	</T.Group>
{/if}
