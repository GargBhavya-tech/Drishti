import { storyState } from './activeBeat.svelte';

/**
 * Svelte action: marks `index` as the active beat whenever this element
 * crosses the vertical center band of the viewport. Uses IntersectionObserver
 * (native, no scrollytelling library needed) rather than tying anything to
 * scroll position directly -- the element's PRESENCE near center is the only
 * signal; how fast the user scrolled to get there is irrelevant.
 */
export function scrollBeat(node: HTMLElement, index: number) {
	const observer = new IntersectionObserver(
		(entries) => {
			for (const entry of entries) {
				if (entry.isIntersecting) {
					storyState.activeBeatIndex = index;
				}
			}
		},
		{ rootMargin: '-45% 0px -45% 0px', threshold: 0 }
	);
	observer.observe(node);
	return {
		destroy() {
			observer.disconnect();
		}
	};
}
