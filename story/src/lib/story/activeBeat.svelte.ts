/**
 * Shared reactive story state (Svelte 5 universal reactivity via a .svelte.ts
 * module). Scroll position -> active beat index is the ONLY thing the DOM
 * writes here; the 3D scene only ever reads it -- this is the "DOM triggers
 * state, canvas reacts to state" decoupling from the scroll-triggered-
 * autoplay research, kept in one small shared object rather than prop-drilled.
 */
export const storyState = $state({
	activeBeatIndex: 0
});
