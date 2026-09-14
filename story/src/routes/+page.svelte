<script lang="ts">
	import { Canvas } from '@threlte/core';
	import Scene from '$lib/scene/Scene.svelte';
	import { BEATS } from '$lib/story/beats';
	import { scrollBeat } from '$lib/story/scrollBeat';
	import { storyState } from '$lib/story/activeBeat.svelte';
	import { hudState } from '$lib/story/hud.svelte';
</script>

<div class="canvas-layer">
	<Canvas renderMode="always" autoRender={false}>
		<Scene activeBeatIndex={storyState.activeBeatIndex} />
	</Canvas>

	{#if hudState.visible}
		<svg class="hud-svg">
			<line
				x1={hudState.screenX}
				y1={hudState.screenY}
				x2={hudState.cardX + 90}
				y2={hudState.cardY - 70}
				stroke="rgba(255,180,120,0.85)"
				stroke-width="1.5"
			/>
			<circle cx={hudState.screenX} cy={hudState.screenY} r="4" fill="rgba(255,180,120,0.9)" />
		</svg>
		<div class="hud-card" style:left="{hudState.cardX + 96}px" style:top="{hudState.cardY - 118}px">
			<p class="hud-label">HAZARD RANGE</p>
			<p class="hud-value">{hudState.distanceM.toFixed(1)} m</p>
			<p class="hud-label">SAFE SPEED</p>
			<p class="hud-value hud-speed">{hudState.speedKmh.toFixed(0)} km/h</p>
		</div>
	{/if}
</div>

<main class="scroll-layer">
	<section class="intro">
		<div class="intro-glow"></div>
		<div class="intro-content">
			<p class="intro-kicker">A real off-road LiDAR perception system</p>
			<h1>DRISHTI</h1>
			<p class="intro-sub">
				Distance-Resolved Instantaneous Semantic Height &amp; Traversability Imaging
			</p>
			<p class="intro-scroll">scroll to begin ↓</p>
		</div>
	</section>

	{#each BEATS as beat, i (beat.id)}
		<section class="beat" use:scrollBeat={i} class:active={storyState.activeBeatIndex === i}>
			<div class="card">
				<p class="eyebrow">{beat.eyebrow}</p>
				<h2>{beat.title}</h2>
				<p class="body">{beat.body}</p>
			</div>
		</section>
	{/each}
</main>

<style>
	:global(html, body) {
		margin: 0;
		height: 100%;
		background: #05070a;
		color: #eef4f8;
		font-family: 'Inter', system-ui, sans-serif;
	}

	.canvas-layer {
		position: fixed;
		inset: 0;
		width: 100vw;
		height: 100vh;
		z-index: 0;
	}

	.hud-svg {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
		z-index: 2;
	}

	.hud-card {
		position: absolute;
		pointer-events: none;
		z-index: 2;
		padding: 0.6rem 0.9rem;
		background: rgba(8, 12, 16, 0.72);
		border: 1px solid rgba(255, 180, 120, 0.5);
		border-radius: 3px;
		font-family: 'Courier New', monospace;
		white-space: nowrap;
	}

	.hud-label {
		margin: 0;
		font-size: 0.62rem;
		letter-spacing: 0.08em;
		color: rgba(255, 180, 120, 0.8);
	}

	.hud-value {
		margin: 0 0 0.4rem;
		font-size: 1.15rem;
		color: #fff;
	}

	.hud-speed {
		margin-bottom: 0;
		color: #ffdd55;
	}

	.scroll-layer {
		position: relative;
		z-index: 1;
	}

	.intro {
		min-height: 100vh;
		display: flex;
		align-items: center;
		justify-content: center;
		text-align: center;
		position: relative;
		overflow: hidden;
	}

	.intro-glow {
		position: absolute;
		inset: -20%;
		background: radial-gradient(circle at center, rgba(90, 170, 255, 0.16), transparent 60%);
		animation: intro-breathe 6s ease-in-out infinite;
		pointer-events: none;
	}

	@keyframes intro-breathe {
		0%,
		100% {
			opacity: 0.55;
			transform: scale(1);
		}
		50% {
			opacity: 1;
			transform: scale(1.08);
		}
	}

	.intro-content {
		position: relative;
		z-index: 1;
		padding: 2rem;
	}

	.intro-kicker {
		margin: 0 0 0.75rem;
		font-size: 0.85rem;
		letter-spacing: 0.14em;
		text-transform: uppercase;
		color: #7fd0ff;
	}

	.intro-content h1 {
		margin: 0;
		font-size: clamp(3rem, 10vw, 6rem);
		font-weight: 800;
		letter-spacing: 0.06em;
		background: linear-gradient(180deg, #fff, #9fd6ff);
		-webkit-background-clip: text;
		background-clip: text;
		color: transparent;
	}

	.intro-sub {
		margin: 1rem auto 0;
		max-width: 32rem;
		color: #a7b7c2;
		font-size: 0.95rem;
	}

	.intro-scroll {
		margin-top: 3rem;
		font-size: 0.8rem;
		letter-spacing: 0.08em;
		color: #5f7280;
		animation: intro-bob 2s ease-in-out infinite;
	}

	@keyframes intro-bob {
		0%,
		100% {
			transform: translateY(0);
			opacity: 0.7;
		}
		50% {
			transform: translateY(6px);
			opacity: 1;
		}
	}

	.beat {
		min-height: 100vh;
		display: flex;
		align-items: center;
		padding: 0 8vw;
		box-sizing: border-box;
	}

	.card {
		max-width: 30rem;
		padding: 1.75rem 2rem;
		background: rgba(5, 8, 12, 0.6);
		backdrop-filter: blur(6px);
		border-left: 2px solid rgba(120, 200, 255, 0.35);
		border-radius: 4px;
		opacity: 0.55;
		transform: translateY(6px);
		transition:
			opacity 0.4s ease,
			transform 0.4s ease,
			border-color 0.4s ease;
	}

	.beat.active .card {
		opacity: 1;
		transform: translateY(0);
		border-color: rgba(120, 200, 255, 0.9);
	}

	.eyebrow {
		margin: 0 0 0.5rem;
		font-size: 0.8rem;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: #7fd0ff;
	}

	h2 {
		margin: 0 0 0.75rem;
		font-size: 1.9rem;
		line-height: 1.2;
	}

	.body {
		margin: 0;
		font-size: 1.02rem;
		line-height: 1.55;
		color: #c7d4dc;
	}
</style>
