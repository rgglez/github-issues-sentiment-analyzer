<script lang="ts">
	// Load global styles and the application icon once at the root layout level.
	import './layout.css';
	import favicon from '$lib/assets/favicon.svg';

	// Receive SvelteKit's child-page render function for this layout slot.
	let { children } = $props();
</script>

<!-- Publish the shared favicon for every route. -->
<svelte:head><link rel="icon" href={favicon} /></svelte:head>

<!-- Render the active route inside the root layout. -->
{@render children()}
