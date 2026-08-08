<script lang="ts">
	import { onMount } from 'svelte';

	/** Names of the sentiment models that the UI and API support. */
	type Model = 'svm' | 'bayes' | 'logistic';

	/** Sentiment labels returned by every trained classifier. */
	type Sentiment = 'negative' | 'neutral' | 'positive';

	/** One analyzed GitHub issue displayed as a kanban card. */
	interface Issue {
		title: string;
		body: string;
		url: string;
		prediction: Sentiment;
		confidence: number;
	}

	/** Shape of successful and error responses from the issues API. */
	interface ApiResponse {
		issues?: Issue[];
		hasMore?: boolean;
		cache?: string;
		error?: string;
	}

	/** Fixed-position dimensions that align a preview with its destination column. */
	interface PreviewPosition {
		left: number;
		width: number;
	}

	// Populate the model selector with stable values understood by the API.
	const models: Array<{ value: Model; label: string; description: string }> = [
		{ value: 'svm', label: 'SVM', description: 'Support vector machine' },
		{ value: 'bayes', label: 'Bayes', description: 'Probabilistic classifier' },
		{ value: 'logistic', label: 'Logistic regression', description: 'Linear classifier' }
	];

	// Map classifier labels to the human-readable metadata used by each kanban column.
	const sentiment = {
		negative: { emoji: '😠', label: 'Negative' },
		positive: { emoji: '😊', label: 'Positive' },
		neutral: { emoji: '😐', label: 'Neutral' }
	} satisfies Record<Sentiment, { emoji: string; label: string }>;

	// Hold form values, accumulated pages, request state, and UI feedback as reactive state.
	let repository = $state('https://github.com/tinymce/tinymce');
	let selectedModel = $state<Model>('svm');
	let issues = $state<Issue[]>([]);
	let page = $state(0);
	let hasMore = $state(false);
	let loading = $state(false);
	let error = $state('');
	let cacheStatus = $state('');
	let started = $state(false);
	let showBackToTop = $state(false);
	let previewIssue = $state<Issue | null>(null);
	let previewLeaving = $state(false);
	let previewPosition = $state<PreviewPosition>({ left: 18, width: 320 });
	let sentinel: HTMLDivElement;

	/** Observe the page sentinel and request another result page before it enters the viewport. */
	onMount(() => {
		const observer = new IntersectionObserver(
			(entries) => {
				if (entries[0]?.isIntersecting && started && hasMore && !loading) void loadNextPage();
			},
			{ rootMargin: '400px' }
		);
		observer.observe(sentinel);

		// Reveal the return control only after the reader has moved beyond the hero area.
		const updateBackToTopVisibility = () => {
			showBackToTop = window.scrollY > 600;
		};
		window.addEventListener('scroll', updateBackToTopVisibility, { passive: true });
		updateBackToTopVisibility();

		// Release both observers when Svelte removes the page component.
		return () => {
			observer.disconnect();
			window.removeEventListener('scroll', updateBackToTopVisibility);
		};
	});

	/** Return the viewport to the page header with native smooth scrolling. */
	function scrollToTop() {
		const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
		window.scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' });
	}

	/** Pause the preview queue without blocking the browser event loop. */
	function wait(milliseconds: number): Promise<void> {
		return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
	}

	/** Report whether any existing card for a sentiment currently intersects the viewport. */
	function hasVisibleCard(kind: Sentiment): boolean {
		const cards = document.querySelectorAll<HTMLElement>(
			`[data-sentiment-column="${kind}"] article`
		);
		return [...cards].some((card) => {
			const bounds = card.getBoundingClientRect();
			return bounds.bottom > 0 && bounds.top < window.innerHeight;
		});
	}

	/** Measure a sentiment column so its fixed preview uses the same horizontal bounds. */
	function measurePreviewPosition(kind: Sentiment): PreviewPosition {
		const column = document.querySelector<HTMLElement>(`[data-sentiment-column="${kind}"]`);
		if (!column) return { left: 18, width: Math.min(440, window.innerWidth - 36) };

		const bounds = column.getBoundingClientRect();
		const left = Math.max(18, bounds.left);
		return {
			left,
			width: Math.max(0, Math.min(bounds.width, window.innerWidth - left - 18))
		};
	}

	/** Preview an infinite-scroll result when its destination stack is outside the viewport. */
	async function appendIssuePage(newIssues: Issue[], showPreviews: boolean) {
		if (!showPreviews) {
			issues = [...issues, ...newIssues];
			return;
		}

		// Limit previews to one card per hidden sentiment stack for each fetched page.
		const previewedSentiments: Sentiment[] = [];
		for (const issue of newIssues) {
			const needsPreview =
				!previewedSentiments.includes(issue.prediction) && !hasVisibleCard(issue.prediction);
			if (needsPreview) {
				previewedSentiments.push(issue.prediction);
				previewPosition = measurePreviewPosition(issue.prediction);
				previewLeaving = false;
				previewIssue = issue;
				await wait(2_000);
				previewLeaving = true;
				if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) await wait(650);
			}
			issues = [...issues, issue];
			if (needsPreview) {
				previewIssue = null;
				previewLeaving = false;
			}
		}
	}

	/** Reset any previous run and start an analysis from the first page. */
	async function analyze() {
		issues = [];
		page = 0;
		hasMore = true;
		error = '';
		cacheStatus = '';
		previewIssue = null;
		previewLeaving = false;
		started = true;
		await loadNextPage();
	}

	/** Fetch, validate, and append the next 25-issue page while preventing duplicate requests. */
	async function loadNextPage() {
		// Ignore observer callbacks while a request is active or pagination is complete.
		if (loading || !hasMore) return;
		loading = true;
		error = '';
		const nextPage = page + 1;
		try {
			// Serialize the current form selection into the API query string.
			const params = new URLSearchParams({
				repository,
				model: selectedModel,
				page: String(nextPage)
			});
			const response = await fetch(`/api/issues?${params}`);
			const result = (await response.json()) as ApiResponse;
			// Convert non-success API responses into the component's normal error path.
			if (!response.ok || !result.issues) throw new Error(result.error ?? 'Unable to load issues.');
			// Preview noteworthy issues only after the initial page is already visible.
			await appendIssuePage(result.issues, page > 0);
			// Update the pagination and cache indicators after every preview has settled.
			page = nextPage;
			hasMore = Boolean(result.hasMore);
			cacheStatus = result.cache ?? '';
		} catch (caught) {
			error = caught instanceof Error ? caught.message : 'An unexpected error occurred.';
			hasMore = false;
		} finally {
			loading = false;
		}
	}
</script>

<!-- Set page-level metadata for browser tabs and search previews. -->
<svelte:head>
	<title>Issue Pulse · GitHub sentiment analyzer</title>
	<meta name="description" content="Analyze the sentiment of public GitHub issues." />
</svelte:head>

<!-- Present the product purpose before the analysis controls. -->
<main>
	<header class="hero">
		<div class="eyebrow"><span class="pulse"></span> GITHUB ISSUE INTELLIGENCE</div>
		<h1>Listen to the pulse of<br /><span>your community.</span></h1>
		<p>
			Analyze the sentiment of issues in any public repository and identify what needs attention.
		</p>
	</header>

	<!-- Collect the classifier and public repository used for a new analysis. -->
	<section class="control-panel" aria-labelledby="setup-title">
		<div class="panel-heading">
			<div>
				<span>01</span>
				<h2 id="setup-title">Configure the analysis</h2>
			</div>
			<p>Choose a model and enter your repository URL</p>
		</div>

		<form
			onsubmit={(event) => {
				// Keep form submission client-side and run the reactive analysis workflow.
				event.preventDefault();
				void analyze();
			}}
		>
			<fieldset>
				<!-- Render one accessible radio-card for every supported model. -->
				<legend>CLASSIFICATION MODEL</legend>
				<div class="model-grid">
					{#each models as model (model.value)}
						<label class:active={selectedModel === model.value}>
							<input type="radio" name="model" value={model.value} bind:group={selectedModel} />
							<span class="radio-dot"></span>
							<strong>{model.label}</strong>
							<small>{model.description}</small>
						</label>
					{/each}
				</div>
			</fieldset>

			<label class="repo-field">
				<!-- Accept either a GitHub URL or the owner/repository shorthand. -->
				<span>PUBLIC REPOSITORY</span>
				<div class="repo-input">
					<svg viewBox="0 0 24 24" aria-hidden="true"
						><path
							d="M12 .7a11.5 11.5 0 0 0-3.64 22.4c.58.1.79-.25.79-.56v-2.2c-3.22.7-3.9-1.36-3.9-1.36-.52-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.7.08-.7 1.17.08 1.78 1.19 1.78 1.19 1.04 1.78 2.72 1.27 3.38.97.1-.75.4-1.27.74-1.56-2.57-.29-5.27-1.28-5.27-5.68 0-1.25.45-2.28 1.19-3.08-.12-.29-.52-1.46.11-3.04 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.2-1.49 3.16-1.18 3.16-1.18.63 1.58.23 2.75.11 3.04.74.8 1.19 1.83 1.19 3.08 0 4.41-2.7 5.38-5.28 5.67.42.36.79 1.06.79 2.14v3.18c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z"
						/></svg
					>
					<input
						bind:value={repository}
						required
						aria-label="Repository URL or name"
						placeholder="https://github.com/owner/repository"
					/>
					<button type="submit" disabled={loading}
						>{loading && issues.length === 0 ? 'Analyzing…' : 'Analyze issues'}
						<span>→</span></button
					>
				</div>
			</label>
		</form>
	</section>

	<!-- Announce request and validation failures without removing the current controls. -->
	{#if error}
		<div class="error" role="alert">
			<strong>We could not complete the analysis.</strong>
			{error}
		</div>
	{/if}

	<!-- Group all accumulated issues into sentiment-specific kanban columns. -->
	{#if started}
		<section class="results" aria-live="polite">
			<div class="results-heading">
				<div>
					<span>02</span>
					<h2>Results</h2>
				</div>
				{#if issues.length}<p>
						{issues.length} issues analyzed · page {page} · cache {cacheStatus}
					</p>{/if}
			</div>
			<div class="kanban">
				<!-- Keep the column order stable regardless of the returned issue distribution. -->
				{#each ['negative', 'neutral', 'positive'] as Sentiment[] as kind (kind)}
					<div class="column {kind}" data-sentiment-column={kind}>
						<div class="column-title">
							<span>{sentiment[kind].emoji}</span>
							<h3>{sentiment[kind].label}</h3>
							<b>{issues.filter((issue) => issue.prediction === kind).length}</b>
						</div>
						<div class="cards">
							<!-- Filter the shared result list into cards belonging to this sentiment. -->
							{#each issues.filter((issue) => issue.prediction === kind) as issue (issue.url)}
								<article>
									<div class="card-top">
										<span>{sentiment[kind].emoji}</span><small
											>{Math.round(issue.confidence * 100)}% confidence</small
										>
									</div>
									<h4>
										<!-- The validated URL intentionally navigates outside this SvelteKit app. -->
										<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -->
										<a href={issue.url} target="_blank" rel="noreferrer">
											{issue.title}<span aria-hidden="true">↗</span>
										</a>
									</h4>
									<p>{issue.body || 'This issue does not include a description.'}</p>
								</article>
							{/each}
						</div>
					</div>
				{/each}
			</div>
			{#if loading}<div class="loader"><span></span> Analyzing 25 more issues…</div>{/if}
			{#if !loading && !hasMore && issues.length}<p class="end">
					You have reached the end of the open issues.
				</p>{/if}
		</section>
	{/if}

	<!-- Trigger the IntersectionObserver when the reader approaches the end of the page. -->
	<div class="sentinel" bind:this={sentinel}></div>
</main>

<!-- Preview newly discovered positive or negative issues in the reader's current viewport. -->
{#if previewIssue}
	<aside
		class="issue-preview {previewIssue.prediction}"
		class:leaving={previewLeaving}
		style={`left: ${previewPosition.left}px; width: ${previewPosition.width}px;`}
		aria-live="polite"
		aria-atomic="true"
	>
		<div class="preview-label">
			<span>{sentiment[previewIssue.prediction].emoji}</span>
			New {sentiment[previewIssue.prediction].label} issue
			<small>Moving to its stack…</small>
		</div>
		<article>
			<div class="card-top">
				<span>{sentiment[previewIssue.prediction].emoji}</span>
				<small>{Math.round(previewIssue.confidence * 100)}% confidence</small>
			</div>
			<h4>
				<!-- The validated URL intentionally navigates outside this SvelteKit app. -->
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -->
				<a href={previewIssue.url} target="_blank" rel="noreferrer">
					{previewIssue.title}<span aria-hidden="true">↗</span>
				</a>
			</h4>
			<p>{previewIssue.body || 'This issue does not include a description.'}</p>
		</article>
	</aside>
{/if}

<!-- Provide a persistent shortcut back to the analysis controls after scrolling. -->
{#if showBackToTop}
	<button class="back-to-top" type="button" onclick={scrollToTop} aria-label="Back to top">
		<span aria-hidden="true">↑</span> Back to top
	</button>
{/if}

<!-- Identify the application stack at the bottom of the page. -->
<footer>
	<span>ISSUE PULSE</span>
	<p>Built with SvelteKit · Cloudflare Workers · GitHub API</p>
</footer>
