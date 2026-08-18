import { env } from '$env/dynamic/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { RecaptchaError, verifyRecaptcha } from '$lib/server/recaptcha';
import {
	classifySentiment,
	loadModel,
	MODEL_NAMES,
	SentimentModelError,
	type LoadedModel,
	type ModelName
} from '$lib/server/sentiment-model';

const PAGE_SIZE = 25;
const MAX_PAGES = 40;
const CACHE_SECONDS = 86_400;

// Reuse immutable model-loading promises across warm Worker requests.
const modelPromises = new Map<ModelName, Promise<LoadedModel>>();

/** Minimal issue fields retained in the Cloudflare cache. */
interface CachedIssue {
	title: string;
	body: string;
	url: string;
}

/** One cached GitHub page plus the pagination signal derived from the upstream response. */
interface CachedPage {
	issues: CachedIssue[];
	hasMore: boolean;
}

/** Untrusted subset of fields that may be present in a GitHub Issues API item. */
interface GithubIssue {
	title?: unknown;
	body?: unknown;
	html_url?: unknown;
	pull_request?: unknown;
}

/** Validate a request, obtain a cached GitHub page, and classify every issue body. */
export const GET: RequestHandler = async ({ url, platform, fetch, request }) => {
	// Fail explicitly when the route is executed outside a Cloudflare-bound environment.
	if (!platform)
		return json({ error: 'Cloudflare platform bindings are unavailable.' }, { status: 503 });

	try {
		// Reject bots before GitHub, cache, or R2 work.
		await verifyRecaptcha(
			request.headers.get('X-Recaptcha-Token'),
			platform.env.RECAPTCHA_SECRET_KEY || env.RECAPTCHA_SECRET_KEY,
			fetch
		);

		// Normalize and validate all query parameters before external or storage operations.
		const repository = parseRepository(url.searchParams.get('repository'));
		const modelName = parseModel(url.searchParams.get('model'));
		const page = parsePage(url.searchParams.get('page'));
		// Version the key whenever the cached issue schema changes.
		const cacheKey = `${url.origin}/__cache/v2/github/${repository}?page=${page}`;
		let cachedPage: CachedPage;
		let cacheStatus = 'HIT';
		const cached = await platform.caches.default.match(cacheKey);

		// Reuse a fresh cached page or retrieve and asynchronously cache it for 24 hours.
		if (cached) {
			cachedPage = await cached.json<CachedPage>();
		} else {
			cacheStatus = 'MISS';
			cachedPage = await fetchIssues(fetch, repository, page);
			const response = Response.json(cachedPage, {
				headers: { 'Cache-Control': `public, max-age=${CACHE_SECONDS}` }
			});
			// The DOM and Workers Response declarations differ only on a legacy type member.
			platform.ctx.waitUntil(platform.caches.default.put(cacheKey, response as never));
		}

		// Share one in-flight R2 model read and parsed vocabulary per warm Worker isolate.
		let modelPromise = modelPromises.get(modelName);
		if (!modelPromise) {
			modelPromise = loadModel(platform.env.MODELS, modelName);
			modelPromises.set(modelName, modelPromise);
		}
		let model: LoadedModel;
		try {
			model = await modelPromise;
		} catch (error) {
			// Remove rejected promises so a later request can retry a transient R2 failure.
			modelPromises.delete(modelName);
			throw error;
		}

		// Classify only the already-truncated cached body before returning the current page.
		return json({
			repository,
			page,
			hasMore: cachedPage.hasMore && page < MAX_PAGES,
			cache: cacheStatus,
			issues: cachedPage.issues.map((issue) => ({
				...issue,
				...classifySentiment(model, issue.body)
			}))
		});
	} catch (error) {
		// Map validation, captcha, GitHub, and model failures to stable API status categories.
		const message = error instanceof Error ? error.message : 'Unable to analyze issues.';
		const status =
			error instanceof RecaptchaError
				? error.status
				: error instanceof SentimentModelError
					? 503
					: message.startsWith('GitHub')
						? 502
						: 400;
		console.error(JSON.stringify({ message: 'issue analysis failed', error: message }));
		return json({ error: message }, { status });
	}
};

/** Fetch one 25-item GitHub page, exclude pull requests, and bound cached body text. */
async function fetchIssues(
	fetcher: typeof fetch,
	repository: string,
	page: number
): Promise<CachedPage> {
	// Use GitHub's versioned public REST representation without requiring credentials.
	const response = await fetcher(
		`https://api.github.com/repos/${repository}/issues?state=open&per_page=${PAGE_SIZE}&page=${page}`,
		{
			headers: {
				Accept: 'application/vnd.github+json',
				'User-Agent': 'github-issues-sentiment-analyzer',
				'X-GitHub-Api-Version': '2022-11-28'
			}
		}
	);
	// Preserve a helpful not-found hint while treating all upstream failures consistently.
	if (!response.ok) {
		const hint = response.status === 404 ? ' Repository not found or not public.' : '';
		throw new Error(`GitHub API returned ${response.status}.${hint}`);
	}
	const value: unknown = await response.json();
	if (!Array.isArray(value)) throw new Error('GitHub API returned an unexpected response.');
	// Derive pagination before filtering because GitHub includes pull requests in this endpoint.
	return {
		hasMore: value.length === PAGE_SIZE,
		issues: (value as GithubIssue[])
			.filter(
				(issue) =>
					issue.pull_request === undefined &&
					typeof issue.title === 'string' &&
					typeof issue.html_url === 'string' &&
					issue.html_url.startsWith('https://github.com/')
			)
			.map((issue) => ({
				title: issue.title as string,
				body: typeof issue.body === 'string' ? issue.body.slice(0, 255) : '',
				url: issue.html_url as string
			}))
	};
}

/** Convert a GitHub URL or shorthand into a validated owner/repository cache key. */
function parseRepository(value: string | null): string {
	if (!value) throw new Error('A repository is required.');
	const candidate = value
		.trim()
		.replace(/^https?:\/\/github\.com\//i, '')
		.replace(/\.git$/i, '')
		.replace(/\/$/, '');
	if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(candidate))
		throw new Error('Use a public GitHub URL or owner/repository.');
	return candidate;
}

/** Narrow the model query parameter to one supported model name. */
function parseModel(value: string | null): ModelName {
	if (!value || !MODEL_NAMES.includes(value as ModelName))
		throw new Error('Select a valid sentiment model.');
	return value as ModelName;
}

/** Parse and bound the page parameter to prevent unbounded upstream traversal. */
function parsePage(value: string | null): number {
	const page = Number(value ?? '1');
	if (!Number.isInteger(page) || page < 1 || page > MAX_PAGES) throw new Error('Invalid page.');
	return page;
}
