import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { classifySentiment, loadSvmModel, SentimentModelError } from '$lib/server/sentiment-model';

//------------------------------------------------------------------------------

const MAX_REQUEST_SIZE = 65_536;
const MAX_TEXT_LENGTH = 20_000;
let modelPromise: ReturnType<typeof loadSvmModel> | undefined;

//------------------------------------------------------------------------------

/** Classify one bounded issue text using the SVM model stored in R2. */
export const POST: RequestHandler = async ({ request, platform }) => {
	if (!platform) {
		return json({ error: 'Cloudflare platform bindings are unavailable.' }, { status: 503 });
	}

	try {
		const body = await readBoundedJson(request);
		const text = getIssueText(body);
		modelPromise ??= loadSvmModel(platform.env.MODELS, platform.env.SENTIMENT_MODEL_KEY);

		let model: Awaited<ReturnType<typeof loadSvmModel>>;
		try {
			model = await modelPromise;
		} catch (error) {
			modelPromise = undefined;
			throw error;
		}

		return json({ text, ...classifySentiment(model, text) });
	} catch (error) {
		const message = error instanceof Error ? error.message : 'Unable to classify sentiment.';
		const status = error instanceof SentimentModelError ? 503 : 400;
		console.error(JSON.stringify({ message: 'sentiment classification failed', error: message }));
		return json({ error: message }, { status });
	}
};

//------------------------------------------------------------------------------

/** Read a JSON request body without allowing unbounded buffering. */
async function readBoundedJson(request: Request): Promise<unknown> {
	if (!request.body) throw new Error('A JSON request body is required.');

	const reader = request.body.getReader();
	const chunks: Uint8Array[] = [];
	let size = 0;

	while (true) {
		const { done, value } = await reader.read();
		if (done) break;
		size += value.byteLength;
		if (size > MAX_REQUEST_SIZE) {
			await reader.cancel();
			throw new Error(`Request body exceeds ${MAX_REQUEST_SIZE} bytes.`);
		}
		chunks.push(value);
	}

	const bytes = new Uint8Array(size);
	let offset = 0;
	for (const chunk of chunks) {
		bytes.set(chunk, offset);
		offset += chunk.byteLength;
	}

	return JSON.parse(new TextDecoder().decode(bytes));
}

//------------------------------------------------------------------------------

/** Validate and return the issue text from an unknown request body. */
function getIssueText(value: unknown): string {
	if (typeof value !== 'object' || value === null || Array.isArray(value)) {
		throw new Error('The request body must be a JSON object.');
	}

	const text = Reflect.get(value, 'text');
	if (typeof text !== 'string' || text.trim().length === 0) {
		throw new Error('The text field must be a non-empty string.');
	}
	if (text.length > MAX_TEXT_LENGTH) {
		throw new Error(`The text field exceeds ${MAX_TEXT_LENGTH} characters.`);
	}

	return text.trim();
}
