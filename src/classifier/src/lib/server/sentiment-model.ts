// Portable inference for the three sentiment models produced by the trainer.

import { SentimentIntensityAnalyzer } from 'vader-sentiment';

const MAX_MODEL_SIZE = 5_000_000;
const LINEAR_KINDS = ['tfidf-linear-svm', 'tfidf-logistic-regression'] as const;

/** Runtime model names accepted by the API and mapped to R2 object keys. */
export const MODEL_NAMES = ['svm', 'bayes', 'logistic'] as const;

/** Union derived from the supported model-name constants. */
export type ModelName = (typeof MODEL_NAMES)[number];

/** Serialized TF-IDF parameters shared by the SVM and logistic classifiers. */
interface LinearModel {
	kind: (typeof LINEAR_KINDS)[number];
	version: 2;
	vocabulary: string[];
	inverseDocumentFrequencies: number[];
	labels: string[];
	weights: number[][];
	biases: number[];
}

/** Serialized feature counts required for multinomial Bayes inference. */
interface BayesModel {
	features: Record<string, number>;
	classifier: {
		classFeatures: Record<string, Record<string, number>>;
		classTotals: Record<string, number>;
		totalExamples: number;
		smoothing: number;
	};
}

/** Any validated serialized model supported by this inference module. */
type Model = LinearModel | BayesModel;

/** Validated model plus precomputed lookup data used during repeated inference. */
export interface LoadedModel {
	name: ModelName;
	model: Model;
	vocabularyIndex?: Map<string, number>;
}

/** Compact classification returned to the issues API for one body. */
export interface SentimentPrediction {
	prediction: string;
	confidence: number;
}

/** Distinguishes model storage or validation failures from invalid API input. */
export class SentimentModelError extends Error {}

/** Load and validate one bounded model through the configured R2 binding. */
export async function loadModel(bucket: R2Bucket, name: ModelName): Promise<LoadedModel> {
	// Resolve the selected model to the naming convention used by the R2 bucket.
	const key = `github-sentiment-${name}-model.json`;
	const object = await bucket.get(key);
	if (object === null) throw new SentimentModelError(`Sentiment model not found in R2: ${key}`);
	if (object.size > MAX_MODEL_SIZE) {
		throw new SentimentModelError(`Sentiment model exceeds ${MAX_MODEL_SIZE} bytes.`);
	}

	// Parse as unknown and validate the complete format before inference can access it.
	const value: unknown = await object.json<unknown>();
	if (name === 'bayes') {
		if (!isBayesModel(value)) throw new SentimentModelError('Invalid Bayes model.');
		// Natural keys classFeatures by each feature's position in `features`,
		// which JSON.parse preserves as Object.keys order.
		return {
			name,
			model: value,
			vocabularyIndex: new Map(
				Object.keys(value.features).map((feature, index) => [feature, index])
			)
		};
	}
	if (!isLinearModel(value)) throw new SentimentModelError(`Invalid ${name} model.`);

	return {
		name,
		model: value,
		vocabularyIndex: new Map(value.vocabulary.map((term, index) => [term, index]))
	};
}

/** Classify a bounded issue body with the selected model. */
export function classifySentiment(loaded: LoadedModel, text: string): SentimentPrediction {
	// Dispatch to the algorithm that matches the validated serialized model.
	return loaded.name === 'bayes'
		? classifyBayes(loaded, cleanText(text))
		: classifyLinear(loaded, cleanText(text));
}

/** Build a sparse TF-IDF vector and score it with SVM margins or logistic logits. */
function classifyLinear(loaded: LoadedModel, text: string): SentimentPrediction {
	const model = loaded.model as LinearModel;
	// Count only tokens present in the trained vocabulary to keep the vector sparse.
	const counts = new Map<number, number>();
	for (const token of featurize(text)) {
		const index = loaded.vocabularyIndex!.get(token);
		if (index !== undefined) counts.set(index, (counts.get(index) ?? 0) + 1);
	}
	// Apply persisted IDF weights and L2 normalization exactly as the trainer does.
	const indices = [...counts.keys()].sort((a, b) => a - b);
	const values = indices.map(
		(index) => counts.get(index)! * model.inverseDocumentFrequencies[index]!
	);
	const magnitude = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
	const normalized = magnitude === 0 ? values : values.map((value) => value / magnitude);
	// Calculate one linear decision score for each sentiment label.
	const rawScores = model.labels.map((_, labelIndex) => {
		let score = model.biases[labelIndex]!;
		for (let offset = 0; offset < indices.length; offset++) {
			score += model.weights[labelIndex]![indices[offset]!]! * normalized[offset]!;
		}
		return score;
	});
	// Convert raw scores into relative confidence values for display.
	const probabilities = softmax(rawScores);
	const best = indexOfMaximum(model.kind === 'tfidf-linear-svm' ? rawScores : probabilities);
	return { prediction: model.labels[best]!, confidence: probabilities[best]! };
}

/** Score the unique trained features in the text with smoothed Bayes log probabilities. */
function classifyBayes(loaded: LoadedModel, text: string): SentimentPrediction {
	const model = loaded.model as BayesModel;
	// Natural trains presence vectors, so repeated tokens count only once per document.
	const present = new Set<number>();
	for (const token of bayesFeatures(text)) {
		const index = loaded.vocabularyIndex!.get(token);
		if (index !== undefined) present.add(index);
	}
	const labels = Object.keys(model.classifier.classFeatures);
	// Accumulate log probabilities to avoid underflow from multiplying tiny values.
	const logScores = labels.map((label) => {
		const total = model.classifier.classTotals[label]!;
		let score = Math.log(total / model.classifier.totalExamples);
		const classFeatures = model.classifier.classFeatures[label]!;
		for (const feature of present) {
			score += Math.log((classFeatures[feature] ?? model.classifier.smoothing) / total);
		}
		return score;
	});
	const probabilities = softmax(logScores);
	const best = indexOfMaximum(probabilities);
	return { prediction: labels[best]!, confidence: probabilities[best]! };
}

/** Replace volatile technical fragments and normalize whitespace as done during training. */
function cleanText(value: string): string {
	return value
		.replace(/```[\s\S]*?```/g, ' CODE_BLOCK ')
		.replace(/https?:\/\/\S+/g, ' URL ')
		.replace(/\b[0-9a-f]{7,40}\b/gi, ' COMMIT_HASH ')
		.replace(/\s+/g, ' ')
		.trim();
}

/** Split normalized text into lowercase Unicode word tokens. */
function tokenize(text: string): string[] {
	return text.toLowerCase().match(/[\p{L}\p{N}_]+(?:['’-][\p{L}\p{N}_]+)*/gu) ?? [];
}

// Negators open a scope over the next tokens, matching VADER's 3-word window.
const NEGATOR = /^(?:not|no|never|none|nobody|nothing|neither|nor|cannot|without|\p{L}+n['’]t)$/u;
const NEGATION_WINDOW = 3;

/** Prefix the tokens that follow a negator so "not good" differs from "good". */
function markNegation(tokens: string[]): string[] {
	let remaining = 0;
	return tokens.map((token) => {
		if (NEGATOR.test(token)) {
			remaining = NEGATION_WINDOW;
			return token;
		}
		if (remaining === 0) return token;
		remaining--;
		return `not_${token}`;
	});
}

/** Join every run of n contiguous tokens into one space-separated feature. */
function ngrams(tokens: string[], n: number): string[] {
	return tokens
		.slice(0, Math.max(0, tokens.length - n + 1))
		.map((_, index) => tokens.slice(index, index + n).join(' '));
}

/** Bucket VADER's compound score, which handles negators and intensifiers. */
function vaderToken(text: string): string {
	const { compound } = SentimentIntensityAnalyzer.polarity_scores(text);
	if (compound <= -0.5) return '__vader_neg2__';
	if (compound <= -0.05) return '__vader_neg1__';
	if (compound >= 0.5) return '__vader_pos2__';
	if (compound >= 0.05) return '__vader_pos1__';
	return '__vader_neu__';
}

/** Build linear-model features exactly as the trainer's featurize does. */
export function featurize(text: string): string[] {
	const tokens = tokenize(text);
	if (tokens.length === 0) return [];
	return [...markNegation(tokens), ...ngrams(tokens, 2), ...ngrams(tokens, 3), vaderToken(text)];
}

/**
 * Build Bayes features as the trainer does: unstemmed, negation-marked
 * unigrams and one VADER bucket. The trainer drops Natural's stopwords, so the
 * caller's model-feature filter removes them here without a stopword list.
 */
export function bayesFeatures(text: string): string[] {
	return [...markNegation(tokenize(text)), vaderToken(text)];
}

/** Convert arbitrary decision scores into a numerically stable probability distribution. */
function softmax(scores: number[]): number[] {
	const maximum = Math.max(...scores);
	const exponentials = scores.map((score) => Math.exp(score - maximum));
	const total = exponentials.reduce((sum, value) => sum + value, 0);
	return exponentials.map((value) => value / total);
}

/** Return the index of the largest numeric value without allocating a sorted copy. */
function indexOfMaximum(values: number[]): number {
	let best = 0;
	for (let index = 1; index < values.length; index++)
		if (values[index]! > values[best]!) best = index;
	return best;
}

/** Narrow an unknown value to a non-null, non-array object. */
function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Verify that every member of an unknown array is a string. */
function isStringArray(value: unknown): value is string[] {
	return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

/** Verify that every member of an unknown array is a finite number. */
function isNumberArray(value: unknown): value is number[] {
	return (
		Array.isArray(value) && value.every((item) => typeof item === 'number' && Number.isFinite(item))
	);
}

/** Validate linear model identity, primitive fields, and matrix dimensions. */
function isLinearModel(value: unknown): value is LinearModel {
	// Reject incompatible model kinds or versions before inspecting their arrays.
	if (
		!isRecord(value) ||
		!LINEAR_KINDS.includes(value.kind as (typeof LINEAR_KINDS)[number]) ||
		value.version !== 2
	)
		return false;
	// Validate every persisted vector and matrix element before checking dimensions.
	if (
		!isStringArray(value.vocabulary) ||
		!isNumberArray(value.inverseDocumentFrequencies) ||
		!isStringArray(value.labels) ||
		!isNumberArray(value.biases) ||
		!Array.isArray(value.weights) ||
		!value.weights.every(isNumberArray)
	)
		return false;
	// Ensure labels, biases, weights, and vocabulary dimensions agree.
	const vocabularyLength = value.vocabulary.length;
	return (
		vocabularyLength === value.inverseDocumentFrequencies.length &&
		value.labels.length > 1 &&
		value.labels.length === value.biases.length &&
		value.labels.length === value.weights.length &&
		value.weights.every((row) => row.length === vocabularyLength)
	);
}

/** Validate the outer structures and numeric metadata required by Bayes inference. */
function isBayesModel(value: unknown): value is BayesModel {
	if (!isRecord(value) || !isRecord(value.features) || !isRecord(value.classifier)) return false;
	const classifier = value.classifier;
	return (
		isRecord(classifier.classFeatures) &&
		isRecord(classifier.classTotals) &&
		typeof classifier.totalExamples === 'number' &&
		typeof classifier.smoothing === 'number'
	);
}
