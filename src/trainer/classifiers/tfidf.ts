// Shared TF-IDF representation used by portable linear classifiers.

import { SentimentIntensityAnalyzer } from "vader-sentiment";

//------------------------------------------------------------------------------

export interface TrainingDocument {
    label: string;
    tokens: string[];
}

export interface SparseVector {
    indices: number[];
    values: number[];
}

export interface TfidfState {
    vocabulary: string[];
    vocabularyIndex: Map<string, number>;
    inverseDocumentFrequencies: number[];
}

//------------------------------------------------------------------------------

/** Tokenize normalized developer text without depending on Node or Natural. */
export function tokenize(text: string | string[]): string[] {
    if (Array.isArray(text)) {
        return text.map(token => token.toLowerCase()).filter(Boolean);
    }

    return (
        text
            .toLowerCase()
            .match(/[\p{L}\p{N}_]+(?:['’-][\p{L}\p{N}_]+)*/gu) ?? []
    );
}

//------------------------------------------------------------------------------

// Negators open a scope over the next tokens, matching VADER's 3-word window.
const NEGATOR = /^(?:not|no|never|none|nobody|nothing|neither|nor|cannot|without|\p{L}+n['’]t)$/u;
const NEGATION_WINDOW = 3;

/** Prefix the tokens that follow a negator so "not good" differs from "good". */
export function markNegation(tokens: string[]): string[] {
    let remaining = 0;

    return tokens.map(token => {
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
export function ngrams(tokens: string[], n: number): string[] {
    return tokens
        .slice(0, Math.max(0, tokens.length - n + 1))
        .map((_, index) => tokens.slice(index, index + n).join(" "));
}

/** Bucket VADER's compound score, which handles negators and intensifiers. */
export function vaderToken(text: string): string {
    const { compound } = SentimentIntensityAnalyzer.polarity_scores(text);

    if (compound <= -0.5) return "__vader_neg2__";
    if (compound <= -0.05) return "__vader_neg1__";
    if (compound >= 0.5) return "__vader_pos2__";
    if (compound >= 0.05) return "__vader_pos1__";
    return "__vader_neu__";
}

/**
 * Build the linear-model features: negation-marked unigrams, contiguous
 * bigrams and trigrams of the raw tokens, and one VADER polarity bucket.
 */
export function featurize(text: string | string[]): string[] {
    const raw = Array.isArray(text) ? text.join(" ") : text;
    const tokens = tokenize(raw);

    if (tokens.length === 0) return [];

    return [
        ...markNegation(tokens),
        ...ngrams(tokens, 2),
        ...ngrams(tokens, 3),
        vaderToken(raw)
    ];
}

//------------------------------------------------------------------------------

/** Build a deterministic vocabulary and smoothed inverse document frequencies. */
export function buildTfidfState(
    documents: TrainingDocument[],
    minimumDocumentFrequency: number,
    maximumFeatures: number
): TfidfState {
    const documentFrequencies = new Map<string, number>();

    for (const document of documents) {
        for (const term of new Set(document.tokens)) {
            documentFrequencies.set(
                term,
                (documentFrequencies.get(term) ?? 0) + 1
            );
        }
    }

    const vocabulary = [...documentFrequencies.entries()]
        .filter(([, frequency]) => frequency >= minimumDocumentFrequency)
        .sort(
            ([leftTerm, leftFrequency], [rightTerm, rightFrequency]) =>
                rightFrequency - leftFrequency ||
                leftTerm.localeCompare(rightTerm)
        )
        .slice(0, maximumFeatures)
        .map(([term]) => term);

    if (vocabulary.length === 0) {
        throw new Error("No terms satisfied the TF-IDF vocabulary criteria.");
    }

    return {
        vocabulary,
        vocabularyIndex: new Map(
            vocabulary.map((term, index) => [term, index])
        ),
        inverseDocumentFrequencies: vocabulary.map(term => {
            const frequency = documentFrequencies.get(term) ?? 0;
            return Math.log((1 + documents.length) / (1 + frequency)) + 1;
        })
    };
}

//------------------------------------------------------------------------------

/** Transform tokens into an L2-normalized sparse TF-IDF vector. */
export function createTfidfVector(
    tokens: string[],
    state: TfidfState
): SparseVector {
    const termCounts = new Map<number, number>();

    for (const token of tokens) {
        const index = state.vocabularyIndex.get(token);
        if (index === undefined) continue;
        termCounts.set(index, (termCounts.get(index) ?? 0) + 1);
    }

    const indices = [...termCounts.keys()].sort((left, right) => left - right);
    const values = indices.map(
        index =>
            termCounts.get(index)! * state.inverseDocumentFrequencies[index]!
    );
    const magnitude = Math.sqrt(
        values.reduce((sum, value) => sum + value * value, 0)
    );

    return {
        indices,
        values:
            magnitude === 0
                ? values
                : values.map(value => value / magnitude)
    };
}
