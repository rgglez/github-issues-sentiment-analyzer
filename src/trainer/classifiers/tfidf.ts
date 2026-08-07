// Shared TF-IDF representation used by portable linear classifiers.

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
