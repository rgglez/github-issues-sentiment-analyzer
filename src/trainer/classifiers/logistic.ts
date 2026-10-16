import fs from "node:fs";
import type {
    Classification,
    SentimentClassifier
} from "./types.js";
import {
    buildTfidfState,
    createTfidfVector,
    featurize,
    type SparseVector,
    type TrainingDocument
} from "./tfidf.js";

//------------------------------------------------------------------------------

const MODEL_KIND = "tfidf-logistic-regression";
const MODEL_VERSION = 2;
const MAX_FEATURES = 5_000;
const MIN_DOCUMENT_FREQUENCY = 2;
const TRAINING_EPOCHS = 40;
const INITIAL_LEARNING_RATE = 0.15;
const L2_REGULARIZATION = 0.0001;

//------------------------------------------------------------------------------

interface SerializedLogisticModel {
    kind: typeof MODEL_KIND;
    version: typeof MODEL_VERSION;
    vocabulary: string[];
    inverseDocumentFrequencies: number[];
    labels: string[];
    weights: number[][];
    biases: number[];
}

//------------------------------------------------------------------------------

/** Multiclass logistic regression over sparse TF-IDF document vectors. */
class TfidfLogisticClassifier implements SentimentClassifier {
    private readonly documents: TrainingDocument[] = [];
    private vocabulary: string[] = [];
    private vocabularyIndex = new Map<string, number>();
    private inverseDocumentFrequencies: number[] = [];
    private labels: string[] = [];
    private weights: number[][] = [];
    private biases: number[] = [];

    //------------------------------------------------------------------------------

    /** Add one labeled document to the in-memory training corpus. */
    addDocument(text: string | string[], classification: string): void {
        const tokens = featurize(text);

        if (tokens.length === 0) return;

        this.documents.push({
            label: classification.trim(),
            tokens
        });
    }

    //------------------------------------------------------------------------------

    /** Build TF-IDF vectors and optimize a multiclass softmax classifier. */
    train(): void {
        if (this.documents.length === 0) {
            throw new Error("Cannot train logistic regression without documents.");
        }

        this.labels = [...new Set(this.documents.map(document => document.label))];

        if (this.labels.length < 2) {
            throw new Error("Logistic regression requires at least two labels.");
        }

        this.buildVocabulary();

        const vectors = this.documents.map(document =>
            this.createVector(document.tokens)
        );
        const labelIndices = new Map(
            this.labels.map((label, index) => [label, index])
        );
        const targets = this.documents.map(document => {
            const index = labelIndices.get(document.label);

            if (index === undefined) {
                throw new Error(`Unknown training label: ${document.label}`);
            }

            return index;
        });

        this.weights = this.labels.map(() =>
            Array<number>(this.vocabulary.length).fill(0)
        );
        this.biases = Array<number>(this.labels.length).fill(0);

        for (let epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
            const learningRate = INITIAL_LEARNING_RATE / (1 + epoch * 0.1);

            for (let index = 0; index < vectors.length; index++) {
                this.updateWeights(vectors[index]!, targets[index]!, learningRate);
            }
        }
    }

    //------------------------------------------------------------------------------

    /** Return the label with the highest softmax probability. */
    classify(text: string | string[]): string {
        const classification = this.getClassifications(text)[0];

        if (!classification) {
            throw new Error("The logistic model has no sentiment labels.");
        }

        return classification.label;
    }

    //------------------------------------------------------------------------------

    /** Return one probability-like score for every sentiment class. */
    getClassifications(text: string | string[]): Classification[] {
        this.assertTrained();

        const probabilities = softmax(this.calculateScores(this.createVector(featurize(text))));

        return this.labels
            .map((label, index) => ({
                label,
                value: probabilities[index]!
            }))
            .sort((left, right) => right.value - left.value);
    }

    //------------------------------------------------------------------------------

    /** Serialize the TF-IDF vocabulary and learned logistic parameters. */
    save(
        filename: string,
        callback: (error: NodeJS.ErrnoException | null) => void
    ): void {
        this.assertTrained();

        const serialized: SerializedLogisticModel = {
            kind: MODEL_KIND,
            version: MODEL_VERSION,
            vocabulary: this.vocabulary,
            inverseDocumentFrequencies: this.inverseDocumentFrequencies,
            labels: this.labels,
            weights: this.weights,
            biases: this.biases
        };

        fs.writeFile(filename, JSON.stringify(serialized), "utf8", callback);
    }

    //------------------------------------------------------------------------------

    /** Restore a TF-IDF logistic model from its serialized representation. */
    static fromSerialized(
        serialized: SerializedLogisticModel
    ): TfidfLogisticClassifier {
        validateSerializedModel(serialized);

        const classifier = new TfidfLogisticClassifier();
        classifier.vocabulary = serialized.vocabulary;
        classifier.vocabularyIndex = new Map(
            serialized.vocabulary.map((term, index) => [term, index])
        );
        classifier.inverseDocumentFrequencies =
            serialized.inverseDocumentFrequencies;
        classifier.labels = serialized.labels;
        classifier.weights = serialized.weights;
        classifier.biases = serialized.biases;

        return classifier;
    }

    //------------------------------------------------------------------------------

    /** Select frequent terms and calculate their smoothed IDF weights. */
    private buildVocabulary(): void {
        const state = buildTfidfState(
            this.documents,
            MIN_DOCUMENT_FREQUENCY,
            MAX_FEATURES
        );
        this.vocabulary = state.vocabulary;
        this.vocabularyIndex = state.vocabularyIndex;
        this.inverseDocumentFrequencies = state.inverseDocumentFrequencies;
    }

    //------------------------------------------------------------------------------

    /** Transform tokens into an L2-normalized sparse TF-IDF vector. */
    private createVector(tokens: string[]): SparseVector {
        return createTfidfVector(tokens, {
            vocabulary: this.vocabulary,
            vocabularyIndex: this.vocabularyIndex,
            inverseDocumentFrequencies: this.inverseDocumentFrequencies
        });
    }

    //------------------------------------------------------------------------------

    /** Apply one stochastic-gradient update for a labeled sparse vector. */
    private updateWeights(
        vector: SparseVector,
        target: number,
        learningRate: number
    ): void {
        const probabilities = softmax(this.calculateScores(vector));

        for (let labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
            const error = probabilities[labelIndex]! -
                (labelIndex === target ? 1 : 0);
            const weights = this.weights[labelIndex]!;
            this.biases[labelIndex] =
                this.biases[labelIndex]! - learningRate * error;

            for (let offset = 0; offset < vector.indices.length; offset++) {
                const featureIndex = vector.indices[offset]!;
                const weight = weights[featureIndex]!;
                const gradient =
                    error * vector.values[offset]! + L2_REGULARIZATION * weight;
                weights[featureIndex] = weight - learningRate * gradient;
            }
        }
    }

    //------------------------------------------------------------------------------

    /** Calculate the linear score for every label without densifying a vector. */
    private calculateScores(vector: SparseVector): number[] {
        return this.labels.map((_, labelIndex) => {
            const weights = this.weights[labelIndex]!;
            let score = this.biases[labelIndex]!;

            for (let offset = 0; offset < vector.indices.length; offset++) {
                score +=
                    weights[vector.indices[offset]!]! * vector.values[offset]!;
            }

            return score;
        });
    }

    //------------------------------------------------------------------------------

    /** Reject inference and serialization before parameters are available. */
    private assertTrained(): void {
        if (
            this.labels.length === 0 ||
            this.vocabulary.length === 0 ||
            this.weights.length !== this.labels.length
        ) {
            throw new Error("The TF-IDF logistic classifier is not trained.");
        }
    }
}

//------------------------------------------------------------------------------

/** Create an empty TF-IDF logistic classifier for sentiment training. */
export function createLogisticClassifier(): SentimentClassifier {
    return new TfidfLogisticClassifier();
}

//------------------------------------------------------------------------------

/** Train the TF-IDF logistic classifier. */
export async function trainLogisticClassifier(
    classifier: SentimentClassifier
): Promise<void> {
    classifier.train();
}

//------------------------------------------------------------------------------

/** Load a serialized TF-IDF logistic sentiment model. */
export async function loadLogisticClassifier(
    modelPath: string
): Promise<SentimentClassifier> {
    const contents = await fs.promises.readFile(modelPath, "utf8");
    const serialized = JSON.parse(contents) as SerializedLogisticModel;
    return TfidfLogisticClassifier.fromSerialized(serialized);
}

//------------------------------------------------------------------------------

/** Convert arbitrary linear scores into numerically stable probabilities. */
function softmax(scores: number[]): number[] {
    const maximum = Math.max(...scores);
    const exponentials = scores.map(score => Math.exp(score - maximum));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    return exponentials.map(value => value / total);
}

//------------------------------------------------------------------------------

/** Validate the identity and dimensions of a serialized logistic model. */
function validateSerializedModel(
    model: SerializedLogisticModel
): void {
    const validDimensions =
        model.inverseDocumentFrequencies?.length === model.vocabulary?.length &&
        model.biases?.length === model.labels?.length &&
        model.weights?.length === model.labels?.length &&
        model.weights?.every(
            weights => weights.length === model.vocabulary?.length
        );

    if (
        model.kind !== MODEL_KIND ||
        model.version !== MODEL_VERSION ||
        !validDimensions
    ) {
        throw new Error("Invalid or incompatible TF-IDF logistic model.");
    }
}
