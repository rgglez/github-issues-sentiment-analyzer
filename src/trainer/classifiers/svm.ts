import fs from "node:fs";
import {
    buildTfidfState,
    createTfidfVector,
    tokenize,
    type SparseVector,
    type TrainingDocument
} from "./tfidf.js";
import type {
    Classification,
    SentimentClassifier
} from "./types.js";

//------------------------------------------------------------------------------

const MODEL_KIND = "tfidf-linear-svm";
const MODEL_VERSION = 1;
const MAX_FEATURES = 10_000;
const MIN_DOCUMENT_FREQUENCY = 2;
const TRAINING_EPOCHS = 40;
const INITIAL_LEARNING_RATE = 0.08;
const L2_REGULARIZATION = 0.0001;

//------------------------------------------------------------------------------

export interface SerializedSvmModel {
    kind: typeof MODEL_KIND;
    version: typeof MODEL_VERSION;
    vocabulary: string[];
    inverseDocumentFrequencies: number[];
    labels: string[];
    weights: number[][];
    biases: number[];
}

//------------------------------------------------------------------------------

/** One-vs-rest linear SVM over sparse TF-IDF document vectors. */
class TfidfLinearSvmClassifier implements SentimentClassifier {
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
        const tokens = tokenize(text);

        if (tokens.length === 0) return;

        this.documents.push({
            label: classification.trim(),
            tokens
        });
    }

    //------------------------------------------------------------------------------

    /** Build TF-IDF vectors and minimize one-vs-rest hinge losses. */
    train(): void {
        if (this.documents.length === 0) {
            throw new Error("Cannot train an SVM without documents.");
        }

        this.labels = [...new Set(this.documents.map(document => document.label))];

        if (this.labels.length < 2) {
            throw new Error("An SVM requires at least two labels.");
        }

        const state = buildTfidfState(
            this.documents,
            MIN_DOCUMENT_FREQUENCY,
            MAX_FEATURES
        );
        this.vocabulary = state.vocabulary;
        this.vocabularyIndex = state.vocabularyIndex;
        this.inverseDocumentFrequencies = state.inverseDocumentFrequencies;

        const vectors = this.documents.map(document =>
            createTfidfVector(document.tokens, state)
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
            const offset = (epoch * 997) % vectors.length;

            for (let step = 0; step < vectors.length; step++) {
                const index = (step + offset) % vectors.length;
                this.updateWeights(vectors[index]!, targets[index]!, learningRate);
            }
        }
    }

    //------------------------------------------------------------------------------

    /** Return the label with the largest decision margin. */
    classify(text: string | string[]): string {
        this.assertTrained();

        const scores = this.calculateScores(this.createVector(tokenize(text)));
        let bestIndex = 0;

        for (let index = 1; index < scores.length; index++) {
            if (scores[index]! > scores[bestIndex]!) bestIndex = index;
        }

        return this.labels[bestIndex]!;
    }

    //------------------------------------------------------------------------------

    /** Return uncalibrated relative scores derived from the SVM margins. */
    getClassifications(text: string | string[]): Classification[] {
        this.assertTrained();

        const scores = this.calculateScores(this.createVector(tokenize(text)));
        const relativeScores = softmax(scores);

        return this.labels
            .map((label, index) => ({
                label,
                value: relativeScores[index]!
            }))
            .sort((left, right) => right.value - left.value);
    }

    //------------------------------------------------------------------------------

    /** Serialize the portable TF-IDF and linear decision parameters. */
    save(
        filename: string,
        callback: (error: NodeJS.ErrnoException | null) => void
    ): void {
        this.assertTrained();

        const serialized: SerializedSvmModel = {
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

    /** Restore a portable SVM model from its serialized representation. */
    static fromSerialized(
        serialized: SerializedSvmModel
    ): TfidfLinearSvmClassifier {
        validateSerializedSvmModel(serialized);

        const classifier = new TfidfLinearSvmClassifier();
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

    /** Apply one sparse stochastic-gradient update to every binary margin. */
    private updateWeights(
        vector: SparseVector,
        target: number,
        learningRate: number
    ): void {
        const scores = this.calculateScores(vector);
        const shrinkage = 1 - learningRate * L2_REGULARIZATION;

        for (let labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
            const expected = labelIndex === target ? 1 : -1;
            const weights = this.weights[labelIndex]!;
            const violatesMargin = expected * scores[labelIndex]! < 1;

            for (let offset = 0; offset < vector.indices.length; offset++) {
                const featureIndex = vector.indices[offset]!;
                const regularizedWeight = weights[featureIndex]! * shrinkage;
                weights[featureIndex] = violatesMargin
                    ? regularizedWeight +
                      learningRate * expected * vector.values[offset]!
                    : regularizedWeight;
            }

            if (violatesMargin) {
                this.biases[labelIndex] =
                    this.biases[labelIndex]! + learningRate * expected;
            }
        }
    }

    //------------------------------------------------------------------------------

    /** Transform tokens using the model's persisted TF-IDF state. */
    private createVector(tokens: string[]): SparseVector {
        return createTfidfVector(tokens, {
            vocabulary: this.vocabulary,
            vocabularyIndex: this.vocabularyIndex,
            inverseDocumentFrequencies: this.inverseDocumentFrequencies
        });
    }

    //------------------------------------------------------------------------------

    /** Calculate every linear decision margin without densifying the vector. */
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
            throw new Error("The TF-IDF linear SVM is not trained.");
        }
    }
}

//------------------------------------------------------------------------------

/** Create an empty TF-IDF linear SVM for sentiment training. */
export function createSvmClassifier(): SentimentClassifier {
    return new TfidfLinearSvmClassifier();
}

//------------------------------------------------------------------------------

/** Train the TF-IDF linear SVM. */
export async function trainSvmClassifier(
    classifier: SentimentClassifier
): Promise<void> {
    classifier.train();
}

//------------------------------------------------------------------------------

/** Load a serialized TF-IDF linear SVM. */
export async function loadSvmClassifier(
    modelPath: string
): Promise<SentimentClassifier> {
    const contents = await fs.promises.readFile(modelPath, "utf8");
    const serialized = JSON.parse(contents) as SerializedSvmModel;
    return TfidfLinearSvmClassifier.fromSerialized(serialized);
}

//------------------------------------------------------------------------------

/** Convert decision margins to comparable but uncalibrated relative scores. */
function softmax(scores: number[]): number[] {
    const maximum = Math.max(...scores);
    const exponentials = scores.map(score => Math.exp(score - maximum));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    return exponentials.map(value => value / total);
}

//------------------------------------------------------------------------------

/** Validate the identity and dimensions of a serialized SVM model. */
function validateSerializedSvmModel(model: SerializedSvmModel): void {
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
        throw new Error("Invalid or incompatible TF-IDF linear SVM model.");
    }
}
