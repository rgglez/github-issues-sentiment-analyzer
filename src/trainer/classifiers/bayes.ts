import fs from "node:fs";
import natural from "natural";
import { markNegation, tokenize, vaderToken } from "./tfidf.js";
import type { Classification, SentimentClassifier } from "./types.js";

//------------------------------------------------------------------------------

const { BayesClassifier, stopwords } = natural;
const STOPWORDS = new Set<string>(stopwords);

//------------------------------------------------------------------------------

/**
 * Build Bayes features: negation-marked unigrams without stopwords plus one
 * VADER polarity bucket. Tokens are not stemmed, which scored better on the
 * held-out split and lets the Worker reproduce them exactly. Natural uses
 * token arrays as-is, so no stemming or stopword removal happens inside it.
 */
// ponytail: no n-grams because Natural never prunes rare features and the JSON
// would exceed the Worker's 5 MB limit; prune by document frequency first.
export function bayesFeatures(text: string | string[]): string[] {
    const raw = Array.isArray(text) ? text.join(" ") : text;
    const features = markNegation(tokenize(raw)).filter(
        token => !STOPWORDS.has(token.replace(/^not_/, ""))
    );

    return [...features, vaderToken(raw)];
}

//------------------------------------------------------------------------------

/** Adapt Natural's Bayes classifier so every input goes through bayesFeatures. */
class FeaturizedBayesClassifier implements SentimentClassifier {
    constructor(private readonly classifier: natural.BayesClassifier) {}

    addDocument(text: string | string[], classification: string): void {
        this.classifier.addDocument(bayesFeatures(text), classification);
    }

    train(): void {
        this.classifier.train();
    }

    classify(text: string | string[]): string {
        return this.classifier.classify(bayesFeatures(text));
    }

    getClassifications(text: string | string[]): Classification[] {
        return this.classifier.getClassifications(bayesFeatures(text));
    }

    save(
        filename: string,
        callback: (error: NodeJS.ErrnoException | null) => void
    ): void {
        // Keep only what inference needs. Natural would also serialize `docs`,
        // a copy of every training text that is most of the file.
        const { features, classifier } = this.classifier;
        fs.writeFile(
            filename,
            JSON.stringify({ features, classifier }),
            "utf8",
            callback
        );
    }
}

//------------------------------------------------------------------------------

/** Create a Naive Bayes classifier for sentiment training. */
export function createBayesClassifier(): SentimentClassifier {
    return new FeaturizedBayesClassifier(new BayesClassifier());
}

//------------------------------------------------------------------------------

/** Train a Naive Bayes classifier on the current process. */
export async function trainBayesClassifier(
    classifier: SentimentClassifier
): Promise<void> {
    classifier.train();
}

//------------------------------------------------------------------------------

/** Load a serialized Naive Bayes sentiment model. */
export function loadBayesClassifier(
    modelPath: string
): Promise<SentimentClassifier> {
    return new Promise((resolve, reject) => {
        BayesClassifier.load(modelPath, null, (error, classifier) => {
            if (error) {
                reject(error);
                return;
            }

            if (!classifier) {
                reject(new Error("Natural did not return a Bayes classifier."));
                return;
            }

            resolve(new FeaturizedBayesClassifier(classifier));
        });
    });
}
