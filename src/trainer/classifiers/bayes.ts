import natural from "natural";
import type { SentimentClassifier } from "./types.js";

//------------------------------------------------------------------------------

const { BayesClassifier } = natural;

//------------------------------------------------------------------------------

/** Create a Naive Bayes classifier for sentiment training. */
export function createBayesClassifier(): SentimentClassifier {
    return new BayesClassifier();
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

            resolve(classifier);
        });
    });
}
