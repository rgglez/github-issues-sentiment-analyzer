export const CLASSIFIER_NAMES = ["bayes", "logistic", "svm"] as const;

//------------------------------------------------------------------------------

export type ClassifierName = (typeof CLASSIFIER_NAMES)[number];

export interface Classification {
    label: string;
    value: number;
}

export interface SentimentClassifier {
    addDocument(text: string | string[], classification: string): void;
    train(): void;
    classify(text: string | string[]): string;
    getClassifications(text: string | string[]): Classification[];
    save(
        filename: string,
        callback: (error: NodeJS.ErrnoException | null) => void
    ): void;
}
