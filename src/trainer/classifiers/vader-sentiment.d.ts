// Minimal typings for the untyped vader-sentiment CommonJS bundle.
declare module "vader-sentiment" {
    export const SentimentIntensityAnalyzer: {
        polarity_scores(text: string): {
            neg: number;
            neu: number;
            pos: number;
            compound: number;
        };
    };
}
