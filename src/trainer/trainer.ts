// Load Node.js utilities and third-party packages used for file streaming,
// path resolution, CSV parsing, runtime validation, CLI handling, and training.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "csv-parse";
import { Command } from "commander";
import { z } from "zod";
import {
    createBayesClassifier,
    loadBayesClassifier,
    trainBayesClassifier
} from "./classifiers/bayes.js";
import {
    createLogisticClassifier,
    loadLogisticClassifier,
    trainLogisticClassifier
} from "./classifiers/logistic.js";
import {
    createSvmClassifier,
    loadSvmClassifier,
    trainSvmClassifier
} from "./classifiers/svm.js";
import {
    CLASSIFIER_NAMES,
    type Classification,
    type ClassifierName,
    type SentimentClassifier
} from "./classifiers/types.js";

//------------------------------------------------------------------------------

// Define canonical sentiment classes once for runtime validation and types.
const SENTIMENT_LABELS = ["positive", "neutral", "negative"] as const;
const DEFAULT_SAMPLES = [
    "Thanks, this works perfectly now!",
    "This is completely broken and keeps crashing.",
    "The patch updates the dependency to version 4.",
    "not so bad",
    "This is not good",
    "It's not working at all",
    "Not bad at all, thanks!"
];

//------------------------------------------------------------------------------

// Normalize accepted aliases before validating the canonical sentiment class.
const sentimentLabelSchema = z.preprocess(value => {
    if (value === undefined || value === null) return value;

    const label = String(value).trim().toLowerCase();
    const mappings: Record<string, SentimentLabel> = {
        positive: "positive",
        pos: "positive",
        "1": "positive",
        neutral: "neutral",
        neu: "neutral",
        "0": "neutral",
        negative: "negative",
        neg: "negative",
        "-1": "negative"
    };

    return mappings[label] ?? label;
}, z.enum(SENTIMENT_LABELS));

//------------------------------------------------------------------------------

// Validate every CSV record before allowing it into the training corpus.
const datasetRowSchema = z.object({
    ID: z.string().trim().min(1),
    Polarity: sentimentLabelSchema,
    Text: z.string().trim().min(1)
});

//------------------------------------------------------------------------------

// Validate Commander output as an additional runtime boundary.
const cliOptionsSchema = z.object({
    classifier: z.enum([...CLASSIFIER_NAMES, "all"]),
    dataset: z.string().trim().min(1),
    model: z.string().trim().min(1).optional(),
    test: z.boolean(),
    eval: z.boolean(),
    issue: z.string().trim().min(1).optional()
});

//------------------------------------------------------------------------------

// Derive TypeScript types from Zod so runtime and compile-time definitions
// cannot silently drift apart.
type SentimentLabel = (typeof SENTIMENT_LABELS)[number];
type DatasetRow = z.infer<typeof datasetRowSchema>;
type CliOptions = z.infer<typeof cliOptionsSchema>;

//------------------------------------------------------------------------------

interface InvalidRowExample {
    rowNumber: number;
    issues: string[];
}

interface RelativeClassification {
    label: string;
    rawScore: number;
    relativeScore: number;
}

//------------------------------------------------------------------------------

// Define the command-line interface, including default input and output paths.
const program = new Command();

program
    .name("trainer")
    .description("Train or test a GitHub issue sentiment classifier.")
    .version("1.0.0")
    .option(
        "-c, --classifier <name>",
        "Classifier to train or test: bayes, logistic, svm, or all",
        "bayes"
    )
    .option(
        "-d, --dataset <path>",
        "Path to the training CSV file",
        "../../data/github_gold.csv"
    )
    .option(
        "-m, --model <path>",
        "Path where the model will be saved or loaded"
    )
    .option(
        "-t, --test",
        "Load the existing model and evaluate sample text without training",
        false
    )
    .option(
        "-e, --eval",
        "Train on 80% of the dataset and report metrics on the other 20% without saving",
        false
    )
    .option(
        "-i, --issue <text>",
        "Evaluate a single issue instead of the default samples"
    );

// Parse the supplied arguments and validate the resulting option values. A
// standalone `--` inserted by a package manager is removed so documented pnpm
// commands behave the same as direct script execution.
program.parse(
    process.argv.filter((argument, index) => index < 2 || argument !== "--")
);
const options: CliOptions = cliOptionsSchema.parse(program.opts());

if (options.issue && !options.test) {
    program.error("--issue can only be used together with --test");
}

if (options.eval && options.test) {
    program.error("--eval cannot be used together with --test");
}

if (options.classifier === "all" && options.model) {
    program.error("--model cannot be used together with --classifier all");
}

//------------------------------------------------------------------------------

// Recreate __dirname for this ES module and locate the trainer package whether
// execution starts from trainer.ts or the compiled dist/trainer.js artifact.
const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const trainerDirectory =
    path.basename(dirname) === "dist" ? path.dirname(dirname) : dirname;

// Resolve CLI paths from the trainer package so execution works from any current
// working directory and produces identical paths before and after compilation.
const datasetPath = path.resolve(trainerDirectory, options.dataset);
const selectedClassifiers: readonly ClassifierName[] =
    options.classifier === "all" ? CLASSIFIER_NAMES : [options.classifier];

/** Resolve the explicit --model path or the classifier's default artifact. */
function resolveModelPath(name: ClassifierName): string {
    return path.resolve(
        trainerDirectory,
        options.model ?? getDefaultModelPath(name)
    );
}

//------------------------------------------------------------------------------

/**
 * Stream the CSV dataset, validate every row, then train and save the selected
 * sentiment classifier.
 */
async function trainModel(name: ClassifierName): Promise<void> {
    // Create the classifier and track accepted rows, rejected rows, and labels.
    const classifier = createClassifier(name);
    const modelPath = resolveModelPath(name);
    const heldOut: DatasetRow[] = [];
    let rowNumber = 1;
    let trainingCount = 0;
    let skippedCount = 0;
    const labelCounts = new Map<SentimentLabel, number>();
    const invalidRowExamples: InvalidRowExample[] = [];

    // Stream the dataset instead of loading the entire CSV file into memory.
    // Parse the semicolon-delimited CSV and expose each record as an object
    // whose properties match the header names, such as Text and Polarity.
    const records = fs.createReadStream(datasetPath).pipe(
        parse({
            columns: true,
            delimiter: ";",
            bom: true,
            skip_empty_lines: true,
            trim: true,
            relax_quotes: true
        })
    );

    try {
        for await (const rawRow of records as AsyncIterable<unknown>) {
            rowNumber++;

            // Validate untrusted CSV data before reading its fields.
            const result = datasetRowSchema.safeParse(rawRow);
            if (!result.success) {
                skippedCount++;

                // Retain a small diagnostic sample without flooding the output
                // when a large number of records is invalid.
                if (invalidRowExamples.length < 5) {
                    invalidRowExamples.push({
                        rowNumber,
                        issues: result.error.issues.map(
                            issue =>
                                `${issue.path.join(".") || "row"}: ${issue.message}`
                        )
                    });
                }
                continue;
            }

            // Clean validated text and add the canonical sentiment label. With
            // --eval, a deterministic 20% of the rows is held out for testing.
            const row: DatasetRow = result.data;
            if (options.eval && isHeldOut(row.ID)) {
                heldOut.push(row);
                continue;
            }
            classifier.addDocument(cleanText(row.Text), row.Polarity);
            trainingCount++;
            labelCounts.set(
                row.Polarity,
                (labelCounts.get(row.Polarity) ?? 0) + 1
            );
        }
    } catch (error) {
        // Report file-system and CSV parsing failures with a non-zero exit status.
        console.error("Error reading the CSV file:", error);
        process.exitCode = 1;
        return;
    }

    // Prevent creation of an empty model when the input schema is invalid.
    if (trainingCount === 0) {
        console.error("No valid rows were found. Check the CSV schema.", {
            skippedCount,
            invalidRowExamples
        });
        process.exitCode = 1;
        return;
    }

    // Report validation and label-distribution results before fitting.
    console.log("Dataset loaded:", {
        datasetPath,
        trainingCount,
        skippedCount,
        labels: Object.fromEntries(labelCounts),
        invalidRowExamples
    });

    // Fit the classifier using all valid documents collected from the stream.
    console.log(`Training ${name} model...`);
    try {
        await trainClassifier(name, classifier);
    } catch (error) {
        console.error("Error training the classifier:", error);
        process.exitCode = 1;
        return;
    }

    if (options.eval) {
        evaluate(classifier, heldOut);
        return;
    }

    // Serialize the trained classifier for use by the analysis application.
    await new Promise<void>(resolve => {
        classifier.save(modelPath, error => {
            if (error) {
                console.error(`Error saving model to ${modelPath}:`, error);
                process.exitCode = 1;
            } else {
                console.log(`Model saved to ${modelPath}`);
            }
            resolve();
        });
    });
}

//------------------------------------------------------------------------------

/** Select a stable 20% of the rows for evaluation from their numeric IDs. */
function isHeldOut(id: string): boolean {
    // ponytail: ID modulo split, stratify by label if class balance drifts.
    return Number(id) % 5 === 0;
}

//------------------------------------------------------------------------------

/** Report accuracy, macro-F1, and a confusion matrix for held-out rows. */
function evaluate(
    classifier: SentimentClassifier,
    rows: readonly DatasetRow[]
): void {
    // confusion[actual][predicted] counts every held-out prediction.
    const confusion = Object.fromEntries(
        SENTIMENT_LABELS.map(actual => [
            actual,
            Object.fromEntries(SENTIMENT_LABELS.map(predicted => [predicted, 0]))
        ])
    ) as Record<SentimentLabel, Record<string, number>>;
    let correct = 0;

    for (const row of rows) {
        const predicted = classifier.classify(cleanText(row.Text));
        confusion[row.Polarity][predicted] =
            (confusion[row.Polarity][predicted] ?? 0) + 1;
        if (predicted === row.Polarity) correct++;
    }

    // Macro-F1 averages per-class F1 so the majority class cannot dominate.
    const f1Scores = SENTIMENT_LABELS.map(label => {
        const truePositives = confusion[label][label] ?? 0;
        const predictedCount = SENTIMENT_LABELS.reduce(
            (sum, actual) => sum + (confusion[actual][label] ?? 0),
            0
        );
        const actualCount = Object.values(confusion[label]).reduce(
            (sum, count) => sum + count,
            0
        );
        const precision = predictedCount ? truePositives / predictedCount : 0;
        const recall = actualCount ? truePositives / actualCount : 0;
        return precision + recall
            ? (2 * precision * recall) / (precision + recall)
            : 0;
    });

    console.log("Held-out evaluation:", {
        testCount: rows.length,
        accuracy: (correct / rows.length).toFixed(4),
        macroF1: (
            f1Scores.reduce((sum, value) => sum + value, 0) / f1Scores.length
        ).toFixed(4)
    });
    console.table(confusion);
    testModel(classifier, DEFAULT_SAMPLES);
}

//------------------------------------------------------------------------------

/** Create the classifier implementation selected through the CLI. */
function createClassifier(name: ClassifierName): SentimentClassifier {
    switch (name) {
        case "bayes":
            return createBayesClassifier();
        case "logistic":
            return createLogisticClassifier();
        case "svm":
            return createSvmClassifier();
    }
}

//------------------------------------------------------------------------------

/** Train the selected classifier with its corresponding strategy. */
function trainClassifier(
    name: ClassifierName,
    classifier: SentimentClassifier
): Promise<void> {
    switch (name) {
        case "bayes":
            return trainBayesClassifier(classifier);
        case "logistic":
            return trainLogisticClassifier(classifier);
        case "svm":
            return trainSvmClassifier(classifier);
    }
}

//------------------------------------------------------------------------------

/** Load the selected classifier implementation from disk. */
function loadClassifier(
    name: ClassifierName,
    modelPath: string
): Promise<SentimentClassifier> {
    switch (name) {
        case "bayes":
            return loadBayesClassifier(modelPath);
        case "logistic":
            return loadLogisticClassifier(modelPath);
        case "svm":
            return loadSvmClassifier(modelPath);
    }
}

//------------------------------------------------------------------------------

/** Select a stable default artifact name for each classifier implementation. */
function getDefaultModelPath(name: ClassifierName): string {
    switch (name) {
        case "bayes":
            return "../../data/github-sentiment-bayes-model.json";
        case "logistic":
            return "../../data/github-sentiment-logistic-model.json";
        case "svm":
            return "../../data/github-sentiment-svm-model.json";
    }
}

//------------------------------------------------------------------------------

/** Load a trained model and evaluate either the defaults or one supplied issue. */
async function testSavedModel(name: ClassifierName): Promise<void> {
    const modelPath = resolveModelPath(name);
    try {
        const classifier = await loadClassifier(name, modelPath);
        const samples = options.issue ? [options.issue] : DEFAULT_SAMPLES;

        console.log(`Model loaded from ${modelPath}`);
        testModel(classifier, samples);
    } catch (error) {
        console.error(`Error loading model from ${modelPath}:`, error);
        process.exitCode = 1;
    }
}

//------------------------------------------------------------------------------

/** Print predictions and normalized class scores for each supplied sample. */
function testModel(
    model: SentimentClassifier,
    samples: readonly string[]
): void {
    for (const sample of samples) {
        const text = cleanText(sample);
        const relativeScores = normalizeScores(
            model.getClassifications(text)
        ).map(({ label, rawScore, relativeScore }) => ({
            label,
            rawScore,
            relativeScore,
            relativePercentage: `${(relativeScore * 100).toFixed(1)}%`
        }));

        console.log({
            text: sample,
            prediction: model.classify(text),
            relativeScores
        });
    }
}

//------------------------------------------------------------------------------

/** Normalize positive classifier scores so that their values add up to one. */
function normalizeScores(
    classifications: Classification[]
): RelativeClassification[] {
    const total = classifications.reduce(
        (sum, classification) =>
            Number.isFinite(classification.value) && classification.value > 0
                ? sum + classification.value
                : sum,
        0
    );

    if (total === 0) {
        return classifications.map(({ label, value }) => ({
            label,
            rawScore: value,
            relativeScore: 0
        }));
    }

    return classifications.map(({ label, value }) => ({
        label,
        rawScore: value,
        relativeScore: value / total
    }));
}

//------------------------------------------------------------------------------

/**
 * Normalize developer text while preserving placeholders for technical content
 * that should remain meaningful to the classifier.
 */
function cleanText(value: string): string {
    // Replace high-variance technical values with stable tokens, collapse
    // whitespace, and remove leading or trailing spaces.
    return value
        .replace(/```[\s\S]*?```/g, " CODE_BLOCK ")
        .replace(/https?:\/\/\S+/g, " URL ")
        .replace(/\b[0-9a-f]{7,40}\b/gi, " COMMIT_HASH ")
        .replace(/\s+/g, " ")
        .trim();
}

//------------------------------------------------------------------------------

// Select test-only mode or the dataset training pipeline.
// Run the selected classifiers one after another so their output stays apart.
for (const name of selectedClassifiers) {
    await (options.test ? testSavedModel(name) : trainModel(name));
}
