// Load Node.js utilities and third-party packages used for file streaming,
// path resolution, CSV parsing, runtime validation, CLI handling, and training.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import natural from "natural";
import { parse } from "csv-parse";
import { Command } from "commander";
import { z } from "zod";

// Natural is a CommonJS package at runtime, so consume its default export and
// extract the classifier instead of relying on an ESM named export.
const { BayesClassifier } = natural;
type BayesModel = InstanceType<typeof BayesClassifier>;

//------------------------------------------------------------------------------

// Define canonical sentiment classes once for runtime validation and types.
const SENTIMENT_LABELS = ["positive", "neutral", "negative"] as const;
type SentimentLabel = (typeof SENTIMENT_LABELS)[number];

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

// Validate every CSV record before allowing it into the training corpus.
const datasetRowSchema = z.object({
  ID: z.string().trim().min(1),
  Polarity: sentimentLabelSchema,
  Text: z.string().trim().min(1)
});

// Validate Commander output as an additional runtime boundary.
const cliOptionsSchema = z.object({
  dataset: z.string().trim().min(1),
  model: z.string().trim().min(1),
  test: z.boolean()
});

// Derive TypeScript types from Zod so runtime and compile-time definitions
// cannot silently drift apart.
type DatasetRow = z.infer<typeof datasetRowSchema>;
type CliOptions = z.infer<typeof cliOptionsSchema>;

interface Classification {
  label: string;
  value: number;
}

interface RelativeClassification {
  label: string;
  rawScore: number;
  relativeScore: number;
}

interface InvalidRowExample {
  rowNumber: number;
  issues: string[];
}

//------------------------------------------------------------------------------

// Define the command-line interface, including default input and output paths.
const program = new Command();

program
  .name("train")
  .description("Train a sentiment model from a CSV file.")
  .version("1.0.0")
  .option(
    "-d, --dataset <path>",
    "Path to the training CSV file",
    "../data/github_gold.csv"
  )
  .option(
    "-m, --model <path>",
    "Path where the trained model will be saved",
    "../data/github-sentiment-model.json"
  )
  .option(
    "-t, --test",
    "Load the existing model and run only the smoke test",
    false
  );

// Parse the supplied arguments and validate the resulting option values. A
// standalone `--` inserted by a package manager is removed so documented pnpm
// commands behave the same as direct script execution.
program.parse(process.argv.filter((argument, index) => index < 2 || argument !== "--"));
const options: CliOptions = cliOptionsSchema.parse(program.opts());

//------------------------------------------------------------------------------

// Recreate __dirname for this ES module and locate the train package whether
// execution starts from train.ts or the compiled dist/train.js artifact.
const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const trainDirectory =
  path.basename(dirname) === "dist" ? path.dirname(dirname) : dirname;

// Resolve CLI paths from the train package so execution works from any current
// working directory and produces identical paths before and after compilation.
const datasetPath = path.resolve(trainDirectory, options.dataset);
const modelPath = path.resolve(trainDirectory, options.model);

//------------------------------------------------------------------------------

// Select test-only mode or the complete dataset training pipeline.
if (options.test) {
  loadAndTestModel();
} else {
  trainModel();
}

//------------------------------------------------------------------------------

/**
 * Stream the CSV dataset, validate every row, train and save a Naive Bayes
 * classifier, and run the smoke test against the newly generated model.
 */
function trainModel(): void {
  // Create the classifier and track accepted rows, rejected rows, and labels.
  const classifier = new BayesClassifier();
  let rowNumber = 1;
  let trainingCount = 0;
  let skippedCount = 0;
  const labelCounts = new Map<SentimentLabel, number>();
  const invalidRowExamples: InvalidRowExample[] = [];

  // Stream the dataset instead of loading the entire CSV file into memory.
  fs.createReadStream(datasetPath)
    .pipe(
      // Parse the semicolon-delimited CSV and expose each record as an object
      // whose properties match the header names, such as Text and Polarity.
      parse({
        columns: true,
        delimiter: ";",
        bom: true,
        skip_empty_lines: true,
        trim: true,
        relax_quotes: true
      })
    )
    .on("data", (rawRow: unknown) => {
      rowNumber++;

      // Validate untrusted CSV data before reading its fields.
      const result = datasetRowSchema.safeParse(rawRow);
      if (!result.success) {
        skippedCount++;

        // Retain a small diagnostic sample without flooding the output when a
        // large number of records is invalid.
        if (invalidRowExamples.length < 5) {
          invalidRowExamples.push({
            rowNumber,
            issues: result.error.issues.map(issue =>
              `${issue.path.join(".") || "row"}: ${issue.message}`
            )
          });
        }
        return;
      }

      // Clean validated text and add the canonical sentiment label.
      const row: DatasetRow = result.data;
      const text = cleanText(row.Text);
      classifier.addDocument(text, row.Polarity);
      trainingCount++;
      labelCounts.set(
        row.Polarity,
        (labelCounts.get(row.Polarity) ?? 0) + 1
      );
    })
    .on("end", () => {
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
      console.log("Training model...");
      classifier.train();

      // Serialize the trained classifier for use by the analysis application.
      classifier.save(modelPath, error => {
        if (error) {
          console.error(`Error saving model to ${modelPath}:`, error);
          process.exitCode = 1;
          return;
        }

        console.log(`Model saved to ${modelPath}`);

        // Run a small smoke test to make the resulting behavior visible.
        testModel(classifier);
      });
    })
    .on("error", (error: Error) => {
      // Report file-system and CSV parsing failures with a non-zero exit status.
      console.error("Error reading the CSV file:", error);
      process.exitCode = 1;
    });
}

//------------------------------------------------------------------------------

/**
 * Load an existing serialized model and run the smoke test without reading the
 * dataset or retraining the classifier.
 */
function loadAndTestModel(): void {
  // Restore the Natural classifier saved at the path selected by --model.
  BayesClassifier.load(modelPath, null, (error, classifier) => {
    if (error || !classifier) {
      console.error(
        `Error loading model from ${modelPath}:`,
        error ?? new Error("Natural did not return a classifier.")
      );
      process.exitCode = 1;
      return;
    }

    // Confirm the loaded artifact and display its smoke-test predictions.
    console.log(`Model loaded from ${modelPath}`);
    testModel(classifier);
  });
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

/**
 * Convert raw classifier scores into relative scores whose values add up to
 * one. These values express model preference, not calibrated probabilities.
 */
function normalizeScores(
  classifications: Classification[]
): RelativeClassification[] {
  // Add all finite positive scores to obtain the normalization denominator.
  const total = classifications.reduce(
    (sum, classification) =>
      Number.isFinite(classification.value) && classification.value > 0
        ? sum + classification.value
        : sum,
    0
  );

  // Natural may underflow to zero for long documents. Preserve raw scores and
  // labels, but return zero relative scores when normalization is impossible.
  if (total === 0) {
    return classifications.map(({ label, value }) => ({
      label,
      rawScore: value,
      relativeScore: 0
    }));
  }

  // Divide every raw score by the total so relative scores add up to one.
  return classifications.map(({ label, value }) => ({
    label,
    rawScore: value,
    relativeScore: value / total
  }));
}

//------------------------------------------------------------------------------

/**
 * Classify representative sentences as a smoke test after training and print
 * the selected label together with raw and relative class scores.
 */
function testModel(model: BayesModel): void {
  // Cover an expected positive, negative, and neutral developer message.
  const samples = [
    "Thanks, this works perfectly now!",
    "This is completely broken and keeps crashing.",
    "The patch updates the dependency to version 4."
  ];

  // Print one prediction record for each representative sample.
  for (const text of samples) {
    // Preserve Natural's raw scores and add normalized relative representations.
    const relativeScores = normalizeScores(
      model.getClassifications(text)
    ).map(({ label, rawScore, relativeScore }) => ({
      label,
      rawScore,
      relativeScore,
      relativePercentage: `${(relativeScore * 100).toFixed(1)}%`
    }));

    console.log({
      text,
      prediction: model.classify(text),
      relativeScores
    });
  }
}
