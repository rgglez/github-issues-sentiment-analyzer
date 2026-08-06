# github-issues-sentiment-analyzer

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub all releases](https://img.shields.io/github/downloads/rgglez/github-issues-sentiment-analyzer/total)
![GitHub issues](https://img.shields.io/github/issues/rgglez/github-issues-sentiment-analyzer)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/rgglez/github-issues-sentiment-analyzer)
[![GitHub release](https://img.shields.io/github/release/rgglez/github-issues-sentiment-analyzer.svg)](https://github.com/rgglez/github-issues-sentiment-analyzer/releases/)
![GitHub stars](https://img.shields.io/github/stars/rgglez/github-issues-sentiment-analyzer?style=social)
![GitHub forks](https://img.shields.io/github/forks/rgglez/github-issues-sentiment-analyzer?style=social)

**github-issues-sentiment-analyzer** analyzes the sentiment expressed in GitHub
issues.

## Training

The sentiment classifier is trained with the GitHub Gold Standard dataset. To
rebuild the model using the default paths, run:

```sh
cd train
pnpm install
pnpm train
```

The trainer reads `data/github_gold.csv` and writes the serialized model to
`data/github-sentiment-model.json`.

The trainer is written in TypeScript. Zod validates command-line options and
every CSV row at runtime before a document reaches Natural. Invalid rows are
skipped and counted, and up to five validation errors are included in the
training summary. The command fails if the dataset contains no valid rows.

Additional development commands:

- `pnpm check`: type-checks `train.ts` without producing output files.
- `pnpm build`: compiles the trainer to `train/dist/train.js`.
- `pnpm start`: runs the previously compiled JavaScript program.

### Command-line options

- `-d, --dataset <path>`: sets the training CSV path. It defaults to
  `../data/github_gold.csv`. This option is ignored when `--test` is used.
- `-m, --model <path>`: sets the output path for a newly trained model or the
  input path for `--test`. It defaults to
  `../data/github-sentiment-model.json`.
- `-t, --test`: loads the existing model selected by `--model` and runs only
  the smoke tests. It does not read the dataset, train, or overwrite the model.
- `-V, --version`: prints the trainer version and exits.
- `-h, --help`: prints the command usage and available options.

Options passed through pnpm must follow `--`. For example, to train from and
write to custom paths:

```sh
pnpm train -- --dataset ../data/custom.csv --model ../data/custom-model.json
```

To load the existing model and run only its smoke tests without reading the
dataset or retraining, run:

```sh
pnpm train -- --test
```

### Test output

Each smoke-test result contains the source `text`, the selected `prediction`,
and one entry in `relativeScores` for every sentiment class:

```js
{
  text: "Thanks, this works perfectly now!",
  prediction: "positive",
  relativeScores: [
    {
      label: "positive",
      rawScore: 0.000016697624740138795,
      relativeScore: 0.942407992111388,
      relativePercentage: "94.2%"
    }
  ]
}
```

- `text`: representative developer message passed to the classifier.
- `prediction`: class selected by Natural from `positive`, `neutral`, or
  `negative`.
- `relativeScores`: scores for all classes, ordered from highest to lowest by
  Natural.
- `label`: sentiment class associated with the score.
- `rawScore`: original unnormalized value returned by Natural. It is useful for
  diagnostics but is not a probability or percentage.
- `relativeScore`: `rawScore` divided by the sum of all finite positive raw
  scores for the same text. The class values normally add up to `1`.
- `relativePercentage`: human-readable percentage representation of
  `relativeScore`.

Relative scores express the model's preference among its three classes; they
are not calibrated probabilities or guarantees of correctness. Natural may
also underflow to zero for long documents. If all raw scores are zero, the
relative scores and percentages are reported as zero.

## Dataset attribution

The included `data/github_gold.csv` is *A gold standard for polarity of
emotions of software developers in GitHub* by Nicole Novielli, Fabio Calefato,
Davide Dongiovanni, Daniela Girardi, and Filippo Lanubile. It is distributed
under the [Creative Commons Attribution 4.0 International License][cc-by-4.0].
The copy included in this repository is unmodified; normalization is performed
in memory during training.

Dataset DOI: [10.6084/m9.figshare.11604597][dataset-doi]

The authors request citation of the associated paper:

> N. Novielli, F. Calefato, D. Dongiovanni, D. Girardi, and F. Lanubile,
> “Can We Use SE-specific Sentiment Analysis Tools in a Cross-Platform
> Setting?”, MSR 2020. [https://doi.org/10.1145/3379597.3387446][paper-doi]

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[data/README.md](data/README.md) for complete attribution and licensing notes.

## License

Copyright 2026 Rodolfo González González.

The project source code is licensed under the Apache License 2.0 as described
in [LICENSE](LICENSE). Third-party data is not covered by the Apache License;
`data/github_gold.csv` remains licensed under CC BY 4.0.

[cc-by-4.0]: https://creativecommons.org/licenses/by/4.0/
[dataset-doi]: https://doi.org/10.6084/m9.figshare.11604597
[paper-doi]: https://doi.org/10.1145/3379597.3387446
