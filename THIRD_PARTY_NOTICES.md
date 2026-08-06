# Third-party notices

## GitHub Gold Standard sentiment dataset

This project includes the following dataset:

> *A gold standard for polarity of emotions of software developers in GitHub*

Authors: Nicole Novielli, Fabio Calefato, Davide Dongiovanni, Daniela Girardi,
and Filippo Lanubile.

- Source: https://doi.org/10.6084/m9.figshare.11604597
- Figshare record: https://figshare.com/articles/dataset/11604597
- License: [Creative Commons Attribution 4.0 International][cc-by-4.0]

The dataset contains manually annotated GitHub pull-request and commit
comments. The `data/github_gold.csv` copy distributed by this project is
unmodified. The training program normalizes URLs, code blocks, commit hashes,
and whitespace in memory without overwriting the source dataset.

The dataset authors request citation of:

> Nicole Novielli, Fabio Calefato, Davide Dongiovanni, Daniela Girardi, and
> Filippo Lanubile. 2020. “Can We Use SE-specific Sentiment Analysis Tools in
> a Cross-Platform Setting?” In *Proceedings of the 17th International
> Conference on Mining Software Repositories (MSR '20)*.
> https://doi.org/10.1145/3379597.3387446

The project source code is licensed separately under Apache License 2.0. That
license does not replace or supersede the CC BY 4.0 license applicable to the
dataset. The generated sentiment model was trained from this dataset, so this
attribution must be retained when the model is redistributed.

[cc-by-4.0]: https://creativecommons.org/licenses/by/4.0/
