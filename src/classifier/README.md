# Sentiment classifier Worker

This SvelteKit application runs on Cloudflare Workers. Its
`GET /api/issues` endpoint obtains public GitHub issues, caches each 25-item
page for 24 hours with the Cloudflare Cache API, and classifies the first 255
characters using the selected SVM, Bayes, or logistic-regression model.

## Setup

Install dependencies and regenerate the Cloudflare binding types:

```sh
pnpm install
pnpm exec wrangler types
```

The configured production bucket is `github-issues-sentiment-models`, and the
object keys are `github-sentiment-{svm,bayes,logistic}-model.json`. Creation and upload commands are documented in the repository's main README.

## Development

```sh
pnpm dev
```

Wrangler uses local R2 storage during local development unless remote bindings
are explicitly enabled. Seed that local bucket before testing the endpoint:

```sh
for model in svm bayes logistic; do
  pnpm exec wrangler r2 object put \
    github-issues-sentiment-models/github-sentiment-$model-model.json \
    --file ../../data/github-sentiment-$model-model.json --local
done
```

## Validation and deployment

From the repository root, the Makefile provides the complete workflow:

```sh
make install
make login
make whoami
make check
make build
make deploy-dry-run
make deploy
```

The Makefile removes `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` from
all recipe environments so exported credentials for another account cannot
override the local Wrangler OAuth session.

Upload the three trained models to the configured production R2 bucket when
initializing the environment or replacing model artifacts:

```sh
make upload-models
```

The equivalent package-level commands are:

```sh
pnpm check
pnpm build
pnpm exec wrangler deploy
```
