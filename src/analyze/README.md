# Sentiment analyzer Worker

This SvelteKit application runs on Cloudflare Workers. Its
`POST /api/sentiment` endpoint loads the portable TF-IDF linear SVM model from
the private `MODELS` R2 binding and classifies the submitted `text`.

## Setup

Install dependencies and regenerate the Cloudflare binding types:

```sh
pnpm install
pnpm exec wrangler types
```

The configured production bucket is `github-issues-sentiment-models`, and the
object key is `github-sentiment-svm-model.json`. Creation and upload commands
are documented in the repository's main README.

## Development

```sh
pnpm dev
```

Wrangler uses local R2 storage during local development unless remote bindings
are explicitly enabled. Seed that local bucket before testing the endpoint:

```sh
pnpm exec wrangler r2 object put \
  github-issues-sentiment-models/github-sentiment-svm-model.json \
  --file ../../data/github-sentiment-svm-model.json \
  --local
```

## Validation and deployment

```sh
pnpm check
pnpm build
pnpm exec wrangler deploy
```
