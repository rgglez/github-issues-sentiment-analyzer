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

### reCAPTCHA v3

`GET /api/issues` requires a Google reCAPTCHA v3 token on every request
(including infinite-scroll pages). The client sends it as
`X-Recaptcha-Token`; the Worker verifies it with Google before GitHub/R2 work.

1. Create a v3 key pair in the [reCAPTCHA admin](https://www.google.com/recaptcha/admin)
   and allow `localhost` plus your production hostname(s).
2. Copy `.env.example` to `.env` and set both keys for local Vite:
   - `PUBLIC_RECAPTCHA_SITE_KEY` — public site key
   - `RECAPTCHA_SECRET_KEY` — secret (local only)
3. For `wrangler dev` / `pnpm preview`, put both keys in `.dev.vars` (Vite's
   `.env` is not read by Wrangler):
   ```sh
   RECAPTCHA_SECRET_KEY=your_secret_key_here
   PUBLIC_RECAPTCHA_SITE_KEY=your_site_key_here
   ```
4. Production secret (never commit it):
   ```sh
   pnpm exec wrangler secret put RECAPTCHA_SECRET_KEY
   ```
5. The site key is read at runtime via `$env/dynamic/public`, not baked in at
   build time, so production takes it from `vars.PUBLIC_RECAPTCHA_SITE_KEY` in
   `wrangler.jsonc`. Update it there when the key pair changes.

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

From this directory (`src/classifier`), the Makefile provides the complete
workflow:

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
