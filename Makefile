# Build and deployment entry points for the SvelteKit sentiment analyzer Worker.

CLASSIFIER_DIR := src/classifier
DATA_DIR := ../../data
PNPM := pnpm
WRANGLER := $(PNPM) exec wrangler
R2_BUCKET ?= github-issues-sentiment-models
MODELS := svm bayes logistic

# Ignore credentials exported for other Cloudflare accounts and use Wrangler's local OAuth session.
unexport CLOUDFLARE_API_TOKEN CLOUDFLARE_ACCOUNT_ID

# Declare command targets so Make never confuses them with files of the same name.
.PHONY: help install login whoami types check lint test build dev preview deploy-dry-run deploy upload-models

# Display the supported workflow commands.
help:
	@echo "Available targets:"
	@echo "  install          Install classifier dependencies from the lockfile"
	@echo "  login            Authenticate Wrangler through the browser"
	@echo "  whoami           Show the active Cloudflare account and permissions"
	@echo "  types            Regenerate Cloudflare Worker binding types"
	@echo "  check            Run Svelte and TypeScript diagnostics"
	@echo "  lint             Check formatting and lint rules"
	@echo "  test             Run the unit and browser test suite"
	@echo "  build            Validate and create the production Worker bundle"
	@echo "  dev              Start the SvelteKit development server"
	@echo "  preview          Preview the built Worker locally on port 4173"
	@echo "  deploy-dry-run   Build and validate a deployment without publishing"
	@echo "  deploy           Build and deploy classifier to Cloudflare Workers"
	@echo "  upload-models    Upload all trained sentiment models to production R2"

# Install the exact dependency graph recorded in pnpm-lock.yaml.
install:
	cd $(CLASSIFIER_DIR) && $(PNPM) install --frozen-lockfile

# Delegate interactive authentication and credential storage to Wrangler.
login:
	cd $(CLASSIFIER_DIR) && $(WRANGLER) login

# Display the Cloudflare identity and scopes available to the active credentials.
whoami:
	cd $(CLASSIFIER_DIR) && $(WRANGLER) whoami

# Synchronize generated Worker environment types with wrangler.jsonc.
types:
	cd $(CLASSIFIER_DIR) && $(WRANGLER) types

# Run framework and TypeScript diagnostics, including a binding-type freshness check.
check:
	cd $(CLASSIFIER_DIR) && $(PNPM) run check

# Verify source formatting and ESLint rules.
lint:
	cd $(CLASSIFIER_DIR) && $(PNPM) run lint

# Execute all configured Vitest projects once.
test:
	cd $(CLASSIFIER_DIR) && $(PNPM) run test

# Produce the SvelteKit client and Cloudflare Worker server bundles.
build:
	cd $(CLASSIFIER_DIR) && $(PNPM) run build

# Start the local development server with hot module replacement.
dev:
	cd $(CLASSIFIER_DIR) && $(PNPM) run dev

# Serve the previously built Cloudflare Worker through Wrangler.
preview: build
	cd $(CLASSIFIER_DIR) && $(PNPM) run preview

# Exercise Wrangler's deployment pipeline without publishing the Worker.
deploy-dry-run: build
	cd $(CLASSIFIER_DIR) && $(WRANGLER) deploy --dry-run

# Publish the validated Worker and its static assets using wrangler.jsonc.
deploy: build
	cd $(CLASSIFIER_DIR) && $(WRANGLER) deploy

# Upload the three trained model artifacts expected by the production R2 binding.
upload-models:
	cd $(CLASSIFIER_DIR) && for model in $(MODELS); do \
		$(WRANGLER) r2 object put "$(R2_BUCKET)/github-sentiment-$$model-model.json" \
			--file "$(DATA_DIR)/github-sentiment-$$model-model.json" --remote; \
	done
