# Entry points for training, evaluating, and testing the sentiment classifiers.

PNPM := pnpm
CLASSIFIER ?= all
ISSUE ?=

# Declare command targets so Make never confuses them with files of the same name.
.PHONY: help install check build clean train eval test start

# Display the supported workflow commands.
help:
	@echo "Available targets:"
	@echo "  install   Install trainer dependencies from the lockfile"
	@echo "  check     Type-check the trainer without producing output files"
	@echo "  build     Compile the trainer to dist/trainer.js"
	@echo "  clean     Remove the compiled dist/ directory"
	@echo "  train     Train and save models in ../../data"
	@echo "  eval      Report hold-out metrics without saving models"
	@echo "  test      Run smoke tests against the saved models"
	@echo "  start     Train with the compiled JavaScript trainer"
	@echo ""
	@echo "Variables:"
	@echo "  CLASSIFIER=bayes|logistic|svm|all   Classifier to use (default: all)"
	@echo "  ISSUE=\"text\"                        With test, classify this text only"

# Install the exact dependency graph recorded in pnpm-lock.yaml.
install:
	$(PNPM) install --frozen-lockfile

# Run the TypeScript compiler as a type checker only.
check:
	$(PNPM) run check

# Compile trainer.ts and the classifiers to dist/.
build:
	$(PNPM) run build

# Delete compiled output so the next build starts clean.
clean:
	rm -rf dist

# Train the selected classifiers and overwrite their default model artifacts.
train:
	$(PNPM) trainer -- --classifier $(CLASSIFIER)

# Train on 80% of the dataset and print metrics for the held-out 20%.
eval:
	$(PNPM) trainer -- --eval --classifier $(CLASSIFIER)

# Load the saved models and classify the default samples or ISSUE.
test:
	$(PNPM) trainer -- --test --classifier $(CLASSIFIER) $(if $(ISSUE),--issue "$(ISSUE)")

# Run the compiled trainer, rebuilding it first.
start: build
	$(PNPM) start -- --classifier $(CLASSIFIER)
