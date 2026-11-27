SHELL := /bin/bash
OPENRESTY ?= openresty
RESTY ?= resty

LIB_EXT := $(if $(filter Darwin,$(shell uname -s)),dylib,so)
CORE_LIB := core/target/release/libaigw_core.$(LIB_EXT)

SEMCACHE_SOCK ?= /tmp/aigw-semcache.sock
# l6, l12, or a model folder such as models/minilm-l6-qqp (make finetune)
SEMCACHE_MODEL ?= l6
FT_MODEL ?= models/minilm-l6-qqp
FT_BURN_MODEL ?= models/minilm-l6-qqp-burn
# Cross-encoder check: empty turns it off (make finetune-rerank builds one)
SEMCACHE_RERANK ?=
RERANK_MODEL ?= models/minilm-l6-qqp-rerank
# Trained router of the sidecar: empty = none (make router-head or make finetune-router builds one)
SEMCACHE_ROUTER ?=
ROUTER_HEAD ?= models/router-head
ROUTER_CLASSIFIER ?= models/router-classifier
# Guardrail classifier of the sidecar: empty = none (make finetune-guard builds one)
SEMCACHE_GUARD ?=
GUARD_MODEL ?= models/guard-classifier

.PHONY: help core semcache semcache-run qqp eval train-env finetune parity ft-threshold train-burn \
	finetune-burn finetune-rerank rerank-parity rerank-threshold sprout router-head finetune-router router-parity router-eval \
	guard-data finetune-guard guard-parity guard-eval bench-data bench bench-onnx models \
	run stop test test-core test-unit smoke cache-smoke docker-up docker-smoke docker-down clean

help: ## Show the targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-17s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

core: ## Build the Rust core (release)
	cargo build --release --manifest-path core/Cargo.toml

semcache: ## Build the semantic cache sidecar (release)
	cargo build --release --manifest-path semcache/Cargo.toml

semcache-run: semcache ## Run the sidecar in the foreground on $(SEMCACHE_SOCK)
	semcache/target/release/aigw-semcache --listen unix:$(SEMCACHE_SOCK) --model $(SEMCACHE_MODEL) \
		$(if $(SEMCACHE_RERANK),--rerank $(SEMCACHE_RERANK)) $(if $(SEMCACHE_ROUTER),--router $(SEMCACHE_ROUTER)) \
		$(if $(SEMCACHE_GUARD),--guard $(SEMCACHE_GUARD))

data/qqp-dev.tsv:
	scripts/fetch-qqp.sh 4000 $@ 0

data/qqp-test.tsv:
	scripts/fetch-qqp.sh 4000 $@ 4000

qqp: data/qqp-dev.tsv data/qqp-test.tsv ## Download QQP dev and test pairs (4000 each)

eval: semcache qqp ## Evaluate cache thresholds and latency on dev, quality on test
	semcache/target/release/semcache-eval data/qqp-dev.tsv --model $(SEMCACHE_MODEL)
	semcache/target/release/semcache-eval data/qqp-test.tsv --model $(SEMCACHE_MODEL) \
		--latency-samples 1 --index-size 0

train-env: ## Create the Python training environment (uv, Python 3.12, PyTorch)
	uv sync --project train

# Precision that the fine-tuned threshold must reach on dev (base L6 at 0.97 has 0.938).
FT_PRECISION ?= 0.94

finetune: qqp semcache ## Fine-tune MiniLM L6, check parity, save its threshold, report on test
	uv run --project train python train/finetune_cache.py --out $(FT_MODEL)
	$(MAKE) parity ft-threshold

parity: semcache ## Make sure Burn and PyTorch embed $(FT_MODEL) the same way
	uv run --project train python train/parity.py --model $(FT_MODEL)

ft-threshold: semcache qqp ## Choose the threshold of $(FT_MODEL) on dev, report it on test
	semcache/target/release/semcache-eval data/qqp-dev.tsv --model $(FT_MODEL) --index-size 0 \
		--save-threshold $(FT_PRECISION) --show-errors 25
	semcache/target/release/semcache-eval data/qqp-test.tsv --model $(FT_MODEL) --index-size 0 \
		--latency-samples 1 --at $$(jq .threshold $(FT_MODEL)/semcache.json)

train-burn: ## Build the Burn trainer (Metal backend)
	cargo build --release --manifest-path train-burn/Cargo.toml

data/qqp-train-raw.tsv:
	uv run --project train python train/export_qqp.py --out $@

finetune-burn: train-burn qqp data/qqp-train-raw.tsv ## Same fine-tuning in Burn, then parity and threshold
	train-burn/target/release/finetune-cache --out $(FT_BURN_MODEL)
	$(MAKE) parity ft-threshold FT_MODEL=$(FT_BURN_MODEL)

# Bi-encoder that finds the candidates, the candidate floor, and the precision
# that the cascade must reach on dev.
RERANK_BI ?= $(FT_BURN_MODEL)
RERANK_CANDIDATE ?= 0.80
RERANK_PRECISION ?= 0.95
# Hard negatives for the cross-encoder (0 = none)
RERANK_HARD ?= 30000

finetune-rerank: train-burn qqp data/qqp-train-raw.tsv ## Train the cross-encoder in Burn, then parity and thresholds
	train-burn/target/release/finetune-rerank --out $(RERANK_MODEL) --hard-negatives $(RERANK_HARD)
	$(MAKE) rerank-parity rerank-threshold

rerank-parity: semcache ## Make sure Burn and transformers score $(RERANK_MODEL) the same way
	uv run --project train python train/parity.py --rerank --model $(RERANK_MODEL)

rerank-threshold: semcache qqp ## Choose the cascade thresholds on dev, report them on test
	semcache/target/release/semcache-eval data/qqp-dev.tsv --model $(RERANK_BI) --index-size 0 --latency-samples 1 \
		--rerank $(RERANK_MODEL) --candidate $(RERANK_CANDIDATE) --save-threshold $(RERANK_PRECISION) --show-errors 15
	semcache/target/release/semcache-eval data/qqp-test.tsv --model $(RERANK_BI) --index-size 0 --latency-samples 1 \
		--rerank $(RERANK_MODEL) --rerank-at $$(jq .threshold $(RERANK_MODEL)/rerank.json)

# The script writes the train, dev and test files together.
data/sprout-train.jsonl:
	uv run --project train python -I train/export_sprout.py

sprout: data/sprout-train.jsonl ## Read the SPROUT router dataset (about 26 MB of its columns)

router-head: semcache sprout ## Train the router head on the cache embedding of $(FT_BURN_MODEL)
	semcache/target/release/router-head --model $(FT_BURN_MODEL) --out $(ROUTER_HEAD)

finetune-router: train-burn sprout ## Train the classifier router in Burn (Metal)
	train-burn/target/release/finetune-router --out $(ROUTER_CLASSIFIER)
	$(MAKE) router-parity

router-parity: semcache ## Make sure Burn and transformers score $(ROUTER_CLASSIFIER) the same way
	uv run --project train python train/parity.py --router --model $(ROUTER_CLASSIFIER)

router-eval: semcache sprout ## Compare the routers on the SPROUT test set
	semcache/target/release/router-eval data/sprout-test.jsonl --by-dataset \
		$(if $(wildcard $(ROUTER_HEAD)/router.json),--router $(ROUTER_HEAD)) \
		$(if $(wildcard $(ROUTER_CLASSIFIER)/router.json),--router $(ROUTER_CLASSIFIER))

data/guard-train.jsonl:
	uv run --project train python -I train/export_guard.py

guard-data: data/guard-train.jsonl ## Download the four prompt-injection datasets (about 3.5 MB)

finetune-guard: train-burn guard-data ## Train the guardrail classifier in Burn (Metal), then parity
	train-burn/target/release/finetune-guard --train data/guard-train.jsonl --train data/guard-extra.jsonl \
		--out $(GUARD_MODEL)
	$(MAKE) guard-parity

guard-parity: semcache ## Make sure Burn and transformers score $(GUARD_MODEL) the same way
	uv run --project train python train/parity.py --guard --model $(GUARD_MODEL)

guard-eval: semcache guard-data ## Compare the regex deny list and the classifier on all guard test sets
	semcache/target/release/guard-eval data/guard-test.jsonl data/guard-deepset.jsonl \
		data/guard-jailbreak.jsonl data/guard-gandalf.jsonl \
		$(if $(wildcard $(GUARD_MODEL)/guard.json),--guard $(GUARD_MODEL))

# Short prompts (QQP questions) and long prompts (SPROUT), one per line.
data/bench-short.txt: data/qqp-dev.tsv
	awk -F'\t' 'NR > 1 { print $$1 }' $< > $@

data/bench-long.txt: data/sprout-train.jsonl
	jq -r '.prompt | gsub("[\n\r\t]"; " ")' data/sprout-test.jsonl | head -2000 > $@

bench-data: data/bench-short.txt data/bench-long.txt ## Write the prompt files of the load test

bench: semcache bench-data ## Load test of the sidecar with 1 to 6 embedding threads
	scripts/bench-sidecar.sh $(FT_BURN_MODEL)

bench-onnx: semcache bench-data ## Export the cache model to ONNX, time ONNX Runtime, compare with Burn
	uv run --project train python train/export_onnx.py --model $(FT_BURN_MODEL)
	uv run --project train python train/bench_onnx.py --model $(FT_BURN_MODEL)

models: ## List the model folders with their run records (train.json)
	scripts/models.sh

run: core ## Run the gateway in the foreground with your .env
	scripts/gateway.sh start

stop: ## Stop a background gateway
	scripts/gateway.sh stop

test: test-core test-unit smoke ## Run all tests

test-core: ## Rust unit tests
	cargo test --manifest-path core/Cargo.toml

test-unit: core ## Lua unit tests (needs resty from OpenResty)
	$(RESTY) -I gateway/lua -I tests/unit tests/unit/run.lua $(CORE_LIB) gateway/conf/nginx.conf

smoke: core ## End-to-end test: local gateway against the mock upstream, then the Anthropic format
	tests/local-smoke.sh
	tests/anthropic-smoke.sh

cache-smoke: core semcache ## End-to-end test of the semantic cache (real model)
	SEMCACHE_MODEL=$(SEMCACHE_MODEL) SEMCACHE_RERANK=$(SEMCACHE_RERANK) SEMCACHE_ROUTER=$(SEMCACHE_ROUTER) \
		SEMCACHE_GUARD=$(SEMCACHE_GUARD) tests/cache-smoke.sh

docker-up: ## Build and run the gateway in Docker with your .env
	docker compose up --build gateway

docker-smoke: ## End-to-end test in Docker against the mock upstream
	AIGW_API_KEYS=test-key \
	AIGW_CHEAP_BASE_URL=http://mock:19000/cheap/v1 AIGW_CHEAP_MODEL=mock-cheap-model \
	AIGW_STRONG_BASE_URL=http://mock:19000/strong/v1 AIGW_STRONG_MODEL=mock-strong-model \
	AIGW_STRONG_API_KEY=mock-strong-key \
	docker compose --profile test up --build -d --wait
	tests/smoke.sh; status=$$?; docker compose --profile test down; exit $$status

docker-down: ## Stop the Docker services
	docker compose --profile test down

clean: ## Remove build output and logs
	cargo clean --manifest-path core/Cargo.toml
	cargo clean --manifest-path semcache/Cargo.toml
	cargo clean --manifest-path train-burn/Cargo.toml
	rm -rf gateway/logs gateway/*_temp gateway/conf/generated.conf tests/mock/logs tests/mock/*_temp
