# Configuration
LM_EVAL_TASKS_PATH ?= /home/primeintellect/repos/filtering_for_danger/lm_eval_tasks
TASKS ?= wmdp_bio_categorized_mcqa,wmdp_bio_cloze_verified,mmlu,piqa,lambada,hellaswag
WANDB_PROJECT ?= Deep-Ignorance-Evals-HF
WANDB_ENTITY ?= EleutherAI

# Run evaluation directly on host
eval_hf:
ifndef MODEL
	$(error MODEL is required. Usage: make eval_hf MODEL=<model_name>)
endif
	lm_eval --model hf \
		--model_args pretrained=$(MODEL),dtype=bfloat16,parallelize=True,attn_implementation=flash_attention_2 \
		--wandb_args project=$(WANDB_PROJECT),entity=$(WANDB_ENTITY),name=$(MODEL) \
		--tasks $(TASKS) \
		--batch_size 64 \
		--include_path ./lm_eval_tasks/

# Run evaluation in Docker container
eval_hf_docker:
ifndef MODEL
	$(error MODEL is required. Usage: make eval_hf_docker MODEL=<model_name>)
endif
ifndef WANDB_API_KEY
	$(error WANDB_API_KEY is required. Export it and use: sudo -E make eval_hf_docker MODEL=<model_name>)
endif
ifndef HF_TOKEN
	$(error HF_TOKEN is required. Export it and use: sudo -E make eval_hf_docker MODEL=<model_name>)
endif
	sudo docker build -f neox/Dockerfile.evals -t lm-eval-harness:latest neox/
	$(eval WANDB_RUN_NAME := $(shell echo $(MODEL) | sed 's/[\/,]/_/g'))
	sudo docker run --rm \
		--gpus all \
		-v $(LM_EVAL_TASKS_PATH):/lm_eval_tasks \
		-e CUDA_VISIBLE_DEVICES \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e HF_TOKEN=$(HF_TOKEN) \
		lm-eval-harness:latest \
		bash -c "cd /root/lm-evaluation-harness && python -m lm_eval --model hf \
			--model_args pretrained=$(MODEL),dtype=bfloat16,parallelize=True,attn_implementation=flash_attention_2 \
			--tasks $(TASKS) \
			--batch_size 128 \
			--include_path /lm_eval_tasks"

eval_hf_docker_final_models:
	# Baseline Model
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-unfiltered

	# Filtering-Only Models
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-e2e-strong-filter
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-e2e-weak-filter
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-weak-filter-pt-strong-filter-anneal

	# CB Models
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-unfiltered-cb
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-e2e-strong-filter-cb
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal-cb

	# CB-Lat Models
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-unfiltered-cb-lat
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-e2e-strong-filter-cb-lat
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal-cb-lat

	# Pretraining Stage Models
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-pretraining-stage-unfiltered
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-pretraining-stage-strong-filter
	make eval_hf_docker MODEL=EleutherAI/deep-ignorance-pretraining-stage-weak-filter

eval_hf_final_models:
	# Baseline Model
	make eval_hf MODEL=EleutherAI/deep-ignorance-unfiltered

	# Filtering-Only Models
	make eval_hf MODEL=EleutherAI/deep-ignorance-e2e-strong-filter
	make eval_hf MODEL=EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal
	make eval_hf MODEL=EleutherAI/deep-ignorance-e2e-weak-filter
	make eval_hf MODEL=EleutherAI/deep-ignorance-weak-filter-pt-strong-filter-anneal
	make eval_hf MODEL=EleutherAI/deep-ignorance-pretraining-stage-unfiltered
	make eval_hf MODEL=EleutherAI/deep-ignorance-pretraining-stage-strong-filter
	make eval_hf MODEL=EleutherAI/deep-ignorance-pretraining-stage-weak-filter

	# CB Models
	make eval_hf MODEL=EleutherAI/deep-ignorance-unfiltered-cb
	make eval_hf MODEL=EleutherAI/deep-ignorance-e2e-strong-filter-cb
	make eval_hf MODEL=EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal-cb

	# CB-Lat Models
	make eval_hf MODEL=EleutherAI/deep-ignorance-unfiltered-cb-lat
	make eval_hf MODEL=EleutherAI/deep-ignorance-e2e-strong-filter-cb-lat
	make eval_hf MODEL=EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal-cb-lat

	# Pretraining Stage Models
	make eval_hf MODEL=EleutherAI/deep-ignorance-pretraining-stage-unfiltered
	make eval_hf MODEL=EleutherAI/deep-ignorance-pretraining-stage-strong-filter
	make eval_hf MODEL=EleutherAI/deep-ignorance-pretraining-stage-weak-filter
