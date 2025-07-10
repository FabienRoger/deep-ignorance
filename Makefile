# Configuration
LM_EVAL_TASKS_PATH ?= /home/primeintellect/repos/filtering_for_danger/lm_eval_tasks
TASKS ?= wmdp_bio_categorized_mcqa,wmdp_bio_cloze_verified,mmlu,piqa,lambada,hellaswag

# Run evaluation directly on host
eval_hf:
ifndef MODEL
	$(error MODEL is required. Usage: make eval_hf MODEL=<model_name>)
endif
	lm_eval --model hf \
		--model_args pretrained=$(MODEL),dtype=bfloat16,parallelize=True,attn_implementation=flash_attention_2 \
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

eval_hf_final_models:
	# Baseline
	make eval_hf_docker MODEL=EleutherAI/camus

	# Blocklist Pretraining, Blocklist Annealing
	make eval_hf_docker MODEL=EleutherAI/stranger

	# Blocklist Pretraining, ModernBERT Annealing
	make eval_hf_docker MODEL=EleutherAI/sisyphus

	# ModernBERT Pretraining, ModernBERT Annealing
	make eval_hf_docker MODEL=EleutherAI/plague

	# ModernBERT Pretraining, Blocklist Annealing
	make eval_hf_docker MODEL=EleutherAI/absurd
