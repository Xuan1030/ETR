#!/bin/bash
set -x
export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(torch.__path__[0], 'lib'))"):$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT_MS=0     # disables timeout completely




python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/workspace/verl/data/deepmath_subset/train.parquet \
  data.val_files=/workspace/verl/data/deepmath_subset/test.parquet \
  data.train_batch_size=32 \
  data.max_prompt_length=512 \
  data.max_response_length=16384 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.rollout.load_format=dummy_hf \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.005 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0001 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=17408 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='verl_ablation_reward' \
  trainer.experiment_name='7B_no_momentum' \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=300 \
  trainer.test_freq=5 \
  trainer.default_local_dir='checkpoints/${trainer.project_name}/${trainer.experiment_name}' \
  trainer.total_epochs=1 $@
