CUDA_VISIBLE_DEVICES=0,1 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=1 src/open_r1/grpo.py \
    --config recipes/grpo/config_simple_rl.yaml \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_device 'cuda:0' \
    --use_peft \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --wandb_project 'grpo-7b-base' \



