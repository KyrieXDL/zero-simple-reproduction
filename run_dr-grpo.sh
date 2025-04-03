CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=1 --main_process_port=29688 src/open_r1/dr_grpo.py \
    --config recipes/grpo/config_simple_rl.yaml \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_device 'cuda:0' \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --wandb_project 'zero-drgrpo-1.5b-math-base' \
    --dataset_name ./data/Math-rl-12k \
    --output_dir /data/xdl/projects/openr1/checkpoints_drgrpo-1.5b-math-base \
    --model_name_or_path /data/xdl/pretrained_models/nlp/qwen2.5-math-1.5b \
    --max_completion_length 3000 \


