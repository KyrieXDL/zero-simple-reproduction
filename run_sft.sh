CUDA_VISIBLE_DEVICES=1 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=1 --main_process_port=29888 src/open_r1/sft.py \
    --config recipes/sft/config_simple_sft.yaml \
    --use_peft \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --wandb_project 'openr1_sft' \

