# zero-simple-reproduction
This project is a simple reproduction of deepseek-zero. And we utilized the Qwen2.5 LLM as the primary LLM.

This projest mainly refers to the [openr1](https://github.com/huggingface/open-r1/tree/main).

The Qwen2.5-7b-zero is obtained through grpo training on the train dataset of [MATH-lighteval](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval) using Qwen2.5-7b-base. And we used lora instead of full-training. The examples of Qwen2.5-7b-zero and Qwen2.5-7b-base are shown in the following figure.
<center>
    <img src="assets/example1.png" width="700">
</center>


The curves of loss and reward throughout the entire training process are shown in the following figure.
<center>
    <img src="assets/train_loss.png" width="850">
</center>


### Experiment Results
We used the greedy search decoding strategy. Run the following script to evaluate on the math datasets (aime2024, math-500, olympiad bench).
```shell
sh run_eval.sh
```

| Model                                    | AIME 2024 | MATH-500 | Olympiad Bench |
|:-----------------------------------------|:---------:|:--------:|:--------------:|
| Qwen-math-1.5B                           |   13.3    |   38.4   |      11.2      |
| Qwen-math-1.5B +  GRPO (Full training)   |           |          |                |
| Qwen-math-1.5B + Dr GRPO (Full training) |   13.3    |    69    |      18.2      |
| Qwen-math-1.5B + DAPO (Full training)    |           |          |                |
| Qwen-7B-base                             |    10     |   43.2   |     22.67      |
| Qwen-7B-base +  GRPO (Lora)              |    3.3    |   63.6   |     21.92      |
| Qwen-7B-base + Dr GRPO (Lora)            |           |          |                |
| Qwen-7B-base + DAPO (Lora)               |           |          |                |

### Environment
```shell
conda creat zero -n zero python=3.10
    
conda activate zero

pip install -r requirements.txt
```


### Train
Run this to training:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=1 src/open_r1/grpo.py \
    --config recipes/grpo/config_simple_rl.yaml \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_device 'cuda:0' \
    --use_peft \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --wandb_project 'openr1_grpo-7b-base' \
    
```


### Test
Run this to generate:
```shell
CUDA_VISIBLE_DEVICES=0 python test_generate.py \
  --model_path 'base_model_path' \
  --lora_path 'lora_adapter_path' \
  --data_path 'data_path' \
  
```
