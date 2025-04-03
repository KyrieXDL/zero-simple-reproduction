NUM_GPUS=1 # Set to 8 for 32B and 70B models
#MODEL='/data/xdl/pretrained_models/nlp/qwen2.5-math-1.5b'
MODEL='/data/xdl/projects/openr1/checkpoints_drgrpo-1.5b-math-base/checkpoint-375'
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=4096,max_num_batched_tokens=4096,gpu_memory_utilization=0.75,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:4096,temperature:0,top_p:0.95,top_k:1}"
OUTPUT_DIR="data/evals/drgrpo-qwen2.5-1.5b"

lighteval vllm $MODEL_ARGS "custom|aime24|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR


lighteval vllm $MODEL_ARGS "custom|math_500|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

lighteval vllm $MODEL_ARGS "extended|olympiad_bench|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR