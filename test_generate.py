from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen2ForCausalLM, GenerationConfig, Qwen2Tokenizer
from peft import PeftModel
from datasets import load_dataset
import argparse
import os

SYSTEM_PROMPT = "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags, i.e., <think>\n reasoning process here \n</think>\n summarizing the reasoning results here, and put your final answer within \\boxed{}."

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--lora_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='DigitalLearningGmbH/MATH-lighteval')

    args = parser.parse_args()

    # load dataset
    dataset = load_dataset(args.data_path, split='test')
    print(dataset)

    # model init
    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2ForCausalLM.from_pretrained(args.model_path)

    if os.path.exists(args.lora_path):
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
    model.to("cuda")

    print(model)

    genration_config = GenerationConfig(do_sample=False,
                                        pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
                                        repetition_penalty=1.1)

    # generate
    idx = 450
    problem = dataset[idx]['problem']
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        generation_config=genration_config,
        max_new_tokens=4096,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(prompt, '\n\n')
    print('*'*20, '\n\n', response)
    print('*'*20, '\n\n', dataset[idx]['solution'])
