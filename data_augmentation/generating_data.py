import sys
sys.path+=['.','..']

import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import openai
import json
from tqdm import tqdm, trange
from transformers import AutoTokenizer
import time
import string
import numpy as np
import re
from IPython import embed

from promptor_cognitive import ParaphrasedPrompter
from utils import get_finished_sample_ids
from vllm import LLM, SamplingParams

llama_chat_template = """
[INST] <<SYS>>
{instruction}
<</SYS>>

{input} [/INST]
""".strip('\n')

def get_qr_list(paraphrased_conv):
    pattern = re.compile(r'(Query\d+:|Response\d+:)')
    
    cleaned_text = re.sub(pattern, '', paraphrased_conv).strip()

    queries_responses = cleaned_text.split("\t")

    qr_list = [item.strip() for item in queries_responses if item]
    return qr_list

class MyLLM:

    def __init__(self, args):
        self.args = args
        self.url = None
        self.model = LLM(model=args.model)
        self.tokenizer = self.model.get_tokenizer()
        
        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0
        self.azure_filter_fail = 0


    def generate(self, prompt, max_tokens, stop=None, batching = False):
        args = self.args
        if max_tokens <= 0:
            self.prompt_exceed_max_length += 1
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, n=5, max_tokens = max_tokens)
        if batching:
            prompt = [llama_chat_template.format(instruction="",input=p) for p in prompt]
        else:
            prompt = llama_chat_template.format(instruction="",input=prompt)
        outputs = self.model.generate(prompt, sampling_params, use_tqdm=False)
        generation=[]
        for output in outputs:
            for o in output.outputs:
                generated_text = o.text
                generation.append(generated_text)
            
        return generation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_file", type=str, help="Path to the demo file")
    parser.add_argument("--output_file", type=str, help="Path to the output file")

    parser.add_argument("--eval_file", type=str, help="Path to the eval file")
    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")

    # ICL setting
    parser.add_argument("--shot", type=int, help="Number of ICL demonstrations")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")

    # Model and name
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--openai_api", type=bool, default=False, help="Whether to use OpenAI API")
    parser.add_argument("--baidu", type=bool, default=False, help="Whether to use OpenAI API")

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, default=1, help="Sample multiple answers.")

    args = parser.parse_args()
    print(args)

    if "turbo" in args.model:
        # ChatGPT has a longer max length
        args.max_length = 16384

    if "16k" in args.model:
        args.max_length = 16384
    elif "32k" in args.model:
        args.max_length = 32768
    elif "turbo" in args.model:
        args.max_length = 4096
    elif "gpt-4" in args.model:
        args.max_length = 8192
    elif "llama-2" in args.model.lower() or "llama2" in args.model.lower():
        args.max_length = 4096
    elif "qwen-7b" in args.model.lower():
        args.max_length = 32768
    elif "qwen-14b" in args.model.lower():
        args.max_length = 8192
    
    print(f"Set the model max length to {args.max_length} (if not correct, check the code)")
        

    # Load the model or setup the API
    llm = MyLLM(args)
    
    # Generate prompts
    np.random.seed(args.seed)

    # Load data
    demo_data = json.load(open(args.demo_file))
    eval_data = json.load(open(args.eval_file))

    head_prompt = ""
    promptor = ParaphrasedPrompter()
    head_prompt += promptor.demo_instruction

    train_ids = np.random.choice(len(demo_data), args.shot, replace=False)
    for train_id in train_ids:
        train_item = demo_data[train_id]
        head_prompt += promptor.make_demo_prompt(train_item)
        head_prompt += promptor.demo_sep
    
    head_prompt += promptor.instruction
    responses = []
    
    finished_samples = get_finished_sample_ids(args.output_file)
    prompts = []
    sample_ids = []
    expect_lens = {}
    
    for conv in eval_data[:]:
        conv_id = conv["conv_id"]
        context = ""
        history = []
        for i in range(len(conv['turns'])):
            turn = conv['turns'][i]
            turn_id = turn['turn_id']
            qprompt = promptor.make_query_prompt(turn_id, turn['question'])
            rprompt = promptor.make_response_prompt(turn_id, turn['response'])
            prompt = head_prompt
            prompt += "Original Conversation:\n"
            prompt += context + qprompt
            context += qprompt + rprompt
            sample_id = str(conv_id) + '_' + str(turn_id)
            expect_lens[sample_id] = len(history)+1
            if len(turn["pos_doc_ids"]) == 0:
                history.append(turn['question'])
                history.append(turn['response'])
                continue
            if sample_id in finished_samples:
                history.append(turn['question'])
                history.append(turn['response'])
                continue
            prompt += "\nStep 1: Comprehension Synthesis:\n"
            prompt += "[Identify key themes and intents of the conversation]\n"
            prompt += "Step 2: Associative Expansion:\n"
            prompt += "[Your associative diversification output here]\n"
            prompt += "Step 3: Conclusion:\n"
            prompt += "Paraphrased Conversation:\n"
            prompt += "[Your paraphrased conversation here]\n"
            prompt += "Only output the paraphrasing process and the paraphrased conversation.\n"
            prompts.append(prompt)
            sample_ids.append(sample_id)
            history.append(turn['question'])
            history.append(turn['response'])
    
    batch_size = 2
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    sid_batches = [sample_ids[i:i + batch_size] for i in range(0, len(sample_ids), batch_size)]
    
    with open(args.output_file, "a+") as f:
        for ind, batch in enumerate(tqdm(batches)):
            sample_id = sid_batches[ind][0]
            prompt_lens = [len(llm.tokenizer.tokenize(prompt)) for prompt in batch]
            max_lens = [min(args.max_new_tokens, args.max_length-prompt_len) for prompt_len in prompt_lens]
            
            paraphrased_convs = llm.generate(batch, min(max_lens), batching=True, expect_len=expect_lens[sample_id])
            sample_ids = sid_batches[ind]
            
            for i, paraphrased_conv in enumerate(paraphrased_convs):
                record = {}
                sample_id = sample_ids[i // 5]
                record['sample_id'] = sample_id
                record['paraphrased'] = paraphrased_conv
                responses.append(record)
                if sample_id not in finished_samples:
                    f.write(json.dumps(record))
                    f.write('\n')
                    f.flush()

if __name__ == "__main__":
    main()