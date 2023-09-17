#! /usr/bin/env python

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.load_datasets import get_clear_dataset

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--readability', type=str, help='Target Flesch readability ease score')
parser.add_argument('--cache_dir', type=str, help='Load path to huggingface cache dir')
parser.add_argument('--data_path', type=str, help='Load path to CLEAR dataset')
parser.add_argument('--save_path', type=str, help='Load path to save results with paraphrases')

def get_prompt(r):
    if r == '95':
        prompt = "Paraphrase this document for 5th grade school level (US). It should be very easy to read and easily understood by an average 11-year old student."
    elif r == '85':
        prompt = "Paraphrase this document for 6th grade school level (US). It should be easy to read and conversational English for consumers."
    elif r == '75':
        prompt = "Paraphrase this document for 7th grade school level (US). It should be fairly easy to read."
    elif r == '65':
        prompt = "Paraphrase this document for 8th/9th grade school level (US). It should be plain English and easily understood by 13- to 15-year-old students."
    elif r == '55':
        prompt = "Paraphrase this document for 10th-12th grade school level (US). It should be fairly difficult to read."
    elif r == '40':
        prompt = "Paraphrase this document for college level (US). It should be difficult to read."
    elif r == '20':
        prompt = "Paraphrase this document for college graduate level (US). It should be very difficult to read and best understood by university graduates."
    elif r == '5':
        prompt = "Paraphrase this document for a professional. It should be extremely difficult to read and best understood by university graduates."
    else:
        print("ERROR ERROR ERROR")
    return prompt

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):

    seed_val = 42
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = get_default_device()

    with open(args.save_path) as f:
        expanded_examples = json.load(f)

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.half()
    model.eval()
    model.to(device)

    start_point = len(expanded_examples)

    contexts = get_clear_dataset(args.data_path)
    req = get_prompt(args.readability)
    batch_examples = []
    for count, item in enumerate(contexts):
        if count < start_point: continue
        print(count)
        # get paraphrase here
        prompt = item + "\n\n" + req 
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],
                top_k=10,
                do_sample=False,
                max_new_tokens=None
            )
            output_tokens = output[0]
            input_tokens = inputs.input_ids[0]
            new_tokens = output_tokens[len(input_tokens):]
            assert torch.equal(output_tokens[:len(input_tokens)], input_tokens)
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        curr_example = {'original':item, 'paraphrase':generated_text}
        batch_examples.append(curr_example)
        if len(batch_examples) == 1:
            expanded_examples += batch_examples
            batch_examples = []
            with open(args.save_path, 'w') as f:
                json.dump(expanded_examples, f)
            print("Saved up to:", count)
            print("----------------------")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



