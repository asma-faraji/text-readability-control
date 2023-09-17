#! /usr/bin/env python

import argparse
import json
import time
import openai

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--api_key', type=str, help='Load OpenAI API key')
parser.add_argument('--readability', type=str, help='Target Flesch readability ease score')
parser.add_argument('--data_path', type=str, help='Load path to step 1 dataset')
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

def main(args):
    openai.api_key = args.api_key

    ##Assume file doesn't exits
    with open(args.save_path) as f:
        expanded_examples = json.load(f)

    start_point = len(expanded_examples)

    with open(args.data_path, 'r') as f:
        paraphrases1 = json.load(f)

    req = get_prompt(args.readability)
    batch_examples = []
    for count, item in enumerate(paraphrases1):
        if count < start_point: continue
        print(count)
        # get paraphrase here
        prompt = item['paraphrase'] + "\n\n" + req 
        model = "gpt-3.5-turbo"
        response = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
        generated_text = response.choices[0].message.content.strip()
        curr_example = {'original':item['original'], 'paraphrase':generated_text}
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

    for count in range(1,100):
        try:
            main(args)
            time.sleep(1)
        except openai.error.RateLimitError:
            print("openai.error.RateLimitError... #{}".format(count))
            print("restart in 10 seconds")
            time.sleep(10)
        except openai.error.ServiceUnavailableError:
            print("openai.error.ServiceUnavailableError... #{}".format(count))
            print("restart in 10 seconds")
            time.sleep(10)
        except openai.error.APIError:
            print("openai.error.APIError... #{}".format(count))
            print("restart in 20 seconds")
            time.sleep(20)
        except openai.error.Timeout:
            print("openai.error.TimeoutError... #{}".format(count))
            print("restart in 20 seconds")
            time.sleep(20)