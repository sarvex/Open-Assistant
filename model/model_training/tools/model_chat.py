#!/usr/bin/env python3
"""

A very simple script to test model locally


"""
import argparse
from enum import Enum
from typing import List, Tuple

import torch
from model_training.custom_datasets.formatting import QA_SPECIAL_TOKENS
from model_training.utils.utils import _strtobool
from tokenizers import pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatRole(str, Enum):
    system = "<|system|>"
    prompter = "<|prompter|>"
    assistant = "<|assistant|>"


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--bot_name", type=str, default="Joi", help="Use this when your format isn't in OA format")
parser.add_argument("--format", type=str, default="v2")
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--top_k", type=int, default=40)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--do-sample", type=_strtobool, default=True)
parser.add_argument("--per-digit-tokens", action="store_true")
args = parser.parse_args()

bot_name: str = args.bot_name
model_name: str = args.model_path
method: str = args.format


def talk(human_input: str, history: List[Tuple[str, str]], sep_token: str, prefix=""):
    histories = []
    if method == "v2":
        histories.extend(
            f'{QA_SPECIAL_TOKENS["Question"]}{question}{QA_SPECIAL_TOKENS["Answer"]}{answer}'
            for question, answer in history
        )
        prefix = "<prefix>You are a helpful assistant called Joi trained by OpenAssistant on large corpus of data, you will now help user to answer the question as concise as possible</prefix>"
        if histories:
            prefix += sep_token.join(histories)
            # add sep at the end
            prefix += sep_token
        prefix += f'{QA_SPECIAL_TOKENS["Question"]}{human_input}{QA_SPECIAL_TOKENS["Answer"]}'
    elif method == "v2.5":
        # personality = "You are a helpful assistant called Joi, you are a smart and helpful bot."
        # prefix = f"{ChatRole.system}{personality}{SeqToken.end}"
        histories.extend(
            f"{ChatRole.prompter}{question}</s>"
            + f"{ChatRole.assistant}{answer}</s>"
            for question, answer in history
        )
        if histories:
            prefix += "".join(histories)
            # add sep at the end
        prefix += f"{ChatRole.prompter}{human_input}</s>{ChatRole.assistant}"
    else:
        histories.extend(
            f"User: {question}" + f"\n\n{bot_name}: " + answer + "\n"
            for question, answer in history
        )
        if histories:
            prefix += "\n".join(histories)
        prefix += "\nUser: " + human_input + f"\n\n{bot_name}: "

    return prefix


def process_output(output):
    if method == "v2":
        answer = output.split(QA_SPECIAL_TOKENS["Answer"])[-1]
        answer = answer.split("</s>")[0].replace("<|endoftext|>", "").lstrip().split(QA_SPECIAL_TOKENS["Answer"])[0]
    elif method == "v2.5":
        answer = output.split(f"{ChatRole.assistant}")[-1]
    else:
        answer = output.split(f"\n\n{bot_name}:")[-1]
        answer = (
            answer.split("</s>")[0]
            .replace("<|endoftext|>", "")
            .lstrip()
            .split(f"\n\n{bot_name}:")[0]
        )
    return answer


tokenizer = AutoTokenizer.from_pretrained(model_name)
if method != "v2":
    tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

model.eval().cuda()

if args.per_digit_tokens:
    tokenizer._tokenizer.pre_processor = pre_tokenizers.Digits(True)

model = AutoModelForCausalLM.from_pretrained(model_name).half().eval().cuda()

if __name__ == "__main__":
    histories = []
    prefix = ""
    while True:
        print(">", end=" ")
        try:
            prompt = input()
        except (EOFError, KeyboardInterrupt):  # Catch ctrl+d and ctrl+c respectively
            print()
            break
        if prompt == "!reset":
            histories = []
        else:
            input_text = talk(prompt, histories, prefix)
            inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(0)
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            outputs = model.generate(
                **inputs,
                early_stopping=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                top_k=args.top_k,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
            output = tokenizer.decode(outputs[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
            reply = process_output(output, method, bot_name)

            if len(reply) != 0:
                print(reply)
                histories.append((prompt, reply))
            else:
                print("empty token")
