# Copyright © 2023-2024 Apple Inc.

import argparse
import json

import mlx.core as mx

from .generate import stream_generate
from .models.cache import make_prompt_cache
from .sample_utils import make_sampler
from .utils import load, sharded_load

DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.0
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=DEFAULT_XTC_PROBABILITY,
        help="Probability of XTC sampling to happen each next token",
    )
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=0.0,
        help="Thresold the probs of each next token candidate to be sampled by XTC",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    # Grammar-constrained generation options
    grammar_group = parser.add_mutually_exclusive_group()
    grammar_group.add_argument(
        "--json-schema",
        type=str,
        default=None,
        help="JSON schema to constrain output (as JSON string or @file.json)",
    )
    grammar_group.add_argument(
        "--grammar",
        type=str,
        default=None,
        help="Lark grammar string to constrain output",
    )
    grammar_group.add_argument(
        "--regex",
        type=str,
        default=None,
        help="Regular expression to constrain output",
    )
    grammar_group.add_argument(
        "--tools",
        type=str,
        default=None,
        help="Tool definitions for grammar-constrained tool calling "
        "(as JSON string or @file.json). Forces tool call output.",
    )
    grammar_group.add_argument(
        "--choices",
        type=str,
        nargs="+",
        default=None,
        help="List of allowed output strings",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    group = mx.distributed.init()
    rank = group.rank()
    pipeline_group = group if args.pipeline else None
    tensor_group = group if not args.pipeline else None

    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    mx.random.seed(args.seed)

    if group.size() > 1:
        if args.adapter_path:
            parser.error("Adapters not supported in distributed mode")
        model, tokenizer = sharded_load(args.model, pipeline_group, tensor_group)
    else:
        model, tokenizer = load(
            args.model,
            adapter_path=args.adapter_path,
            tokenizer_config={
                "trust_remote_code": True if args.trust_remote_code else None
            },
        )

    def print_help():
        rprint("The command list:")
        rprint("- 'q' to exit")
        rprint("- 'r' to reset the chat")
        rprint("- 'h' to display these commands")

    # Build grammar logits processor if requested
    grammar_processor = None
    json_schema = args.json_schema
    if json_schema is not None and json_schema.startswith("@"):
        with open(json_schema[1:]) as f:
            json_schema = f.read()
    tools_json = args.tools
    if tools_json is not None and tools_json.startswith("@"):
        with open(tools_json[1:]) as f:
            tools_json = f.read()
    has_grammar = any([
        json_schema is not None,
        args.grammar is not None,
        args.regex is not None,
        args.choices is not None,
        tools_json is not None,
    ])
    if has_grammar:
        from .sample_utils import make_grammar_logits_processor

        json_schema_dict = json.loads(json_schema) if json_schema else None
        tools_list = json.loads(tools_json) if tools_json else None
        if tools_list is not None:
            tools_list = [
                t["function"] if t.get("type") == "function" and "function" in t else t
                for t in tools_list
            ]
        grammar_processor = make_grammar_logits_processor(
            tokenizer,
            json_schema=json_schema_dict,
            grammar=args.grammar,
            regex=args.regex,
            choices=args.choices,
            tools=tools_list,
        )

    rprint(f"[INFO] Starting chat session with {args.model}.")
    if grammar_processor is not None:
        rprint("[INFO] Grammar constraints active.")
    print_help()
    prompt_cache = make_prompt_cache(model, args.max_kv_size)
    while True:
        query = input(">> " if rank == 0 else "")
        if query == "q":
            break
        if query == "r":
            prompt_cache = make_prompt_cache(model, args.max_kv_size)
            continue
        if query == "h":
            print_help()
            continue
        messages = []
        if args.system_prompt is not None:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": query})
        # Include tool definitions in the chat template if --tools is used
        chat_tools = None
        if args.tools is not None:
            t = args.tools
            if t.startswith("@"):
                with open(t[1:]) as f:
                    t = f.read()
            chat_tools = json.loads(t)
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tools=chat_tools,
        )
        logits_processors = None
        if grammar_processor is not None:
            grammar_processor.reset()
            logits_processors = [grammar_processor]
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(
                args.temp,
                args.top_p,
                xtc_threshold=args.xtc_threshold,
                xtc_probability=args.xtc_probability,
                xtc_special_tokens=(
                    tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
                ),
            ),
            prompt_cache=prompt_cache,
            logits_processors=logits_processors,
        ):
            rprint(response.text, flush=True, end="")
        rprint()


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.chat...` directly is deprecated."
        " Use `mlx_lm.chat...` or `python -m mlx_lm chat ...` instead."
    )
    main()
