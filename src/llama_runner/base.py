"""
This module provides functionality for generating text completions using a
pre-trained Llama model.

The main function in this module is `main()`, which takes a list of prompts and
generates text completions for each prompt using a pre-trained Llama model. The
generated completions are printed to the console.

This module requires the `llama` package to be installed.

Example usage:
    python text_completion.py --prompts "The quick brown fox" "Once upon a time"
    --ckpt_dir /path/to/checkpoint --tokenizer_path /path/to/tokenizer

For more information on the available command-line arguments, run `python
text_completion.py --help`.
"""

from llama import Llama, generation


def text_completion(
    prompts,
    ckpt_dir,
    tokenizer_path,
    temperature=0.6,
    top_p=0.9,
    max_seq_len=128,
    max_gen_len=64,
    max_batch_size=4,
) -> list[generation.CompletionPrediction]:
    """
    Generates text completions for a given set of prompts using a pre-trained Llama model.

    Args:
        prompts: A list of prompts to generate completions for.
        ckpt_dir: The path to the directory containing the pre-trained Llama model checkpoint.
        tokenizer_path: The path to the tokenizer used by the pre-trained Llama model.
        temperature: The temperature to use when generating completions. Defaults to 0.6.
        top_p: The top-p value to use when generating completions. Defaults to 0.9.
        max_seq_len: The maximum sequence length to use when generating completions. Defaults to 128.
        max_gen_len: The maximum length of generated completions. Defaults to 64.
        max_batch_size: The maximum batch size to use when generating completions. Defaults to 4.
    Returns:
        None
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

    return results
