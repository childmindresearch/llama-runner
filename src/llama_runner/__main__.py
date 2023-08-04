"""
This module is the entry point for the Llama Runner application. It uses the
Fire library to provide a command-line interface for the `text_completion`
function in the `base` module of the `llama_runner` package.
"""
import fire

from llama_runner import base

if __name__ == "__main__":
    fire.Fire(base.text_completion)
