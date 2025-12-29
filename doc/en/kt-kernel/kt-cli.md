# KT-CLI

> ⚠️ **Note:** This feature is currently under active development. Many functionalities are not yet complete and are being improved. Please stay tuned for updates.

## Design Philosophy

KT-CLI is designed to **minimize the burden of reading documentation**. Instead of requiring users to read lengthy docs, the CLI provides:

- **Interactive Mode**: Run commands without arguments to get step-by-step guided prompts
- **Direct Mode**: Pass arguments directly for automation and scripting
    > 💡 **Tip:** The arguments are fully compatible with the previous SGLang + KTransformers approach, so you can migrate seamlessly.

Simply run a command, and the CLI will interactively guide you through the process!

## Usage

You can check the usage by `kt --help`

```
kt [OPTIONS] COMMAND [ARGS]...
```

KTransformers CLI - A unified command-line interface for KTransformers.

## Options

| Option | Description |
|--------|-------------|
| `--help` | Show this message and exit. |

## Commands

| Command | Description |
|---------|-------------|
| `version` | Show version information |
| `chat` | Interactive chat with running model |
| `quant` | Quantize model weights |
| `bench` | Run full benchmark |
| `microbench` | Run micro-benchmark |
| `doctor` | Diagnose environment issues |
| `model` | Manage models and storage paths |
| `config` | Manage configuration |
| `sft` | Fine-tuning with LlamaFactory |
