# NVdB Eval

This repository contains the experimental code, notebooks, and resources used in a research workflow centered on lexical competence evaluation for Italian through dictionary-entry generation.

## Related paper

This work is related to the paper:

> Do LLMs Know Basic Italian? Evaluating Lexical Competence through Dictionary Entry Generation against the De Mauro Basic Vocabulary
>
> Authors: Manuel Gozzi, Guido Vetere, Francesca Fallucchi
>
> Conference: CLiC-it 2026

The repository supports the analysis, sampling, generation, and evaluation procedures described in that study.

## What is in this repository

- Notebooks for dataset sampling, exploratory analysis, and false-positive analysis
- Python source code for batch generation and evaluation workflows
- Resources derived from Sensocomune and generated model outputs

## Project structure

- `notebooks/` — analysis notebooks and exploratory experiments
- `src/` — Python modules for the experimental pipeline
- `resources/` — input corpora, sampled datasets, and generated outputs

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

Install dependencies with:

```bash
uv sync
# Or
uv sync --group dev
```

Launch the notebooks with:

```bash
uv run jupyter lab notebooks
```

## Reproducibility notes

The notebooks are intended to be read and executed in order where relevant. The repository includes both source code and generated artifacts so that experiments can be inspected and reproduced with minimal additional setup.

## Citation and reuse

If you reuse this repository or any of its contents in a publication, presentation, or educational context, please cite the associated paper and clearly attribute the work.

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
