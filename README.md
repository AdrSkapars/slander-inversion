# Was it Slander? Towards Exact Inversion of Generative Language Models

This repository contains the code and filtered datasets for the paper. Given an output of a language model, we search for the input that produced it, using only the probabilities the model assigns to the output. We compare two search methods:

- a genetic algorithm (GA) over tokens;
- particle swarm optimisation (PSO) over the embeddings of a T5 autoencoder.

Each method can use either the full objective or a progressive one, which reveals the target output one token at a time.

## Setup

```bash
pip install -r requirements.txt
```

The scripts assume a CUDA GPU. They load the filtered Chatbot Arena pairs from `processed_dataset/` (30 pairs) or `processed_dataset2/` (50 pairs, used for the model comparison).

## Replicating the experiments

Each script runs one configuration over the dataset and prints the search progress.

To compare the search algorithms, run one script per row of the comparison table:

```bash
python3 ga_regular.py
python3 ga_progressive.py
python3 pso_regular.py
python3 pso_progressive.py
```

To compare models, run one script per line of the model-comparison figure:

```bash
python3 gpt_baseline.py
python3 gpt_optimal.py
python3 llama_baseline.py
python3 llama_optimal.py
```

## Metropolis–Hastings sampling

These scripts are an extension that is not part of the paper. Instead of searching for the single most likely input, they run many Metropolis–Hastings chains that sample inputs in proportion to the probability that each one produces the target output. The text-based sampler is the counterpart of GA and edits tokens. The embedding-based sampler is the counterpart of PSO and adds noise to T5 embeddings. Both have a regular and a progressive version:

```bash
python3 mh_text_regular.py
python3 mh_text_progressive.py
python3 mh_embed_regular.py
python3 mh_embed_progressive.py
```

## Evaluation

`Evaluation.ipynb` computes the paper's metrics and figures from the printed output of the scripts, saved to a file (e.g. `python3 ga_regular.py > normal_1.txt`). The embedding similarity needs an `OPENAI_API_KEY`.

## Custom experiments

`Notebook.ipynb` contains the same code as the scripts in one place and is set up for Google Colab. Its T4 GPU runtime can run both models. Outside Colab, skip the `!pip install` cells and use `requirements.txt` instead. The notebook has four sections:

- **Main Models** loads the target models, plus the T5 autoencoder needed by PSO.
- **Initialisation** loads the models needed by some initialisation strategies.
- **Datasets** filters the dataset from scratch or loads the processed copies.
- **Algorithms** contains GA and PSO, each with a regular and a progressive main loop.

The main settings sit in the cell before each main loop:

- `start_population_as`: the initialisation strategy, one of `random`, `random_dataset`, `random_fluent`, `random_fluent_output`, `output`, `output_paraphrase`, `output_synonym`, `inversion` or `inversion_sample`. The `random_dataset` strategy samples tweets from `tweets.json`;
- `main_model_name`: the target model, `gpt2` or `llama2`;
- `initial_model_name`: an optional smaller model to search with before switching to the target;
- `progressive_based_on`: whether the progressive objective reveals the next token after a fixed number of iterations or once enough of the population scores above zero.

## Citation

```bibtex
@misc{skapars2024slanderexactinversiongenerative,
      title={Was it Slander? Towards Exact Inversion of Generative Language Models}, 
      author={Adrians Skapars and Edoardo Manino and Youcheng Sun and Lucas C. Cordeiro},
      year={2024},
      eprint={2407.11059},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2407.11059}, 
}
```
