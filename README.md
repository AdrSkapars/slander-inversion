# Text-to-Text Inversion of Language Models
This repo contains the filtered datasets, notebooks and scripts needed to replicate our work.

To replicate initial search algorithm comparisons, you must run one file per row in the table:
```
python3 ga_regular.py
python3 ga_progressive.py
python3 pso_regular.py
python3 pso_progressive.py
```

To replicate secondary model comparisons, you must run one file per line in the figure:
```
python3 gpt_baseline.py
python3 gpt_optimal.py
python3 llama_baseline.py
python3 llama_optimal.py
```

Custom experiments can be run with Notebook.ipynb:
- It is recommended that you use Google Colab
    - Otherwise do not run cells including "!pip install" statements and see the requirements.txt file instead
    - The T4 GPU environment is able to run both LLMs involved in our work
- The notebook is split into several sections
    - "Main Models" contains set up for target LLMs and the embedding model used for PSO (do not set up embedding model if using GA)
    - "Initialisation" contains set up needed for certain initialisations (only run code related to desired initialisation)
    - "Datasets" contains code for filtering the dataset from scratch as well as loading the local processed_dataset folders
    - "Algorithms" contains the code for GA and PSO
        - For both, you simply run the first blocks and everything under "Set Up"
        - Then, depending on if you want to run rregular search or progressive search, run either "Main loop - for dataset" or "Main loop - for dataset (progressive)"
        - The main hyperparameters are set just before algorithm functions and are given as reference below

```
# Run/ search params
max_input_len = 15  # Won't matter while bounded=False
num_inputs = 1000 # Population size
max_gens = 500 # Maximum itterations of optimisation to do 
max_minutes = None # Maximum minutes of optimisation to do 
stop_early = True # Stop if have maxed out objective function

# Initialisation params, just change the index
start_population_as = [
    "random",
    "random_dataset",
    "random_fluent",
    "random_fluent_output",
    "output",
    "output_paraphrase",
    "output_synonym",
    "inversion",
    "inversion_sample"
][0]

# Params for smaller model search before larger model, just change the index
initial_model_name = [None, "gpt2", "gpt2-large", "llama2"][1]
# Should switch to larger model be based on the current itteration or the best score achieved for smaller model
initial_model_based_on = ["gens", "max_score"][1]
initial_model_param = [[50, 100], [0.01, 0.1]][0][0]

# Model params, just change the index
main_model_name = ["gpt2", "llama2"][0]
if main_model_name == "gpt2":
    model = gpt2_model
    tokenizer = gpt2_tokenizer
    in_batch_size = 200#400
    prefix = " "
    remove_prefix = 0
    vocab_size = 50257
    prompt_template = lambda x: f"User: {x}\nAssistant:"
elif main_model_name == "llama2":
    model = llama2_model
    tokenizer = llama2_tokenizer
    in_batch_size = 200
    prefix = "\n\n"
    remove_prefix = 4
    vocab_size = 32000
    prompt_template = lambda x: f"[INST]{x}[/INST]\n\n"

# Proggressive params
# Should progressive search be based on the current itteration or the percentage of non zero inputs in the population
progressive_based_on = ["gens", "non_zero_scores"][0]
progressive_param = 500 # if "gens" chosen then this should match max_gens

```