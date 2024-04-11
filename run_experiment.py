import json


run_info_params = {
    "dataset_name": ["chatbot_arena", "rlhf_hh"][0],
    "num_examples": 30,
    # "time_per_example": "00:10:00",
    "gens_per_example": 200,
}
search_params = {
    "embed": {
        "search_type": "embedding_(pso)",
        "population_size": 1000,
        "max_individual_length": 64,
        "smin_smax":0.5,
        "phi1":2.0,
        "phi2":2.0,
    },
    "text": {
        "search_type": "text_(genetic)",
        "population_size": 1000,
        "initialisation_length": 15,
        "mate_prob": 0.3,
        "mutate_prob": 0.1,
        "selection_param": 15,
    },
}
progressive_params = {
    "order_of_splits": "normal", #"random", "ranked"
    "progressive_based_on": ["gens", "non_zero_scores"][0],
    "progressive_param": [None, 1, 25, 50, 100, 200, 300, 400][5],
}
initialisation_params = [
    "random",
    "random_dataset",
    "random_fluent",
    "random_fluent_output",
    "output",
    "output_paraphrase",
    "output_synonym",
    "inversion",
    "inversion_sample"
][2]
initial_model_params = {
    "model_name": ["gpt2-small", "gpt2-large"][0],
    "in_batch_size": 400,
    "top_k": [100, 300, 500][1],
    "top_p": [0.91, 0.95, 0.99][1],
    "switch_to_target_based_on": ["gens", "max_score"][1],
    "switch_trigger": [[50, 100] [0.01, 0.1]][0][0],
}
inversion_model_params = {
    "do_sample": [False, True][0],
    "temperature": [None,0.4,0.6,0.8,1][0],
    "top_k": [None,100, 300, 500][0],
    "top_p": [None,0.91, 0.95, 0.99][0],
}
semantic_model_params = {
    "model_name": ["contra_small", "contra_base", "contra_large", "contra_xl"][0],
    "batch_size": 200,
}
target_model_params1 = {
    "model_name": ["gpt2-small", "gpt2-large", "llama2"][0],
    "batch_size": [200, 400][1],
    "top_k": [100, 300, 500][1],
    "top_p": [0.91, 0.95, 0.99][1],
}

# ===========================================================
config = {
    "run_info": run_info_params,
    "search": search_params["text"],
    "progressive_fitness": [None, progressive_params][0],
    "initialisation": initialisation_params,
    "initial_model": [None, initial_model_params][0],
    "semantic_model": [None, semantic_model_params][0],
    "target_model": target_model_params1,
}

with open("experiment_config.json", "w") as file:
    json.dump(config, file)