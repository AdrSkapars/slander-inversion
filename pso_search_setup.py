import operator
import random
import pickle

import numpy
import math

import torch.nn.functional as F
import torch
from deap import base, creator, tools

import gc

from datetime import datetime, timedelta
import time

# transformers.logging.set_verbosity_error()

# from deap import benchmarks



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
    smin=None, smax=None, best=None)

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def generate_static(size, smin, smax, p_static):
    part = creator.Particle(p_static)
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))
    

def eval_inds(emb_list, in_batch_size, target_tokens, t5_batch_size, prompt_template=None, max_input_len=15, target_tok_id=0, ind_evals=None):
    # global autoencoder

    # Have to convert to list of strings to stay same with previous code
    emb_list_list = [list([e]) for e in emb_list]
    emb_tensor = torch.Tensor(emb_list_list).to("cuda")
    string_list = t5_decode(emb_tensor, max_out_length=max_input_len, t5_batch_size=t5_batch_size)
    # del emb_tensor
    # # del autoencoder
    # gc.collect()
    # torch.cuda.empty_cache()
    # gc.collect()
    # torch.cuda.empty_cache()
    # # autoencoder = BottleneckT5Autoencoder(model_path='thesephist/contra-bottleneck-t5-small-wikipedia', device=device)
    # print("cleaned")

    if prompt_template is not None:
        string_list = [prompt_template(s) for s in string_list]

    num_strings = len(string_list)
    batch_size = min(in_batch_size, num_strings)
    # Initialise batch
    batch_string_ids = list(range(0,batch_size))
    next_string_to_batch = batch_size

    # Initialise string trackers
    if target_tok_id == 0 or ind_evals == None:
        string_probabilities = [1]*num_strings
        string_target_tok_id = [0]*num_strings
    else:
        # Individuals have already gotten probabilities for some tokens
        string_probabilities = [eval for eval in ind_evals]
        string_target_tok_id = [target_tok_id for _ in ind_evals]

    successful_string_ids = []

    # Go until there are no more strings being checked nor yet to be checked
    empty_batch_slots = 0
    counter = 0
    while empty_batch_slots != batch_size:
        # Get batch ready for inference
        batch_strings = []
        for b_id in range(batch_size):
            if b_id >= num_strings:
                break
            s_id = batch_string_ids[b_id]
            if s_id == -1:
                continue
            # Need to add target tokens checked so far to end of string
            b_string = string_list[batch_string_ids[b_id]]
            b_string += tokenizer.decode(target_tokens[:string_target_tok_id[s_id]])
            batch_strings.append(b_string)

        # Do the inference
        input_ids = tokenizer(batch_strings, return_tensors="pt", padding=True).to("cuda")
        generate_output = model.generate(**input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=300, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, pad_token_id=eos_token,)
        # generate_output = model.generate(**input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)

        # Update validity of strings based on inference
        empty = 0
        for b_id in range(batch_size):
            s_id = batch_string_ids[b_id]
            if s_id == -1:
                empty += 1
                continue
            curr_target_tok = target_tokens[string_target_tok_id[s_id]]
            prob = generate_output.scores[0][b_id-empty][curr_target_tok]

            if not (math.isinf(prob) and prob < 0):
                # String still viable candidate
                # Update probability
                True_distribution = F.softmax(generate_output.scores[0][b_id-empty], dim=0)
                string_probabilities[s_id] *= float(True_distribution[curr_target_tok])

                # Prepare to check for next target token
                string_target_tok_id[s_id] += 1
                if string_target_tok_id[s_id] == len(target_tokens):
                    # Have already checked all the target tokens
                    successful_string_ids.append(s_id)
                    # Finished so pretend string is not viable so it gets replaced
                    prob = -math.inf
            else:
                # String not a viable candidate
                # Update probability
                string_probabilities[s_id] = 0

            if math.isinf(prob) and prob < 0:
                # Replace non-viable and finished strings in batch
                if next_string_to_batch >= num_strings:
                    # No new strings to start checking
                    batch_string_ids[b_id] = -1
                    empty_batch_slots += 1
                else:
                    batch_string_ids[b_id] = next_string_to_batch
                    next_string_to_batch += 1
        counter += 1

    for part, prob in zip(emb_list,string_probabilities):
        part.fitness.values = (prob,)
        

def initialise_embed_pop(start_population_as, sample_x, max_input_len, num_inputs):
    if start_population_as == "random":
        pop = toolbox.population(n=num_inputs)
    elif start_population_as == "random_dataset":
        pop = []
        random_dataset = rand_data_model_generate(num_data=num_inputs, max_new_tokens=max_input_len)
        for n in range(num_inputs):
            one_embedding = t5_encode(random_dataset[n]).tolist()[0]
            pop.append(toolbox.particle_static(p_static=one_embedding))
    elif start_population_as == "random_fluent":
        pop = []
        random_fluent = fluent_model_generate(num_generate=num_inputs, max_new_tokens=max_input_len)
        for n in range(num_inputs):
            one_embedding = t5_encode(random_fluent[n]).tolist()[0]
            pop.append(toolbox.particle_static(p_static=one_embedding))
    elif start_population_as == "random_fluent_output":
        pop = []
        random_fluent_output = fluent_model_generate(seed=assistant_text(sample_x), num_generate=num_inputs, max_new_tokens=max_input_len)
        for n in range(num_inputs):
            one_embedding = t5_encode(random_fluent_output[n]).tolist()[0]
            pop.append(toolbox.particle_static(p_static=one_embedding))
    elif start_population_as == "output":
        pop = []
        one_embedding = t5_encode(assistant_text(sample_x)).tolist()[0]
        for n in range(num_inputs):
            pop.append(toolbox.particle_static(p_static=one_embedding))
    elif start_population_as == "output_paraphrase":
        pop = []
        output_paraphrase = paraphrase_model_generate(assistant_text(sample_x), num_paraphrase=num_inputs, max_new_tokens=max_input_len)
        for n in range(num_inputs):
            one_embedding = t5_encode(output_paraphrase[n]).tolist()[0]
            pop.append(toolbox.particle_static(p_static=one_embedding))
    elif start_population_as == "output_synonym":
        pop = []
        output_synonym = synonyms_model_generate(assistant_text(sample_x), num_synonyms=num_inputs, max_new_tokens=max_input_len)
        for n in range(num_inputs):
            one_embedding = t5_encode(output_synonym[n]).tolist()[0]
            pop.append(toolbox.particle_static(p_static=one_embedding))
    elif start_population_as == "inversion":
        pop = []
        inversion = inversion_model_generate(assistant_text(sample_x), num_inversions=1, max_new_tokens=max_input_len, sample=False)[0]
        one_embedding = t5_encode(inversion).tolist()[0]
        for n in range(num_inputs):
            pop.append(toolbox.particle_static(p_static=one_embedding))
    elif start_population_as == "inversion_sample":
        pop = []
        inversion_sample = inversion_model_generate(assistant_text(sample_x), num_inversions=num_inputs, max_new_tokens=max_input_len, sample=True)
        for n in range(num_inputs):
            one_embedding = t5_encode(inversion_sample[n]).tolist()[0]
            pop.append(toolbox.particle_static(p_static=one_embedding))

    return pop


