import random
import numpy
import pickle

from deap import base, creator, tools #, algorithms

import math
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import time


def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring



def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__,
             gens_so_far=0, experiment_start_time=None, max_minutes=10, stop_early=False):
    global using_main_model

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'time'] + (stats.fields if stats else [])

    # # Evaluate the individuals with an invalid fitness
    # invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    # fitnesses = toolbox.evaluate(ind_list=invalid_ind)
    # for ind, fit in zip(invalid_ind, fitnesses):
    #     ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    # record = stats.compile(population) if stats else {}
    # logbook.record(gen=0, nevals=len(invalid_ind), **record)
    # if verbose:
    #     print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        fitnesses = toolbox.evaluate(ind_list=invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        # logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        logbook.record(gen=(gen+gens_so_far), nevals=len(invalid_ind), time=timedelta(seconds=(time.process_time()-experiment_start_time)), **record)
        # if verbose:
            # print(logbook.stream)
            # if gen % 10 == 0:
            # if max([ind.fitness.values[0] for ind in population]) > 0:
            # print("Best so far:", [tokenizer.decode(tools.selBest(population, k=1)[0])])

        # Print best ones
        print(logbook.__str__().split("\n")[-1])
        best = tools.selBest(population, k=1)#000)
        best_fitnesses = [ind.fitness.values[0] for ind in best]
        print([[best_fitnesses[i], [tokenizer.decode(best[i]).replace("'", "\\'").replace('"', '\\"')]] for i in range(len(best))])

        # Check if should be stopping
        if stop_early and best_fitnesses[0] == 1:
            return population, logbook, True
        # if timedelta(seconds=(time.process_time()-experiment_start_time)) >= timedelta(minutes=max_minutes):
        #     return population, logbook, True

        # Check if should be switching from intial model to main model, if havent already
        if not using_main_model:
            if initial_model_based_on == "gens":
                if (gen+gens_so_far) >= initial_model_param:
                    set_model_params(main_model_name)
                    using_main_model = True
            elif initial_model_based_on == "max_score":
                if best_fitnesses[0] >= initial_model_param:
                    set_model_params(main_model_name)
                    using_main_model = True

    return population, logbook, False


def eval_inds(ind_list, in_batch_size, target_tokens, prompt_template=None, target_tok_id=0):
    # Have to convert to list of strings to stay same with previous code
    string_list = tokenizer.batch_decode(ind_list) #, skip_special_tokens=True)
    if prompt_template is not None:
        string_list = [prompt_template(s) for s in string_list]

    num_strings = len(string_list)
    batch_size = min(in_batch_size, num_strings)
    # Initialise batch
    batch_string_ids = list(range(0,batch_size))
    next_string_to_batch = batch_size

    # Initialise string trackers
    if target_tok_id == 0:
        string_probabilities = [1]*num_strings
        string_target_tok_id = [0]*num_strings
    else:
        # Individuals have already gotten probabilities for some tokens
        string_probabilities = [ind.fitness.values[0] if ind.fitness.valid else 1 for ind in ind_list]
        string_target_tok_id = [target_tok_id if ind.fitness.valid else 0 for ind in ind_list]

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

    ind_probabilities = []
    for prob in string_probabilities:
        ind_probabilities.append((prob,))
    return ind_probabilities


def custom_mutate(individual, indpb, bounded):
    # Keep track of elements index offset since inserts and deletes move it around
    offset = 0
    for i in range(len(individual)):
        # Each element has independent probability of recieving mutation
        if random.random() >= indpb:
            continue

        # Which mutations can be performed depends on the length of the individual
        types = [1,2,3,4]
        if len(individual) == 0:
            types.remove(1) # modify
            types.remove(3) # delete
            types.remove(4) # swap
        elif len(individual) == 1:
            types.remove(3) # delete
            types.remove(4) # swap
        elif bounded and len(individual) == max_input_len:
            types.remove(2) # insert

        # Apply just one of the available mutations to this element
        choice = random.choice(types)
        ind = i + offset
        if choice == 1: # modify
            individual[ind] = random.randint(0, vocab_size-1)
        elif choice == 2: # insert
            individual.insert(ind, random.randint(0, vocab_size-1))
            offset += 1
        elif choice == 3: # delete
            del individual[ind]
            offset -= 1
        elif choice == 4: # swap
            ind2 = (ind + random.randint(1,len(individual)-1)) % len(individual)
            temp = individual[ind]
            individual[ind] = individual[ind2]
            individual[ind2] = temp

    return individual,


def initialise_text_pop(start_population_as, sample_x, max_input_len, num_inputs):
    if start_population_as == "random":
        pop = [toolbox.individual(n=random.randint(1,max_input_len)) for _ in range(num_inputs)]
    elif start_population_as == "random_dataset":
        random_dataset = rand_data_model_generate(num_data=num_inputs, max_new_tokens=max_input_len)
        pop = [creator.Individual(tokenizer(random_dataset[i]).input_ids) for i in range(num_inputs)]
    elif start_population_as == "random_fluent":
        random_fluent = fluent_model_generate(num_generate=num_inputs, max_new_tokens=max_input_len)
        pop = [creator.Individual(tokenizer(random_fluent[i]).input_ids) for i in range(num_inputs)]
    elif start_population_as == "random_fluent_output":
        random_fluent_output = fluent_model_generate(seed=assistant_text(sample_x), num_generate=num_inputs, max_new_tokens=max_input_len)
        pop = [creator.Individual(tokenizer(random_fluent_output[i]).input_ids) for i in range(num_inputs)]
    elif start_population_as == "output":
        target_seq = prefix + assistant_text(sample_x)
        target_tokens = tokenizer(target_seq, return_tensors='pt').input_ids[0]
        target_tokens = target_tokens[remove_prefix:]
        output = target_tokens.tolist()[:max_input_len]
        pop = [creator.Individual(output) for _ in range(num_inputs)]
    elif start_population_as == "output_paraphrase":
        output_paraphrase = paraphrase_model_generate(assistant_text(sample_x), num_paraphrase=num_inputs, max_new_tokens=max_input_len)
        pop = [creator.Individual(tokenizer(output_paraphrase[i]).input_ids) for i in range(num_inputs)]
    elif start_population_as == "output_synonym":
        output_synonym = synonyms_model_generate(assistant_text(sample_x), num_synonyms=num_inputs, max_new_tokens=max_input_len)
        pop = [creator.Individual(tokenizer(output_synonym[i]).input_ids) for i in range(num_inputs)]
    elif start_population_as == "inversion":
        inversion = tokenizer(inversion_model_generate(assistant_text(sample_x), num_inversions=1, max_new_tokens=max_input_len, sample=False)).input_ids
        pop = [creator.Individual(inversion) for _ in range(num_inputs)]
    elif start_population_as == "inversion_sample":
        inversion_sample = inversion_model_generate(assistant_text(sample_x), num_inversions=num_inputs, max_new_tokens=max_input_len, sample=True)
        pop = [creator.Individual(tokenizer(inversion_sample[i]).input_ids) for i in range(num_inputs)]

    return pop


