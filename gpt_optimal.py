########################################################################
######################## LOAD GENERATIVE MODEL
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# GPT 2
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda").eval()

gpt2_tokenizer.padding_side = "left"
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
eos_token = gpt2_tokenizer(gpt2_tokenizer.eos_token).input_ids[0]
# gpt2_tokenizer.pad_token = gpt2_tokenizer.bos_token
gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id


# # LLAMA 2
# llama2_model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ",
#                                              device_map="auto",
#                                              trust_remote_code=False,
#                                              revision="main")

# llama2_tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ", use_fast=True)

# llama2_tokenizer.padding_side = "left"
# # llama2_tokenizer.pad_token = llama2_tokenizer.eos_token
# eos_token = llama2_tokenizer(llama2_tokenizer.bos_token).input_ids[0]
# llama2_tokenizer.pad_token = llama2_tokenizer.bos_token
# llama2_model.config.pad_token_id = llama2_model.config.eos_token_id



########################################################################
######################## MODEL FUNCTIONS
model = None
tokenizer = None
in_batch_size = None
prefix = None
remove_prefix = None
vocab_size = None
prompt_template = None

def set_model_params(model_name):
    global model, tokenizer, in_batch_size, prefix, remove_prefix, vocab_size, prompt_template

    if model_name == "gpt2":
        model = gpt2_model
        tokenizer = gpt2_tokenizer
        in_batch_size = 200#400
        prefix = " "
        remove_prefix = 0
        vocab_size = 50257
        prompt_template = lambda x: f"User: {x}\nAssistant:"

    elif model_name == "llama2":
        model = llama2_model
        tokenizer = llama2_tokenizer
        in_batch_size = 200
        prefix = "\n\n"
        remove_prefix = 4
        vocab_size = 32000
        prompt_template = lambda x: f"[INST]{x}[/INST]\n\n"


# Find probability of any sequence given context
def prob_of_sequence(context, test_seq):
    prompt_template = lambda x: f"User: {x}\nAssistant:"

    test_string = " " + test_seq
    test_tokens = tokenizer(test_string, return_tensors='pt').input_ids[0]
    # test_tokens = test_tokens[4:] # Remove '<s> \n\n'

    current_toks = []
    seq_prob = 1
    for t_tok in test_tokens:
        current_context = context + tokenizer.decode(current_toks)
        new_input_ids = tokenizer(current_context, return_tensors='pt').input_ids.cuda()
        new_generate_output = model.generate(inputs=new_input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=300, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, pad_token_id=eos_token,)
        new_generate_output_scores = F.softmax(new_generate_output.scores[0][0], dim=0)
        t_score = new_generate_output_scores[t_tok]

        # Just for test, distributions
#         distribution = []
#         nonzero_indices = torch.nonzero(new_generate_output_scores)
#         nonzero_values = new_generate_output_scores[nonzero_indices[:, 0]]
#         sorted_tensor = torch.sort(nonzero_values, descending=True)
#         sorted_indices = nonzero_indices[sorted_tensor.indices]
#         for i in range(len(sorted_indices)):
#             distribution.append([tokenizer.decode(sorted_indices[i][0]), int(sorted_indices[i][0]), round(float(sorted_tensor.values[i]),4)])
#         print(tokenizer.decode(t_tok), "\t-", distribution)

        if t_score == 0:
            # print("\nProbability of sequence: 0 %")
            return 0

        seq_prob *= float(t_score)
        current_toks.append(t_tok)

    # print("Probability of sequence:", seq_prob * 100, "%")
    return seq_prob


########################################################################
######################## LOAD INVERSION MODEL
# inv_model = AutoModelForSeq2SeqLM.from_pretrained("wentingzhao/inversion_seq2seq_baseline").eval().to("cuda")
inv_tokenizer = AutoTokenizer.from_pretrained("wentingzhao/inversion_seq2seq_baseline")

inv_tokenizer.padding_side = "left"
inv_tokenizer.pad_token = inv_tokenizer.eos_token
inv_model.config.pad_token_id = inv_model.config.eos_token_id
inv_eos_token = inv_tokenizer(inv_tokenizer.eos_token).input_ids[0]

def inversion_model_generate(target_output, num_inversions=1, batch_size=100, max_new_tokens=20, sample=False):
    # Load model
    inv_model = AutoModelForSeq2SeqLM.from_pretrained("wentingzhao/inversion_seq2seq_baseline").eval().to("cuda")

    outputs = []
    for i in range(0, num_inversions, batch_size):
        batch_s = min(batch_size, num_inversions-i)
        input_ids = inv_tokenizer([target_output]*batch_s, return_tensors="pt", padding=True).input_ids.to("cuda")

        if sample:
            output = inv_model.generate(
                inputs=input_ids,
                temperature=1,
                do_sample=True,
                top_p=0.99,
                top_k=500,
                max_new_tokens=max_new_tokens,
                pad_token_id=inv_eos_token,
            )
        else:
            output = inv_model.generate(
                inputs=input_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=inv_eos_token,
            )
        outputs += list(output)

    # Delete model
    inv_model = None
    torch.cuda.empty_cache()
    del inv_model
    torch.cuda.empty_cache()

    return [inv_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


########################################################################
######################## LOAD DATASETS
# Chatbot Arena
from datasets import load_dataset, load_from_disk

reloaded_data = load_from_disk("processed_dataset2")
processed_data = reloaded_data[:50]
user_text = lambda x: processed_data["conversation_a" if processed_data["winner"][x] == "model_a" else "conversation_b"][x][0]['content']
assistant_text = lambda x: processed_data["conversation_a" if processed_data["winner"][x] == "model_a" else "conversation_b"][x][1]['content']


########################################################################
######################## DEFINE ALGORITHMS

######################## Text-Based Genetic
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


#>>>>>>>>>>>>>>>>>>>>>>> Simple Algo Definitions
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
        best = tools.selBest(population, k=1000)
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



#>>>>>>>>>>>>>>>>>>>>>>> Evaluation Function
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


#>>>>>>>>>>>>>>>>>>>>>>> Mutate Function
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


#>>>>>>>>>>>>>>>>>>>>>>> Initialisation Function
def initialise_text_pop(start_population_as, sample_x, max_input_len, num_inputs):
    if start_population_as == "random":
        pop = [toolbox.individual(n=random.randint(1,max_input_len)) for _ in range(num_inputs)]
    # elif start_population_as == "random_dataset":
    #     random_dataset = rand_data_model_generate(num_data=num_inputs, max_new_tokens=max_input_len)
    #     pop = [creator.Individual(tokenizer(random_dataset[i]).input_ids) for i in range(num_inputs)]
    # elif start_population_as == "random_fluent":
    #     random_fluent = fluent_model_generate(num_generate=num_inputs, max_new_tokens=max_input_len)
    #     pop = [creator.Individual(tokenizer(random_fluent[i]).input_ids) for i in range(num_inputs)]
    # elif start_population_as == "random_fluent_output":
    #     random_fluent_output = fluent_model_generate(seed=assistant_text(sample_x), num_generate=num_inputs, max_new_tokens=max_input_len)
    #     pop = [creator.Individual(tokenizer(random_fluent_output[i]).input_ids) for i in range(num_inputs)]
    # elif start_population_as == "output":
    #     target_seq = prefix + assistant_text(sample_x)
    #     target_tokens = tokenizer(target_seq, return_tensors='pt').input_ids[0]
    #     target_tokens = target_tokens[remove_prefix:]
    #     output = target_tokens.tolist()[:max_input_len]
    #     pop = [creator.Individual(output) for _ in range(num_inputs)]
    # elif start_population_as == "output_paraphrase":
    #     output_paraphrase = paraphrase_model_generate(assistant_text(sample_x), num_paraphrase=num_inputs, max_new_tokens=max_input_len)
    #     pop = [creator.Individual(tokenizer(output_paraphrase[i]).input_ids) for i in range(num_inputs)]
    # elif start_population_as == "output_synonym":
    #     output_synonym = synonyms_model_generate(assistant_text(sample_x), num_synonyms=num_inputs, max_new_tokens=max_input_len)
    #     pop = [creator.Individual(tokenizer(output_synonym[i]).input_ids) for i in range(num_inputs)]
    # elif start_population_as == "inversion":
    #     inversion = tokenizer(inversion_model_generate(assistant_text(sample_x), num_inversions=1, max_new_tokens=max_input_len, sample=False)).input_ids[0]
    #     pop = [creator.Individual(inversion) for _ in range(num_inputs)]
    elif start_population_as == "inversion_sample":
        inversion_sample = inversion_model_generate(assistant_text(sample_x), num_inversions=num_inputs, max_new_tokens=max_input_len, sample=True)
        pop = [creator.Individual(tokenizer(inversion_sample[i]).input_ids) for i in range(num_inputs)]

    return pop


# #>>>>>>>>>>>>>>>>>>>>>>> Main Loop
# # Run/ search params
# max_input_len = 64 # Doesnt matter since bounded=False
# num_inputs = 1000
# max_gens = 500#200 #999 # unbounded
# max_minutes = None#10
# stop_early = True

# # Initialisation params
# start_population_as = [
#     "random",
#     "random_dataset",
#     "random_fluent",
#     "random_fluent_output",
#     "output",
#     "output_paraphrase",
#     "output_synonym",
#     "inversion",
#     "inversion_sample"
# ][7]

# initial_model_name = [None, "gpt2", "gpt2-large", "llama2"][0]
# initial_model_based_on = ["gens", "max_score"][1]
# initial_model_param = [[10, 50, 100], [0.01, 0.1]][1][1]

# # Model params
# main_model_name = ["gpt2", "llama2"][0]
# if initial_model_name is None:
#     set_model_params(main_model_name)
#     using_main_model = True
# else:
#     set_model_params(initial_model_name)
#     using_main_model = False

# # Define population
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)
# toolbox = base.Toolbox()
# toolbox.register("attr_bool", random.randint, 0, vocab_size-1)
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool)

# # Define algorithm operations
# toolbox.register("mate", tools.cxUniform, indpb=0.3)
# toolbox.register("mutate", custom_mutate, indpb=0.1, bounded=False)
# toolbox.register("select", tools.selTournament, tournsize=15)

# # Set up recorded statistics
# stats = tools.Statistics(key=lambda ind: ind.fitness.values)
# stats.register("max", numpy.max)
# stats.register("min", numpy.min)
# stats.register("avg", numpy.mean)
# stats.register("std", numpy.std)


# def normal_ga(start_arg, end_arg):
#     for x in range(start_arg, end_arg):
#         experiment_start_time = time.process_time()#datetime.now()

#         print("\n\n"+"-"*50+"\n"+"-"*50+"\nOriginal input:", [user_text(x)])
#         print("Target output:", [assistant_text(x)])
#         print("\nCurrent target output:", [assistant_text(x)])
#         print("gen  nevals  time            max         min avg         std        ")

#         # Tokenize target sequence
#         target_seq = prefix + assistant_text(x)
#         target_tokens = tokenizer(target_seq, return_tensors='pt').input_ids[0]
#         target_tokens = target_tokens[remove_prefix:]

#         # Eval function and hall of fame need to be reset between runs
#         toolbox.register("evaluate", eval_inds, in_batch_size=in_batch_size, target_tokens=target_tokens, prompt_template=prompt_template)
#         halloffame = tools.HallOfFame(maxsize=10)

#         # Initalise population
#         pop = initialise_text_pop(start_population_as, x, max_input_len, num_inputs)

#         # Run algorithm (and print proggress stats)
#         pop, logbook, out_of_time = eaSimple(
#             pop,
#             toolbox,
#             cxpb=0.5,
#             mutpb=0.2,
#             ngen=max_gens,
#             halloffame=halloffame,
#             stats=stats,
#             verbose=True,
#             experiment_start_time=experiment_start_time,
#             max_minutes=max_minutes,
#             stop_early=True,
#         )

#         if out_of_time:
#             continue

#>>>>>>>>>>>>>>>>>>>>>>> Main Loop proggressive
# Run/ search params
max_input_len = 64  # Doesnt matter since bounded=False
num_inputs = 1000#1000
max_gens = 500#200 #999 # unbounded
max_minutes = None#10
stop_early = True

# Initialisation params
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
][7]

initial_model_name = [None, "gpt2", "gpt2-large", "llama2"][0]
initial_model_based_on = ["gens", "max_score"][1]
initial_model_param = [[50, 100], [0.01, 0.1]][0][0]

# Model params
main_model_name = ["gpt2", "llama2"][0]
if initial_model_name is None:
    set_model_params(main_model_name)
    using_main_model = True
else:
    set_model_params(initial_model_name)
    using_main_model = False

# Proggressive params
progressive_based_on = ["gens", "non_zero_scores"][0]
progressive_param = 200

# Define population
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, vocab_size-1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool)

# Define algorithm operations
toolbox.register("mate", tools.cxUniform, indpb=0.3)
toolbox.register("mutate", custom_mutate, indpb=0.1, bounded=False)
toolbox.register("select", tools.selTournament, tournsize=15)

# Set up recorded statistics
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", numpy.max)
stats.register("min", numpy.min)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)


def proggressive_gens_ga(start_arg, end_arg):
    for x in range(start_arg, end_arg):
        experiment_start_time = time.process_time()#datetime.now()

        print("\n\n"+"-"*50+"\n"+"-"*50+"\nOriginal input:", [user_text(x)])
        print("Target output:", [assistant_text(x)])

        # Tokenize target sequence
        target_seq = prefix + assistant_text(x)
        target_tokens = tokenizer(target_seq, return_tensors='pt').input_ids[0]
        target_tokens = target_tokens[remove_prefix:]

        # Things reset between examples from dataset
        halloffame = tools.HallOfFame(maxsize=10)

        # Initalise population
        pop = initialise_text_pop(start_population_as, x, max_input_len, num_inputs)

        # Decide on progressive intervals over the total number of generations
        gen_slices = [progressive_param // len(target_tokens) for _ in range(len(target_tokens))]
        # Spread the leftover generations evenly starting from final slice and going backwards until run out (earlier slices likely won't get any extra)
        for slice_i in range(len(gen_slices)-1, (len(gen_slices)-1-(progressive_param % len(target_tokens))), -1):
            gen_slices[slice_i] += 1
        assert sum(gen_slices) == progressive_param

        for gen_i in range(len(gen_slices)):
            # gen_slice is the amount of generations to currently do
            gen_slice = gen_slices[gen_i]
            if gen_slice == 0:
                continue
            gens_so_far = sum(gen_slices[:gen_i])

            # Reveal another token of the target output for each gen_slice completed
            run_target_tokens = target_tokens[:gen_i+1]
            print("\nCurrent target output:", [tokenizer.decode(run_target_tokens)])
            print("gen  nevals  time            max         min avg         std        ")

            if gens_so_far != 0:
                # All fitness functions need to be updated based on the new target output (if theyre non zero)
                # Smart method of having the probabilities be updated only by one extra target token
                invalid_ind = [ind for ind in pop if (not ind.fitness.valid) or (ind.fitness.values[0] != 0)]
                fitnesses = eval_inds(ind_list=invalid_ind, target_tok_id=gen_i, in_batch_size=in_batch_size, target_tokens=run_target_tokens, prompt_template=prompt_template)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Update the stats based on this partial update
                record = stats.compile(pop) if stats else {}
                logbook = tools.Logbook()
                logbook.header = ['gen', 'nevals', 'time'] + (stats.fields if stats else [])
                gen_addition = 0.5
                logbook.record(gen=(gens_so_far+gen_addition), nevals=len(invalid_ind), time=timedelta(seconds=(time.process_time()-experiment_start_time)), **record)
                print(logbook.__str__().split("\n")[-1])
                # Print best ones
                best = tools.selBest(pop, k=1000)
                best_fitnesses = [ind.fitness.values[0] for ind in best]
                print([[best_fitnesses[i], [tokenizer.decode(best[i]).replace("'", "\\'").replace('"', '\\"')]] for i in range(len(best))])

            # Run genetic algo with new eval function
            toolbox.register("evaluate", eval_inds, in_batch_size=in_batch_size, target_tokens=run_target_tokens, prompt_template=prompt_template)
            breaking = False
            for g in range(gen_slice):
                pop, logbook, out_of_time = eaSimple(
                    pop,
                    toolbox,
                    cxpb=0.5,
                    mutpb=0.2,
                    ngen=1,
                    halloffame=halloffame,
                    stats=stats,
                    verbose=True,
                    gens_so_far=gens_so_far,
                    experiment_start_time=experiment_start_time,
                    max_minutes=max_minutes,
                    stop_early=False,
                )
                gens_so_far += 1
                if (gens_so_far >= max_gens) or out_of_time:
                    breaking = True
                    break
            if breaking:
                break

        # If still havent reached max gens/ time then continue searching with full target output
        for g in range(gens_so_far,max_gens):
            if out_of_time:
                break

            # Run genetic algo with new eval function
            # toolbox.register("evaluate", eval_inds, in_batch_size=in_batch_size, target_tokens=target_tokens, prompt_template=prompt_template)
            pop, logbook, out_of_time = eaSimple(
                pop,
                toolbox,
                cxpb=0.5,
                mutpb=0.2,
                ngen=1,
                halloffame=halloffame,
                stats=stats, verbose=True,
                gens_so_far=g,
                experiment_start_time=experiment_start_time,
                max_minutes=max_minutes,
                stop_early=False,
            )



proggressive_gens_ga(0,50)

# del autoencoder
del model
torch.cuda.empty_cache()
