import gc

########################################################################
######################## LOAD GENERATIVE MODEL
# GPT 2
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda").eval()

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
eos_token = tokenizer(tokenizer.eos_token).input_ids[0]
# tokenizer.pad_token = tokenizer.bos_token
model.config.pad_token_id = model.config.eos_token_id
prompt_template = lambda x: f"User: {x}\nAssistant:"


########################################################################
######################## LOAD DATASETS
# Chatbot Arena
from datasets import load_dataset, load_from_disk

reloaded_data = load_from_disk("processed_dataset")
processed_data = reloaded_data[:30]
user_text = lambda x: processed_data["conversation_a" if processed_data["winner"][x] == "model_a" else "conversation_b"][x][0]['content']
assistant_text = lambda x: processed_data["conversation_a" if processed_data["winner"][x] == "model_a" else "conversation_b"][x][1]['content']


########################################################################
######################## DEFINE ALGORITHMS

######################## Text-Based Metropolis-Hastings Sampling
import random
import math
import numpy
import copy
from numpy.random import choice


#>>>>>>>>>>>>>>>>>>>>>>> Evaluation Function
def eval_inds(ind_list, in_batch_size, target_tokens, prompt_template=None, target_tok_id=0, ind_evals=None):
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

    return string_probabilities


#>>>>>>>>>>>>>>>>>>>>>>> Mutate Function
def hastings_factor(x, x2, vocab_size):
    # Ratio q(x | x2) / q(x2 | x) of the proposal in mutate_sample (unbounded length), which the acceptance
    # probability is multiplied by. A position is drawn uniformly from the n tokens, then the mutation type with
    # weights [1, 1, 1/vocab_size, 1] (modify, insert, delete, swap), except that delete and swap are unavailable
    # at length 1. Modify and swap are their own reverse with the same probability, so the factor is 1. An
    # insertion goes before one of the n tokens and draws its token uniformly; the reverse deletion picks one of
    # the n+1 tokens, so the factor also counts how many positions give the same string either way. A deletion
    # whose reverse insertion is impossible (e.g. deleting the last token) gets a factor of 0.
    n, m = len(x), len(x2)
    if n == m:
        return 1.0
    z = lambda k: 2.0 if k == 1 else 3.0 + 1.0 / vocab_size  # sum of the mutation-type weights at length k
    longer, shorter = (x2, x) if m > n else (x, x2)
    deletions = [j for j in range(len(longer)) if longer[:j] + longer[j+1:] == shorter]
    insertions = [j for j in deletions if j <= len(shorter) - 1]
    if m > n:  # insertion x -> x2
        return z(n) / z(m) * n / m * len(deletions) / len(insertions)
    return z(n) / z(m) * n / m * len(insertions) / len(deletions)  # deletion x -> x2


def mutate_sample(individual, max_input_len, bounded, vocab_size):
    assert not bounded  # hastings_factor assumes unbounded length
    # Propose on a copy so that a rejected proposal leaves the current state unchanged
    original = list(individual)
    individual = list(individual)

    # Keep track of elements index offset since inserts and deletes move it around
    # offset = 0
    # for i in range(len(individual)):
        # Each element has independent probability of recieving mutation
        # if random.random() >= indpb:
            # continue

    i = random.randint(0,len(individual)-1)

    # Work out probabilities of chosing mutation type
    types = [0,1,2,3] # [modify, insert, delete, swap]
    # Deletion is discounted to match insertion
    types_p = [1,1,1/vocab_size,1]
    # Probabilities also depend on the length of the individual
    if len(individual) == 0:
        types_p[0] = 0 # modify
        types_p[2] = 0 # delete
        types_p[3] = 0 # swap
    elif len(individual) == 1:
        types_p[2] = 0 # delete
        types_p[3] = 0 # swap
    elif bounded and len(individual) == max_input_len:
        types_p[1] = 0 # insert
    # Probabilities need to be normalised
    types_sum = sum(types_p)
    types_p = [p/types_sum for p in types_p]

    # Apply just one of the available mutations to this element
    mut = choice(types, 1, p=types_p)
    # ind = i + offset
    ind = i
    if mut == 0: # modify
        individual[ind] = random.randint(0, vocab_size-1)
    elif mut == 1: # insert
        individual.insert(ind, random.randint(0, vocab_size-1))
        # offset += 1
    elif mut == 2: # delete
        del individual[ind]
        # offset -= 1
    elif mut == 3: # swap
        ind2 = (ind + random.randint(1,len(individual)-1)) % len(individual)
        temp = individual[ind]
        individual[ind] = individual[ind2]
        individual[ind2] = temp

    return individual, hastings_factor(original, individual, vocab_size)


#>>>>>>>>>>>>>>>>>>>>>>> Main Loop
# Parameters which are manually defined
max_input_len = 15 # Doesnt matter since bounded=False for mutation function
num_inputs = 1000
total_gens = 200
model_name = "gpt2"

# Parameters which are model specific
if model_name == "gpt2":
    in_batch_size = 400
    prefix = " "
    remove_prefix = 0
    vocab_size = 50257
    prompt_template = lambda x: f"User: {x}\nAssistant:"
if model_name == "llama2":
    in_batch_size = 200
    prefix = "\n\n"
    remove_prefix = 4
    vocab_size = 32000
    prompt_template = lambda x: f"[INST]{x}[/INST]\n\n"

def normal_sample(start_arg, end_arg):
    for x in range(start_arg, end_arg):
        print("-"*30)
        print("\n\nOriginal input:", [user_text(x)])
        print("Target output:", [assistant_text(x)])

        # Tokenize target sequence
        target_seq = prefix + assistant_text(x)
        target_tokens = tokenizer(target_seq, return_tensors='pt').input_ids[0]
        target_tokens = target_tokens[remove_prefix:]

        # Initialise 'population'
        pop = [[random.randint(0,vocab_size-1) for _ in range(random.randint(1,max_input_len))] for _ in range(num_inputs)]
        pop_eval = eval_inds(pop, in_batch_size, target_tokens, prompt_template, 0)

        best_max = max(pop_eval)
        best_input = tokenizer.decode(pop[numpy.argmax(pop_eval)])
        for gen in range(total_gens):
            # Get new candidates (based on a single mutation)
            proposals = [mutate_sample(individual, max_input_len, False, vocab_size) for individual in pop]
            pop2 = [proposal for proposal, _ in proposals]
            hastings = [factor for _, factor in proposals]
            # Work out acceptence based on evaluations
            pop_eval2 = eval_inds(pop2, in_batch_size, target_tokens, prompt_template, 0)
            # Change pop based on whether accepted
            for ind_i in range(len(pop)):
                # Metropolis-Hastings acceptance; a chain whose current input has zero probability accepts any proposal
                acceptence = min(1, hastings[ind_i] * pop_eval2[ind_i] / pop_eval[ind_i]) if pop_eval[ind_i] != 0 else 1
                if acceptence != 0 and (acceptence == 1 or random.random() < acceptence):
                    pop[ind_i] = pop2[ind_i]
                    pop_eval[ind_i] = pop_eval2[ind_i]

            # Print best inputs of this gen and so far
            curr_max = max(pop_eval)
            curr_input = tokenizer.decode(pop[numpy.argmax(pop_eval)])
            print(gen, "\t", curr_max, "\t", curr_input)
            if curr_max > best_max:
                best_max = curr_max
                best_input = curr_input
            print("max", "\t", best_max, "\t", best_input)


normal_sample(0, 30)
