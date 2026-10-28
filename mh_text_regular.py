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
import time
from datetime import timedelta
import gzip
import json


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
def hastings_factor(x, x2, vocab_size, weights=(1, 1, 1, 1)):
    # Ratio p(x2) q(x | x2) / (p(x) q(x2 | x)) for the proposal in mutate_sample and the prior over inputs, which
    # the acceptance probability is multiplied by. The prior makes every length from 1 to max_len equally
    # likely and every token uniform, so p(x) is proportional to vocab_size^-n; inputs beyond max_len are handled by the
    # caller. A position is drawn uniformly from the n tokens, then the mutation type with the given weights (modify,
    # insert, delete, swap), except that delete and swap are unavailable at length 1. Modify and swap are their own
    # reverse with the same probability and keep the length, so the factor is 1. An insertion goes before one of
    # the n tokens and draws its token uniformly (a factor vocab_size), which the prior's vocab_size^-1 cancels;
    # the reverse deletion picks one of the n+1 tokens, so the factor also counts how many positions give the same
    # string either way, and the ratio of the delete and insert weights. A deletion whose reverse insertion is
    # impossible (e.g. deleting the last token) gets 0.
    n, m = len(x), len(x2)
    if n == m:
        return 1.0
    w_mod, w_ins, w_del, w_swap = weights
    z = lambda k: float(w_mod + w_ins) if k == 1 else float(sum(weights))  # sum of the mutation-type weights at length k
    longer, shorter = (x2, x) if m > n else (x, x2)
    deletions = [j for j in range(len(longer)) if longer[:j] + longer[j+1:] == shorter]
    insertions = [j for j in deletions if j <= len(shorter) - 1]
    if m > n:  # insertion x -> x2
        return w_del / w_ins * z(n) / z(m) * n / m * len(deletions) / len(insertions)
    return w_ins / w_del * z(n) / z(m) * n / m * len(insertions) / len(deletions)  # deletion x -> x2


def prior_move(individual):
    # One edit from mutate_sample, with the factor that the acceptance probability is multiplied by under the
    # chosen prior (hastings_factor assumes uniform_lengths, whose prior has an extra vocab_size^-n)
    proposal, factor = mutate_sample(individual, max_input_len, False, vocab_size)
    if len(proposal) > max_len:
        return proposal, 0.0
    if prior == "uniform_strings":
        factor *= float(vocab_size) ** (len(proposal) - len(individual))
    return proposal, factor


def propose(individual):
    # A proposal of edits_per_step edits. With one edit, its factor goes into the acceptance probability as usual.
    # With several, each edit is first accepted or rejected under the prior alone, which makes every edit, and so
    # the whole sequence, reversible with respect to the prior; the sequence's factor is then exactly 1 and the
    # acceptance probability only compares the output probabilities (two-stage Metropolis-Hastings)
    if edits_per_step == 1:
        return prior_move(individual)
    for _ in range(edits_per_step):
        proposal, factor = prior_move(individual)
        if factor >= 1 or random.random() < factor:
            individual = proposal
    return individual, 1.0


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
    # The edits are weighted by edit_weights; hastings_factor accounts for insertion drawing one of vocab_size tokens
    types_p = list(edit_weights)
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

    return individual, hastings_factor(original, individual, vocab_size, edit_weights)


#>>>>>>>>>>>>>>>>>>>>>>> Initialisation Function
# Generators for the non-random initialisations, as in Notebook.ipynb
def fluent_model_generate(seed=None, num_generate=1, batch_size=100, max_new_tokens=20):
    if seed is None:
        prefix_length = 2
    else:
        if seed[-1] == " ":
            seed = seed[:-1]
        if seed[-1] not in [".", "!", "?"]:
            seed += "."
        prefix_length = len(tokenizer(seed).input_ids)

    all_sequences = []
    for i in range(0, num_generate, batch_size):
        batch_s = min(batch_size, num_generate-i)
        if seed is None:
            prompt_batch = [tokenizer.decode(random.randint(0,vocab_size))+". " for _ in range(batch_s)]
        else:
            prompt_batch = [seed]*batch_s

        input_ids = tokenizer(prompt_batch, return_tensors="pt", padding=True).to(model.device)
        generate_output = model.generate(**input_ids, temperature=1, do_sample=True, top_p=0.95, top_k=100, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, pad_token_id=eos_token)
        sequences = [seq[prefix_length:] for seq in generate_output.sequences]
        all_sequences += sequences

    random_tokens = lambda : tokenizer.decode([random.randint(0, vocab_size) for _ in range(random.randint(1, max_new_tokens))])

    decoded_sequences = tokenizer.batch_decode(all_sequences, skip_special_tokens=True)
    decoded_sequences = [random_tokens() if len(seq) == 0 else seq for seq in decoded_sequences]
    decoded_sequences = [seq[1:] if seq[0] in [" ", "\xa0", "\n"] else seq for seq in decoded_sequences]
    decoded_sequences = [seq[1:] if seq[0] in [" ", "\xa0", "\n"] else seq for seq in decoded_sequences]
    return decoded_sequences


def inversion_model_generate(target_output, num_inversions=1, batch_size=100, max_new_tokens=20, sample=False):
    from transformers import AutoModelForSeq2SeqLM
    inv_tokenizer = AutoTokenizer.from_pretrained("wentingzhao/inversion_seq2seq_baseline")
    inv_tokenizer.padding_side = "left"
    inv_tokenizer.pad_token = inv_tokenizer.eos_token
    inv_eos_token = inv_tokenizer(inv_tokenizer.eos_token).input_ids[0]
    inv_model = AutoModelForSeq2SeqLM.from_pretrained("wentingzhao/inversion_seq2seq_baseline").eval().to("cuda")

    outputs = []
    for i in range(0, num_inversions, batch_size):
        batch_s = min(batch_size, num_inversions-i)
        input_ids = inv_tokenizer([target_output]*batch_s, return_tensors="pt", padding=True).input_ids.to("cuda")
        if sample:
            output = inv_model.generate(inputs=input_ids, temperature=1, do_sample=True, top_p=0.99, top_k=500, max_new_tokens=max_new_tokens, pad_token_id=inv_eos_token)
        else:
            output = inv_model.generate(inputs=input_ids, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=inv_eos_token)
        outputs += list(output)

    del inv_model
    torch.cuda.empty_cache()
    return [inv_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


def initialise_chains(start_population_as, sample_x, max_input_len, num_inputs):
    # Starting inputs for the chains, cut to max_len so that they lie inside the prior's support
    if start_population_as == "random":
        pop = [[random.randint(0,vocab_size-1) for _ in range(random.randint(1,max_input_len))] for _ in range(num_inputs)]
    elif start_population_as == "random_fluent":
        texts = fluent_model_generate(num_generate=num_inputs, max_new_tokens=max_input_len)
        pop = [tokenizer(t).input_ids for t in texts]
    elif start_population_as == "output":
        target_seq = prefix + assistant_text(sample_x)
        output = tokenizer(target_seq, return_tensors='pt').input_ids[0][remove_prefix:].tolist()
        pop = [list(output) for _ in range(num_inputs)]
    elif start_population_as == "inversion":
        inversion = tokenizer(inversion_model_generate(assistant_text(sample_x), num_inversions=1, max_new_tokens=max_input_len, sample=False)[0]).input_ids
        pop = [list(inversion) for _ in range(num_inputs)]
    elif start_population_as == "inversion_sample":
        texts = inversion_model_generate(assistant_text(sample_x), num_inversions=num_inputs, max_new_tokens=max_input_len, sample=True)
        pop = [tokenizer(t).input_ids for t in texts]
    else:
        raise ValueError(start_population_as)
    # an empty input has no tokens to edit, so it is replaced by a single random token
    return [ind[:max_len] if len(ind) > 0 else [random.randint(0,vocab_size-1)] for ind in pop]


def save_states(sample_x, step, pop, pop_eval, full_target=True):
    # Append the chains' current inputs (token IDs) and their output probabilities to states_file as one JSON line,
    # for distribution analyses. full_target is False while a progressive run scores a prefix of the target, and
    # pop_eval is None before the chains have been scored
    if not save_states_every:
        return
    with gzip.open(states_file, "at", encoding="utf-8") as f:
        f.write(json.dumps({"pair": sample_x, "input": user_text(sample_x), "step": step, "full_target": full_target,
                            "states": [list(map(int, ind)) for ind in pop],
                            "probs": None if pop_eval is None else [float(p) for p in pop_eval]}) + "\n")


resumed = None


def resume_chains(sample_x):
    # The chains' last full-target snapshot for this pair in the state files of an earlier run (resume_from, in
    # pair order), so that the run continues them: their inputs, output probabilities and step number
    global resumed
    if resumed is None:
        resumed, offset = {}, 0
        for path in resume_from:
            last = 0
            for line in gzip.open(path, "rt", encoding="utf-8"):
                r = json.loads(line)
                last = max(last, r["pair"] + 1)
                if r["probs"] is None or not r["full_target"]:
                    continue
                key = offset + r["pair"]
                if key not in resumed or r["step"] > resumed[key]["step"]:
                    resumed[key] = r
            offset += last
    r = resumed[sample_x]
    assert r["input"] == user_text(sample_x), (sample_x, r["input"])
    return [list(s) for s in r["states"]], list(r["probs"]), r["step"]


#>>>>>>>>>>>>>>>>>>>>>>> Main Loop
# Parameters which are manually defined
max_input_len = 15 # Longest random initialisation, as for GA
max_len = 64 # Longest input allowed by the prior (PSO's decoding cap in the paper, well above any original input)
# Prior over inputs: every length 1..max_len equally likely with uniform tokens, or every string up to
# max_len equally likely (most of its weight is then on the longest inputs)
prior = ["uniform_lengths", "uniform_strings"][0]
# How the chains start: random tokens, text generated by the target model, the target output, or the inversion
# model's greedy output or samples
start_population_as = ["random", "random_fluent", "output", "inversion", "inversion_sample"][0]
# Proposal: weights of the four edits (modify, insert, delete, swap) and how many edits make up one proposal
edit_weights = [1, 1, 1, 1]
edits_per_step = 1
# Tempering: the output probability enters the acceptance raised to this power (1 samples the posterior itself)
likelihood_power = 1.0
# Saving the chains for distribution analyses: every save_states_every steps, and at the start and the last
# step, to states_file as gzipped JSON lines (0 saves nothing)
save_states_every = 0
states_file = "chain_states.jsonl.gz"
# Continuing an earlier run: its state files in pair order; each pair's chains restart from their last saved state
# (instead of start_population_as), and new snapshots continue its step numbers. [] starts afresh
resume_from = []
num_inputs = 600
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
        print("gen\tnevals\ttime          \tmax        \tmin\tavg        \tstd        ")
        experiment_start_time = time.process_time()

        # Tokenize target sequence
        target_seq = prefix + assistant_text(x)
        target_tokens = tokenizer(target_seq, return_tensors='pt').input_ids[0]
        target_tokens = target_tokens[remove_prefix:]

        # Initialise 'population'
        if resume_from:  # or continue the chains of an earlier run
            pop, pop_eval, first_step = resume_chains(x)
        else:
            pop = initialise_chains(start_population_as, x, max_input_len, num_inputs)
            pop_eval = eval_inds(pop, in_batch_size, target_tokens, prompt_template, 0)
            first_step = 0
            save_states(x, 0, pop, pop_eval)

        best_max = max(pop_eval)
        best_input = tokenizer.decode(pop[numpy.argmax(pop_eval)])
        for gen in range(total_gens):
            # Get new candidates (edits_per_step mutations each)
            proposals = [propose(individual) for individual in pop]
            pop2 = [proposal for proposal, _ in proposals]
            hastings = [factor for _, factor in proposals]
            # Work out acceptence based on evaluations
            pop_eval2 = eval_inds(pop2, in_batch_size, target_tokens, prompt_template, 0)
            # Change pop based on whether accepted
            for ind_i in range(len(pop)):
                # Metropolis-Hastings acceptance under the length prior: proposals longer than max_len are
                # rejected, and a chain whose current input has zero probability follows the prior alone. The
                # output probabilities are raised to likelihood_power (tempering; 1 samples the posterior itself)
                if len(pop2[ind_i]) > max_len:
                    acceptence = 0
                elif pop_eval[ind_i] != 0:
                    acceptence = min(1, hastings[ind_i] * (pop_eval2[ind_i] / pop_eval[ind_i]) ** likelihood_power)
                else:
                    acceptence = min(1, hastings[ind_i])
                if acceptence > 0 and (acceptence >= 1 or random.random() < acceptence):
                    pop[ind_i] = pop2[ind_i]
                    pop_eval[ind_i] = pop_eval2[ind_i]

            # Print best inputs of this gen and so far
            curr_max = max(pop_eval)
            if curr_max > best_max:
                best_max = curr_max
                best_input = tokenizer.decode(pop[numpy.argmax(pop_eval)])
            # Log in the GA format: step line (CPU time since the pair started, as GA), then the best input so far
            elapsed = timedelta(seconds=(time.process_time()-experiment_start_time))
            print("%d  \t%d   \t%s\t%g  \t%g  \t%g  \t%g" % (gen+1, len(pop2), elapsed, max(pop_eval), min(pop_eval), numpy.mean(pop_eval), numpy.std(pop_eval)))
            print([[best_max, [str(best_input).replace("'", "\\'").replace('"', '\\"')]]])
            if save_states_every and ((gen+1) % save_states_every == 0 or gen+1 == total_gens):
                save_states(x, first_step + gen+1, pop, pop_eval)


normal_sample(0, 30)
