
# Run/ search params
max_input_len = 15 # Doesnt matter since bounded=False
num_inputs = 100#1000
max_gens = 50#200 #999 # unbounded
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
][0]

initial_model_name = [None, "gpt2", "gpt2-large", "llama2"][2]
initial_model_based_on = ["gens", "max_score"][1]
initial_model_param = [[10, 50, 100], [0.01, 0.1]][1][1]

# Model params
main_model_name = ["gpt2", "llama2"][0]
if initial_model_name is None:
    set_model_params(main_model_name)
    using_main_model = True
else:
    set_model_params(initial_model_name)
    using_main_model = False

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


def normal_ga(start_arg, end_arg):
    for x in range(start_arg, end_arg):
        experiment_start_time = time.process_time()#datetime.now()

        print("\n\n"+"-"*50+"\n"+"-"*50+"\nOriginal input:", [user_text(x)])
        print("Target output:", [assistant_text(x)])
        print("\nCurrent target output:", [assistant_text(x)])
        print("gen	nevals	time          	max        	min	avg        	std        ")

        # Tokenize target sequence
        target_seq = prefix + assistant_text(x)
        target_tokens = tokenizer(target_seq, return_tensors='pt').input_ids[0]
        target_tokens = target_tokens[remove_prefix:]

        # Eval function and hall of fame need to be reset between runs
        toolbox.register("evaluate", eval_inds, in_batch_size=in_batch_size, target_tokens=target_tokens, prompt_template=prompt_template)
        halloffame = tools.HallOfFame(maxsize=10)

        # Initalise population
        pop = initialise_text_pop(start_population_as, x, max_input_len, num_inputs)

        # Run algorithm (and print proggress stats)
        pop, logbook, out_of_time = eaSimple(
            pop,
            toolbox,
            cxpb=0.5,
            mutpb=0.2,
            ngen=max_gens,
            halloffame=halloffame,
            stats=stats,
            verbose=True,
            experiment_start_time=experiment_start_time,
            max_minutes=max_minutes,
            stop_early=True,
        )

        if out_of_time:
            continue



# Run/ search params
max_input_len = 15  # Doesnt matter since bounded=False
num_inputs = 100#1000
max_gens = 40#200 #999 # unbounded
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
][2]

# initial_model_name = [None, "gpt2", "gpt2-large", "llama2"][1]
# initial_model_based_on = ["gens", "max_score"][1]
# initial_model_param = [[50, 100], [0.01, 0.1]][0][0]

# # Model params
# main_model_name = ["gpt2", "llama2"][0]
# if initial_model_name is None:
#     set_model_params(main_model_name)
#     using_main_model = True
# else:
#     set_model_params(initial_model_name)
#     using_main_model = False

# # Proggressive params
# progressive_based_on = ["gens", "non_zero_scores"][0]
# progressive_param = 1#200

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


def proggressive_nzs_ga(start_arg, end_arg):
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

        # Initialise current target output
        target_tok_id = 0
        target_updated = False
        run_target_tokens = target_tokens[:target_tok_id+1]
        print("\nCurrent target output:", [tokenizer.decode(run_target_tokens)])
        print("gen	nevals	time          	max        	min	avg        	std        ")
        toolbox.register("evaluate", eval_inds, in_batch_size=in_batch_size, target_tokens=run_target_tokens, prompt_template=prompt_template)

        # Run search until out of resources
        gens_so_far = 0
        out_of_time = False
        while not ((gens_so_far >= max_gens) or out_of_time):
            if target_updated:
                # Update current target output
                run_target_tokens = target_tokens[:target_tok_id+1]
                print("\nCurrent target output:", [tokenizer.decode(run_target_tokens)])
                print("gen	nevals	time          	max        	min	avg        	std        ")
                toolbox.register("evaluate", eval_inds, in_batch_size=in_batch_size, target_tokens=run_target_tokens, prompt_template=prompt_template)

                # All fitness functions need to be updated based on the new target output (if theyre non zero)
                # Smart method of having the probabilities be updated only by one extra target token
                invalid_ind = [ind for ind in pop if (not ind.fitness.valid) or (ind.fitness.values[0] != 0)]
                fitnesses = eval_inds(ind_list=invalid_ind, target_tok_id=target_tok_id, in_batch_size=in_batch_size, target_tokens=run_target_tokens, prompt_template=prompt_template)
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

                target_updated = False

            # Run genetic algo for one step
            pop, logbook, out_of_time = eaSimple(
                pop,
                toolbox,
                cxpb=0.5,
                mutpb=0.2,
                ngen=1,
                halloffame=halloffame,
                stats=stats, verbose=True,
                gens_so_far=gens_so_far,
                experiment_start_time=experiment_start_time,
                max_minutes=max_minutes,
                stop_early=True,
            )
            gens_so_far += 1

            # Check if should start considering more tokens, if percentage of non zero values in population are high enough
            if (gens_so_far < max_gens) and (len(run_target_tokens) != len(target_tokens)):
                non_zeros = [1 for ind in pop if ind.fitness.values[0] != 0]
                non_zeros_percent = ((sum(non_zeros)/num_inputs)*100)
                if non_zeros_percent >= progressive_param:
                    target_tok_id += len(target_tokens) // 10 # Start considering next 10% of tokens
                    target_updated = True


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
            print("gen	nevals	time          	max        	min	avg        	std        ")

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
                    stop_early=True,
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
                stop_early=True,
            )

