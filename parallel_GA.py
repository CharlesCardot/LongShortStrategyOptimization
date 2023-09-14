import pygad
import pickle
import time

import numpy as np
import pandas as pd

import random

pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt


with open('market_by_days.pkl', 'rb') as f:
    market_by_days = pickle.load(f)

debug = False
def my_algorithm(random_seed, day_num, starting_balance, x_percent_loss, y_percent_gain, be_threshold, tslp):
    """
    Simulates returns using the algorithm described in the markdown above
    and returns a list of daily balances along with various statistics about
    the returns and drawdowns.
    """

    random.seed(random_seed)

    balance = [starting_balance]

    for df in random.sample(market_by_days, day_num):
        open_price = df.iloc[0]['average']  # Open position at the first minute's average price

        betting_value = balance[-1] / 2

        balance.append(0) # Betting everything, so current balance for the day is zero

        long_position = {'open_value': betting_value, 'current_value': betting_value}
        short_position = {'open_value': betting_value, 'current_value': betting_value}

        sl_value = betting_value * (1 + x_percent_loss) # stop-loss value
        tp_value = betting_value * (1 + y_percent_gain) # take-profit value
        be_value = betting_value * (1 + be_threshold)
        final_bet_value = 0
        trailing_sl_triggered = False
        be_value_triggered = False

        debug_msg_arr = []
        prices = df["average"].tolist()
        for current_price in prices:

            if not long_position and not short_position:
                break # break loop once all positions are closed out


            if long_position:
                debug_msg_arr.append(("long_position", round(long_position['current_value'], 5), round(long_position['current_value'] / long_position['open_value'], 5)))
                long_position['current_value'] = (1 + (current_price - open_price) / open_price) * long_position['open_value']
            if short_position:
                debug_msg_arr.append(("short_position", round(short_position['current_value'], 5), round(short_position['current_value'] / short_position['open_value'], 5)))
                short_position['current_value'] = (1 + -1 * (current_price - open_price) / open_price) * short_position['open_value']


            # Check for stop loss condition reached on long position
            if long_position and long_position['current_value'] <= sl_value:
                debug_msg_arr.append(("Stop loss on long_position reached with long value ratio at ", round(long_position['current_value'] / long_position['open_value'], 5)))
                final_bet_value += sl_value # Assume perfect order filling, no slippage
                long_position = None
            # Check for stop loss condition reached on short position
            elif short_position and short_position['current_value'] <= sl_value:
                debug_msg_arr.append(("Stop loss on short_position reached with short value ratio at ", round(short_position['current_value'] / short_position['open_value'], 5)))
                final_bet_value += sl_value # Assume perfect order filling, no slippage
                short_position = None


            # Trailing stop loss management
            if not trailing_sl_triggered:
                # Check for trailing stop condition reached on long position
                if long_position and long_position['current_value'] >= tp_value:
                    debug_msg_arr.append(("Trigging trailing stop loss on long at ", round(long_position['current_value'] / long_position['open_value'], 5)))
                    sl_value = long_position['current_value'] * (1 - tslp)
                    debug_msg_arr.append(("Setting sl_value to ", round(sl_value, 5), "relative to ", round(long_position['current_value'], 5)))
                    trailing_sl_triggered = True
                # Check for trailing stop condition reached on short position
                elif short_position and short_position['current_value'] >= tp_value:
                    debug_msg_arr.append(("Trigging trailing stop loss on short at ", round(short_position['current_value'] / short_position['open_value'], 5)))
                    sl_value = short_position['current_value'] * (1 - tslp)
                    debug_msg_arr.append(("Setting sl_value to ", round(sl_value, 5), "relative to ", round(short_position['current_value'], 5)))
                    trailing_sl_triggered = True
            else:
                if long_position and sl_value < long_position['current_value'] * (1 - tslp):
                    sl_value = long_position['current_value'] * (1 - tslp)
                    debug_msg_arr.append(("Updating trailing stop loss to", round(sl_value, 5), "with long at", round(long_position['current_value'], 5)))
                elif short_position and sl_value < short_position['current_value'] * (1 - tslp):
                    sl_value = short_position['current_value'] * (1 - tslp)
                    debug_msg_arr.append(("Updating trailing stop loss to", round(sl_value, 5), "with short at", round(short_position['current_value'], 5)))


            # Condition for moving sl_value to breakeven after one of the positions has stopped out
            if be_value_triggered == False:
                if long_position is not None and short_position is None and long_position['current_value'] >= be_value:
                    debug_msg_arr.append(("be_value on long_position reached with short at ", round(long_position['current_value'] / long_position['open_value'], 5)))
                    sl_value = betting_value * (1 + -1 * x_percent_loss)
                    be_value_triggered = True
                if short_position is not None and long_position is None and short_position['current_value'] >= be_value:
                    debug_msg_arr.append(("be_value on short_position reached with short at ", round(short_position['current_value'] / short_position['open_value'], 5)))
                    sl_value = betting_value * (1 + -1 * x_percent_loss)
                    be_value_triggered = True


        # Close any remaining positions at the end of the day
        if long_position:
            debug_msg_arr.append(("Closing long position at ", long_position['current_value']))
            final_bet_value += long_position['current_value']
        if short_position:
            debug_msg_arr.append(("Closing short position at ", short_position['current_value']))
            final_bet_value += short_position['current_value']
        
        balance[-1] += final_bet_value

        if debug:
            for msg in debug_msg_arr:
                print(msg)
            print()


    balance = np.asarray(balance)

    # Calculate average daily return
    daily_returns = np.diff(balance) / balance[:-1]
    average_daily_return = np.mean(daily_returns) * 100

    # Calculate drawdowns
    peak = balance[0]
    drawdowns = [[], [], []]  # [[start_day], [stop_day], [drawdown_value]]
    in_drawdown = False
    current_drawdown_value = 0
    max_drawdown_value = 0

    for i, b in enumerate(balance):
        if b >= peak:
            if in_drawdown:
                drawdown_end = i-1
                drawdowns[0].append(drawdown_start)
                drawdowns[1].append(drawdown_end)
                drawdowns[2].append(max_drawdown_value)
                max_drawdown_value = 0
            in_drawdown = False
            peak = b
        else:
            if not in_drawdown:
                drawdown_start = i-1
            in_drawdown = True
            current_drawdown_value = (peak - b) / peak
            if max_drawdown_value < current_drawdown_value:
                max_drawdown_value = current_drawdown_value
            if i == len(balance) - 1:
                drawdown_end = i
                drawdowns[0].append(drawdown_start)
                drawdowns[1].append(drawdown_end)
                drawdowns[2].append(max_drawdown_value)

    average_drawdown = np.mean(drawdowns[2]) * 100

    max_drawdown_byper = np.max(drawdowns[2]) * 100 # Convert max drawdown to percentage
    max_drawdown_byper_start = drawdowns[0][np.argmax(drawdowns[2])] # measured by percentage
    max_drawdown_byper_end = drawdowns[1][np.argmax(drawdowns[2])] # measured by percentage
    duration_of_max_drawdown = max_drawdown_byper_end - max_drawdown_byper_start # Duration in days of the maximum drawdown (by percent)
    
    drawdown_durations = np.asarray([drawdowns[1][i] - drawdowns[0][i] for i in range(len(drawdowns[0]))])
    max_drawdown_bydur = drawdowns[2][np.argmax(drawdown_durations)] * 100
    max_drawdown_bydur_start = drawdowns[0][np.argmax(drawdown_durations)] # measured by duration
    max_drawdown_bydur_end = drawdowns[1][np.argmax(drawdown_durations)] # measured by duration
    duration_of_longest_drawdown = max_drawdown_bydur_end - max_drawdown_bydur_start # Duration in days of the longest drawdown

    # Calculate average drawdown duration
    average_drawdown_duration = np.mean(drawdown_durations)

    # Calculate average yearly return
    average_yearly_return = ((balance[-1] / balance[0]) ** (252 / len(balance)) - 1) * 100

    return {
        "Balance" : balance,
        "drawdowns": drawdowns,
        "Avg Daily Return": average_daily_return,
        "Avg Drawdown": average_drawdown,
        "Avg Drawdown Duration (Days)": average_drawdown_duration,
        "Max Drawdown By Percent": max_drawdown_byper,
        "Max Drawdown By Percent Start": max_drawdown_byper_start,
        "Max Drawdown By Percent End": max_drawdown_byper_end,
        "Duration of Max Drawdown (Days)": duration_of_max_drawdown,
        "Max Drawdown By Duration": max_drawdown_bydur,
        "Max Drawdown By Duration Start": max_drawdown_bydur_start,
        "Max Drawdown By Duration End": max_drawdown_bydur_end,
        "Duration of Longest Drawdown (Days)": duration_of_longest_drawdown,
        "Avg Yearly Return": average_yearly_return
        }


day_num = 2000
starting_balance = 2000

def are_params_logical(solution):

    x_percent_loss = solution[0]
    y_percent_gain = solution[1]
    be_threshold = solution[2]
    tslp = solution[3]

    if -1*x_percent_loss > (y_percent_gain - tslp):
        return False
    elif be_threshold < -1 * x_percent_loss:
        return False
    elif be_threshold > y_percent_gain:
        return False
    elif tslp > y_percent_gain:
        return False
    elif x_percent_loss >= 0:
        return False
    elif y_percent_gain <= 0:
        return False
    elif be_threshold <= 0:
        return False
    elif tslp <= 0:
        return False
    
    return True

def fitness_function(ga_instance, solution, solution_idx):
    x_percent_loss = solution[0]
    y_percent_gain = solution[1]
    be_threshold = solution[2]
    tslp = solution[3]


    if not(are_params_logical(solution)):
        print("Solution did not satisfy logical requirements")
        return -99999

    profit_fitness = 0
    max_drawdown_fitness = 0
    longest_drawdown_dur_fitness = 0
    seeds = [int(i) for i in np.arange(0, 30, 1)]
    for seed in seeds:
        report = my_algorithm(seed, day_num, starting_balance, x_percent_loss / 100, y_percent_gain / 100, be_threshold / 100, tslp / 100)

        # Use this when using algorithm with no slippage
        profit_fitness += np.tanh(report["Avg Yearly Return"] / 4 - 2.5) # tanh function stretched and centered at 10
        max_drawdown_fitness += -1 * np.tanh(report["Max Drawdown By Percent"] - 3) # negative tanh function centered at 3.0
        longest_drawdown_dur_fitness += -1 * np.tanh(report["Duration of Longest Drawdown (Days)"] / 15 - 10) # Centered near 150

    # Average the fitnesses over the number of runs
    profit_fitness = profit_fitness / len(seeds) * 4 # More heavily weight profit over drawdowns
    max_drawdown_fitness = max_drawdown_fitness / len(seeds)
    longest_drawdown_dur_fitness = longest_drawdown_dur_fitness / len(seeds)

    total_fitness = profit_fitness + max_drawdown_fitness + longest_drawdown_dur_fitness # At it's peak this should have a value of 6
    print(profit_fitness, max_drawdown_fitness, longest_drawdown_dur_fitness, 1 / (6 - total_fitness))
    return 1 / (6 - total_fitness)

num_generations = 20
num_parents_mating = 8

# Define the number of solutions and genes
num_solutions = 16
num_genes = 4

# Define initialization ranges for each gene
gene_ranges = [
    (-1, -0.2),  # x_percent_loss
    (0.2, 5),      # y_percent_gain
    (0.2, 1.5),  # be_threshold
    (0.2, 5)   # tslp
]

# Initialize an empty population array
initial_population = np.empty((num_solutions, num_genes))

# Generate random values for each gene in each solution within the specified ranges
def generate_solution():
    solution = []
    for j in range(num_genes):
        gene_min, gene_max = gene_ranges[j]
        gene_value = np.random.uniform(gene_min, gene_max)
        solution.append(gene_value)
    while not(are_params_logical(solution)):
        solution = []
        for j in range(num_genes):
            gene_min, gene_max = gene_ranges[j]
            gene_value = np.random.uniform(gene_min, gene_max)
            solution.append(gene_value)
    return(solution)

initial_population = [generate_solution() for i in range(num_solutions)]

"""
Roulette Wheel Selection (rws): Roulette wheel selection assigns each individual in 
the population a slice of a roulette wheel, 
with the size of the slice proportional to their fitness. 
Then, a random spin of the wheel determines which individuals are selected as parents. 
This method tends to favor individuals with higher fitness values.
"""
parent_selection_type = "rws" # Roulette Wheel Selection
keep_parents = 0
keep_elitism = 1

mutation_probability = 0.4 # Every gene of every solution has a some percent chance of being mutated

# Define your custom callback function
def custom_callback(ga_instance):
    generation = ga_instance.generations_completed
    print(ga_instance.best_solution())
    print(ga_instance.population)

def crossover_func(parents, offspring_size, ga_instance):

    offspring = []

    for _ in range(offspring_size[0]):
        # Randomly select two different parents
        parent_indices = np.random.choice(range(len(parents)), size=2, replace=False)
        parent1 = parents[parent_indices[0], :].copy()
        parent2 = parents[parent_indices[1], :].copy()

        random_split_point = np.random.choice(range(offspring_size[0]))
        child_solution = np.concatenate((parent1[random_split_point:], parent2[:random_split_point]))
        counter = 0

        while not(are_params_logical(child_solution)):
            random_split_point = np.random.choice(range(offspring_size[0]))
            child_solution = np.concatenate((parent1[random_split_point:], parent2[:random_split_point]))
            if counter > 20:  # Avoid infinite loop
                print("Crossover got stuck in while loop with parents\n", parent1, "\n", parent2)
                child_solution = generate_solution()
            counter += 1

        offspring.append(child_solution)

    return np.array(offspring)

def mutation_func(offspring, ga_instance):

    minmax_by_gene = [[-1,1] for i in range(offspring.shape[1])]
    minmax_by_gene[0] = [-0.2, 0.2]

    for chromosome_idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if random.randint(1, 100) / 100 <= mutation_probability:

                mutation_arr = np.zeros(offspring.shape[1])
                mutation_arr[gene_idx] += np.random.uniform(minmax_by_gene[gene_idx][0], minmax_by_gene[gene_idx][1])
                new_solution = offspring[chromosome_idx] + mutation_arr
                
                counter = 0
                while not(are_params_logical(new_solution)):
                    mutation_arr = np.zeros(offspring.shape[1])
                    mutation_arr[gene_idx] += np.random.uniform(minmax_by_gene[gene_idx][0], minmax_by_gene[gene_idx][1])
                    new_solution = offspring[chromosome_idx] + mutation_arr
                    if counter > 20: # Avoid infinite loop
                        print("Mutation got stuck in while loop with initial values of", offspring[chromosome_idx])
                        new_solution = offspring[chromosome_idx] # Don't mutate anything
                        break
                    counter += 1
                
                offspring[chromosome_idx] = new_solution

    return offspring

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       initial_population = initial_population,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       keep_elitism=keep_elitism,
                       crossover_type=crossover_func,
                       mutation_type=mutation_func,
                       suppress_warnings=True,
                       on_generation=custom_callback,
                       allow_duplicate_genes=False,
                       save_solutions=True,
                       parallel_processing=["process", 16],
                       stop_criteria=["reach_0.8"])

if __name__ == '__main__':
    t1 = time.time()
    print("starting")
    ga_instance.run()
    with open('ga_instance.pkl', 'wb') as f:
        pickle.dump(ga_instance, f)
    t2 = time.time()
    print("Time is", t2-t1)