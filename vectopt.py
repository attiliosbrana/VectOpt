import numpy as np
from numba import njit, prange

# ==== PORTFOLIO STATS utils ==== #


# ==== PORTFOLIO METRICS utils ==== #

@njit
def portfolio_sharpe(assets, population):
    """
    Calculates the Sharpe ratio of a population of portfolio weights.
    """
    t_population = np.ascontiguousarray(population.T)
    dot = vo.get_dot(assets, t_population)
    ratios = []
    for i in prange(dot.shape[1]):
        ratios.append(dot[:, i].mean() / dot[:, i].std())
    return np.array(ratios)

@njit
def gini(x): 
    """
    Calculates gini of MULTIPLE weight matrices
    """
    x = x + 0.0000001
    x = sort_2d_array(x)
    n = x.shape[1]
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n  - 1) * x, axis = 1) / (n * np.sum(x, axis = 1))

# ==== PORTFOLIO BENCHMARKS utils ==== #



# ==== MATH utils ==== #

@njit
def sort_2d_array(x):
    '''Jitted efficient sorting over axis n, of n x m'''
    n = np.shape(x)[0]
    for row in range(n):
        x[row] = np.sort(x[row])
    return x

@njit
def get_dot(a, b):
    '''Jitted dot product'''
    return np.dot(a, b)

@njit
def numba_roll(a):
    '''Jitted np roll on axis n, of n x m'''
    b = np.empty_like(a)
    b[0] = a[-1]
    b[1:] = a[:-1]
    return b

@njit
def filter_pop(pop_fitness, trial_pop_fitness, population, trial_pop):
    '''Jitted implementation of np.where'''
    test = (pop_fitness > trial_pop_fitness)
    new_pop = np.zeros(population.shape)
    new_pop[test] = population[test]
    new_pop[~test] = trial_pop[~test]
    return new_pop

@njit
def permutations(population, pop_size):
    '''Jitted permutation'''
    return population[np.random.permutation(pop_size)]

#Cleaning (weight normalization) utils
@njit
def clean_up_weights(weights, min_thresh):
    '''Cleans weights based on a minimum threshold'''
    weights[weights < min_thresh] = 0
    return weights / weights.sum()

@njit
def clean_pop(pop, min_thresh):
    '''Applies weight cleaning on an entire population'''
    pop_size = pop.shape[0]
    for i in prange(pop_size):
        pop[i] = clean_up_weights(pop[i], min_thresh) 
    return pop


# ==== OPTIMIZER utils ==== #

@njit
def early_stopping_condition(hof_fit, early_stopping):
    '''Function to determine if the algorithm has not improved for n rounds'''
    hof_fit = np.array(hof_fit)
    test = hof_fit[-early_stopping:] - hof_fit[-early_stopping-1:-1]
    return (test > 1e-5).sum() == 0

@njit
def hall_of_fame(population, pop_fitness, hof_fit, hof_pop):
    '''Function to append to the hall_of_fame'''
    idx = np.argmax(pop_fitness)
    hof_fit.append(pop_fitness[idx])
    hof_pop.append(population[idx])
    return hof_fit, hof_pop

### Random portfolio generators
@njit
def random_weights(num_of_assets):
    '''Function to generate a random portfolio of weights adding up to one'''
    return np.random.dirichlet(np.ones(num_of_assets))

@njit
def random_num_of_assets(shape, max_port):
    '''Function to choose a random integer number of assets to pick'''              # THIS NEEDS PARAMETERS FOR MIN MAX
    return np.random.randint(1, max(min([max_port, shape]), 2))

@njit
def random_idxs(assets, num_of_tokens):
    '''Function to choose a random index of assets to pick, based on a number'''
    return np.random.choice(assets, num_of_tokens, replace = False)

@njit
def random_subset_portfolio(sol_size, max_port):
    '''Function to select a subset portfolio of assets'''
    num_of_assets = random_num_of_assets(sol_size, max_port)
    assets_idx = random_idxs(sol_size, num_of_assets)
    return num_of_assets, assets_idx 

@njit
def new_portfolio_weights(sol_size, min_thresh, max_port):
    '''Creates a single clean (rounded to a minimum) portfolio of weights'''
    num_of_assets, assets_idx  = random_subset_portfolio(sol_size, max_port)
    weights = clean_up_weights(random_weights(num_of_assets), min_thresh)
    port = np.zeros(sol_size)
    port[assets_idx] = weights
    return port

@njit
def new_portfolio_batch(pop_size, sol_size, min_thresh, max_port):
    '''
    Determines a new batch of many porfolios n x m
    n = pop_size,
    m = sol_size

    Iteration is done because it is faster in numba
    '''
    batch = np.zeros((pop_size, sol_size))
    for i in prange(pop_size):
        batch[i] = new_portfolio_weights(sol_size, min_thresh, max_port)
    return batch

### Init Function
@njit
def init_pop(pop_size, sol_size, assets, get_fitness, min_thresh, max_port, init_cases):
    '''Initiates a population, clean, and with fitness. Also initiates hall_of_fame.'''
    population = new_portfolio_batch(pop_size, sol_size, min_thresh, max_port)
    if init_cases != None:
        n = init_cases.shape[0]
        population[:n] = init_cases
    pop_fitness = get_fitness(assets, population)
    idx = np.argmax(pop_fitness)
    hof_fit = [pop_fitness[idx]]
    hof_pop = [population[idx]]
    return population, pop_fitness, hof_fit, hof_pop


# ==== DIFFERENTIAL EVOLUTIONARY ==== #

@njit
def trial_population(pop, pop_size, diff_weight):
    '''Creates three population permutations, calculates a midpoint in the triangle''' 
    v1 = permutations(pop, pop_size)
    v2 = permutations(pop, pop_size)
    v3 = permutations(pop, pop_size)
    trial_population = v1 * (1 - diff_weight) + (v2 + v3) / 2 * diff_weight
    return trial_population

@njit
def DE_optim(assets, sol_size, pop_size, rounds, min_gens, get_fitness, early_stopping,
                min_thresh, diff_weight = 0.5, max_port = 41, init_cases = None):
    '''Entire Differential Evolutionary algorithm''' 
    
    #Intiates new population, fitness and hall_of_fame
    population, pop_fitness, hof_fit, hof_pop = init_pop(pop_size, sol_size, assets, get_fitness,
                                                            min_thresh, max_port, init_cases)

    #Iter for rounds
    for i in prange(rounds):

        #Creates three permutations of the population, converges to midpoint
        trial_pop = trial_population(population, pop_size, diff_weight)
        trial_pop_fitness = get_fitness(assets, trial_pop)

        #Filters the counterparts with best fit from old and trial population, cleans, gets fitness
        population = filter_pop(pop_fitness, trial_pop_fitness, population, trial_pop) 
        population = clean_pop(population, min_thresh)
        pop_fitness = get_fitness(assets, population)

        #Adds round results to hall_of_fame
        hof_fit, hof_pop = hall_of_fame(population, pop_fitness, hof_fit, hof_pop) 
        
        #Minimum generations to start Early Stopping
        if i > min_gens:
            if early_stopping_condition(hof_fit, early_stopping):
                break
    
    return hof_fit, hof_pop


# ==== DISPERSIVE FLIES OPTIMIZATION ==== #

@njit
def best_neighbours(population, pop_fitness):
    '''Compares the neighbours of two populations, selects the best''' 
    fit_rolled = numba_roll(pop_fitness)
    pop_rolled = numba_roll(population)
    best = filter_pop(pop_fitness, fit_rolled, population, pop_rolled)
    return best

@njit
def crossover(population, pop_size, trial_pop, disturb_thresh):
    '''Jiited implementation of np.where but based on a threshold''' 
    test = np.random.uniform(0, 1, (pop_size)) > disturb_thresh 
    new_pop = np.zeros(population.shape)
    new_pop[test] = population[test]
    new_pop[~test] = trial_pop[~test]
    return new_pop

@njit
def DFO_optim(assets, sol_size, pop_size, rounds, min_gens, get_fitness, early_stopping,
                min_thresh, move_amount = 0.25, disturb_thresh = 0.05, max_port = 41, init_cases = None):
    '''Entire Dispersive Flies Optimization algorithm'''  

    #Intiates new population, fitness and hall_of_fame
    population, pop_fitness, hof_fit, hof_pop = init_pop(pop_size, sol_size, assets, get_fitness,
                                                            min_thresh, max_port, init_cases)

    for i in prange(rounds):
        #Best case so far
        best = hof_pop[-1]
        
        #Filters for the best neighbours
        trial_pop = best_neighbours(population, pop_fitness)
        
        #Update in the direction of best
        population = trial_pop * (1 - move_amount) + best * move_amount
        
        #Generates trial population, randomly mixes it with old pop based on threshold, cleans and gets fitness
        trial_pop = new_portfolio_batch(pop_size, sol_size, min_thresh, max_port)
        trial_pop = crossover(population, pop_size, trial_pop, disturb_thresh)
        population = (population + trial_pop) / 2 
        population = clean_pop(population, min_thresh)
        pop_fitness = get_fitness(assets, population)

        #Adds round results to hall_of_fame
        hof_fit, hof_pop = hall_of_fame(population, pop_fitness, hof_fit, hof_pop)
        
        #Minimum generations to start Early Stopping
        if i > min_gens:
            if early_stopping_condition(hof_fit, early_stopping):
                break
    
    return hof_fit, hof_pop