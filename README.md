# VectOpt - Vectorized Portfolio Optimization Library

VectOpt is a Python library that provides vectorized implementation of two meta-heuristic optimization algorithms for portfolio selection: Differential Evolutionary (DE) and Dispersive Flies Optimization (DFO).

DE is an efficient version of Evolutionary Algorithms (EAs) that uses difference vectors among candidate solutions to sample the problem space. DFO is a Swarm Intelligence-based metaheuristic that simulates the swarming behavior of flies over food sources in nature.

VectOpt is designed to be used for portfolio optimization tasks, such as asset allocation. The library is designed to be computationally efficient, using matrix operations to minimize the use of for-loops, and allowing for parallelization on GPUs.

## Environment

We suggest a conda environment in order to run the Numba-optimized version of VectOpt.

```bash
conda create --name vectopt_env python=3.10
conda activate vectopt_env
conda install numpy=1.22.4 numba=0.55.2 matplotlib=3.6.2
```

## Installation [pip TBD]

For now, we suggest cloning the repository.

In the near future, we will seek installation via pip:

```bash
pip install vectopt
```

## Usage

Here is a simple example of how to use VectOpt to optimize a portfolio using the DE algorithm:

```python
import vectopt as vo
import numpy as np
from numba import njit, prange

@njit
def get_fitness(assets, population):
    """ calculates the sharpe ratio of the portfolio """
    t_population = np.ascontiguousarray(population.T)
    dot = vo.get_dot(assets, t_population)
    ratios = []
    for i in prange(dot.shape[1]):
        ratios.append(dot[:, i].mean() / dot[:, i].std())
    return np.array(ratios)

#make 10 random assets but only with positive values, and 100 observations
assets = np.random.rand(100, 10)
sol_size = assets.shape[1]
pop_size = 100
rounds = 100
min_gens = 20
early_stopping = 10
min_thresh = 0.01

# Run the DE optimization algorithm
hof_fit, hof_pop = vo.DE_optim(assets, sol_size, pop_size, rounds, min_gens, get_fitness, early_stopping, min_thresh)

print(f"Best fitness: {hof_fit[-1]}")
print(f"Best solution: {hof_pop[-1]}")
```


Here is a similar example using the DFO algorithm:

```python
import vectopt as vo
import numpy as np
from numba import njit, prange

@njit
def get_fitness(assets, population):
    """ calculates the sharpe ratio of the portfolio """
    t_population = np.ascontiguousarray(population.T)
    dot = vo.get_dot(assets, t_population)
    ratios = []
    for i in prange(dot.shape[1]):
        ratios.append(dot[:, i].mean() / dot[:, i].std())
    return np.array(ratios)

#make 10 random assets but only with positive values, and 100 observations
assets = np.random.rand(100, 10)
sol_size = assets.shape[1]
pop_size = 100
rounds = 100
min_gens = 20
early_stopping = 10
min_thresh = 0.01

# Run the DE optimization algorithm
hof_fit, hof_pop = vo.DFO_optim(assets, sol_size, pop_size, rounds, min_gens, get_fitness, early_stopping, min_thresh)

print(f"Best fitness: {hof_fit[-1]}")
print(f"Best solution: {hof_pop[-1]}")
```

## Contributing

VectOpt is an open-source project and contributions are welcome. If you find a bug, or have an idea for a new feature, please open an issue on the [GitHub repository](https://github.com/attiliosbrana/VectOpt). If you would like to contribute code, please open a pull request.

## License

VectOpt is licensed under the MIT License. See the [LICENSE](https://github.com/attiliosbrana/VectOpt/blob/main/LICENSE) file for more information.