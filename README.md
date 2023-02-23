# VectOpt - Vectorized Portfolio Optimization Library

VectOpt is a Python library that provides vectorized implementation of two meta-heuristic optimization algorithms for portfolio selection: Differential Evolutionary (DE) and Dispersive Flies Optimization (DFO).

DE is an efficient version of Evolutionary Algorithms (EAs) that uses difference vectors among candidate solutions to sample the problem space. DFO is a Swarm Intelligence-based metaheuristic that simulates the swarming behavior of flies over food sources in nature.

VectOpt is designed to be used for portfolio optimization tasks, such as asset allocation. The library is designed to be computationally efficient, using matrix operations to minimize the use of for-loops, and allowing for parallelization on GPUs.

## Installation [TO BE DONE]

You can install VectOpt using pip:

```bash
pip install vectopt
```

## Usage

Here is a simple example of how to use VectOpt to optimize a portfolio using the DE algorithm:

```python
import vectopt as vo
import numpy as np

# Define the portfolio optimization problem
def get_fitness(assets, weights):
    returns = np.dot(assets, weights)
    risk = np.dot(weights, np.dot(assets, assets)) ** 0.5
    return returns / risk

assets = np.random.randn(100, 10)
sol_size = assets.shape[1]
pop_size = 100
rounds = 100
min_gens = 20
early_stopping = 10
min_thresh = 0.001

# Run the DE optimization algorithm
best_fit, best_sol = vo.DE_optim(assets, sol_size, pop_size, rounds, min_gens, get_fitness, early_stopping, min_thresh)

print(f"Best fitness: {best_fit[-1]}")
print(f"Best solution: {best_sol[-1]}")
```


Here is a similar example using the DFO algorithm:

```python
import vectopt as vo
import numpy as np

# Define the portfolio optimization problem
def get_fitness(assets, weights):
    returns = np.dot(assets, weights)
    risk = np.dot(weights, np.dot(assets, assets)) ** 0.5
    return returns / risk

assets = np.random.randn(100, 10)
sol_size = assets.shape[1]
pop_size = 100
rounds = 100
min_gens = 20
early_stopping = 10
min_thresh = 0.001

# Run the DFO optimization algorithm
best_fit, best_sol = vo.DFO_optim(assets, sol_size, pop_size, rounds, min_gens, get_fitness, early_stopping, min_thresh)

print(f"Best fitness: {best_fit[-1]}")
print(f"Best solution: {best_sol[-1]}")
```

## Contributing

VectOpt is an open-source project and contributions are welcome. If you find a bug, or have an idea for a new feature, please open an issue on the [GitHub repository](https://github.com/attiliosbrana/VectOpt). If you would like to contribute code, please open a pull request.

## License

VectOpt is licensed under the MIT License. See the [LICENSE](https://github.com/attiliosbrana/VectOpt/blob/main/LICENSE) file for more information.