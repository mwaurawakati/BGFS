import numpy as np
import numpy
from main import mathematical_test_functions as mtf
from main import hybrid_optimisation_algorithms as hyb
from main import fidelity
import random

fit = mtf()
genes = 2
chromosomes = 10
mattingPoolSize = 6
offspringSize = chromosomes - mattingPoolSize
lb = -5
ub = 5
populationSize = (chromosomes, genes)
generations = 100


def fitness(x):
    """
    x: vector of input values
    """
    try:
        d = np.array(x).shape[-1]

        i = np.array([np.sqrt(i + 1) for i in range(d)])
        # print(i)
        ppp = ((np.array(x)) ** 2 / 4000)
        # print(ppp)
        # print(x)
        kkk = []
        for i in x:
            kk = np.cos(i)
            kkk.append(kk)
        # print(kkk)
        return np.sum(ppp, axis=-1)  # - np.prod(kkk, axis=-1) + 1
    except:
        return x


f = numpy.random.uniform(lb, ub, populationSize)
# fitness=mtf.griewank_function
rng = np.random.default_rng()
f = numpy.random.uniform(lb, ub, populationSize)

if __name__ == '__main__':
    hyb = hyb(f, fitness)
    s = np.size(f)
    fid = fidelity(f, s, fitness)

    rand_pop=rng.choice(f)
    new_population=hyb.BFGS(fitness,rand_pop,10)
    new_population=fid.fidelity_choice(new_population,)
    new_population=hyb.BFGS(fitness,new_population,10)
    print("The new population is :",new_population)


    fitness = fitness(new_population)
    try:
        fittestIndex = numpy.where(fitness == numpy.max(fitness))

    except:
        fittestIndex = np.random.choice(fitness)

    try:
        # Extracting index of fittest chromosome
        fittestIndex = fittestIndex[0][0]
        # Getting Best chromosome
        fittestInd = np.array(new_population)[np.array(fittestIndex), :]
        bestFitness = fitness[fittestIndex]
        print("\nBest Individual:")
        print(fittestInd)
        print("\nBest Individual's Fitness:")
        print(bestFitness)
    # print(new_population)
    except:
        print("""
        ==========================================================
        ==========================================================
        ==========================================================


        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        This algorithm uses random choices to optimize functions. 
        The current path taken did not converge.
        Run the program again
        However, depending with the number of choices, this function mat never converge
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        ==========================================================
        ==========================================================
        ==========================================================
        """)
        # print("The algorith did not converge")




