import numpy as np
import numpy
from main import mathematical_test_functions as mtf
from main import hybrid_optimisation_algorithms as hyb
from main import fidelity
import random
fit=mtf()
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
        #print(i)
        ppp=((np.array(x)) ** 2 / 4000)
        #print(ppp)
        #print(x)
        kkk=[]
        for  i in x:
            kk=np.cos(i)
            kkk.append(kk)
        #print(kkk)
        return np.sum(ppp, axis=-1)# - np.prod(kkk, axis=-1) + 1
    except:
        return x
f= numpy.random.uniform(lb, ub, populationSize)
#fitness=mtf.griewank_function
rng = np.random.default_rng()
f= numpy.random.uniform(lb, ub, populationSize)

if __name__ == '__main__':
    hyb = hyb(f, fitness)
    s = np.size(f)

    new_population=[]
    for i in range(generations):
        fid=fidelity(f,s,fitness)
        new_population=fid.fidelity_choice(f)
        determiner=random.randint(0,1)
        if determiner==1:
            new_population = rng.choice(new_population, 1)
        else:
            new_population=new_population
        new_population=hyb.genetic_algorithm(new_population,fitness)

    fitness = fitness(new_population)
    try:
         fittestIndex = numpy.where(fitness == numpy.max(fitness))

    except:
        fittestIndex=rng.choice(fitness)

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
    #print(new_population)
    except:
      print(  """
        ==========================================================
        ==========================================================
        ==========================================================
        
        
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        This algorithm uses random choices to optimize functions. 
        The current path taken did not converge.
        Run the program again
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        
        ==========================================================
        ==========================================================
        ==========================================================
        """)
        #print("The algorith did not converge")




