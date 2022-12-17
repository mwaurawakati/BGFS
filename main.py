from mpl_toolkits.mplot3d import Axes3D
import math
import scipy
from abc import ABC,abstractmethod
import random
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from matplotlib import cm


class mathematical_test_functions:
    def __init__(self,):
        print(('mathematical test function'))
    """
        This is the class that houses mathematical test functions to be optimised
    
        """


    def rastrigin_function(self, *X, **kwargs):
        """
        https://en.wikipedia.org/wiki/Rastrigin_function
        Non-convex function for testing optimization algorithms.
        =====================================================

        params x: is a member of any integer and float value between 5.12 and -5.12
        param A: is equal to 10(**kwargs stands for A=10) You can change the value
        input *X is a list or matrix of float or numbers between -5.12 and 5.12
        """

        A = kwargs.get('A', 10)
        return A + sum([(x ** 2 - A * np.cos(2 * math.pi * x)) for x in X])


    def rosenbrock_function(self, x1, x2):
        """
                https://en.wikipedia.org/wiki/Rosenbrock_function
                Non-convex function for testing optimization algorithms.
                =====================================================

                The function is made up of two inputs x and y
                it has two constants and be where
                param b is set at 100
                and param a is set at 1
                """
        return 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2


    def auckley_function(self, x, **kwargs):
        """
        x: vector of input values
        """
        A = kwargs.get('A', 20)
        B = kwargs.get('B', 0.2)
        C = kwargs.get('C', 2 * np.pi)
        d = x.shape[-1]

        sum_sq_term = -A * np.exp(-B * np.sqrt(np.sum(x * x, axis=-1) / d))

        cos_term = -np.exp(np.sum(np.cos(C * x) / d, axis=-1))

        return A + np.exp(1) + sum_sq_term + cos_term


    def griewank_function(self, x):
        """
        x: vector of input values
        """
        d = np.array(x).shape[-1]

        i = np.array([np.sqrt(i + 1) for i in range(d)])

        return np.sum(np.array(x) ** 2 / 4000, axis=-1) - np.prod(np.cos(x / i), axis=-1) + 1


    def Michalkiewicz_function(self, x, **kwargs):
        """
        x: vector of input values
        """
        M = kwargs.get('M', 2)
        d = x.shape[-1]

        return -np.sum(np.sin(x) * (np.sin(np.ones_like(x) * np.arange(1., d + 1.) * (x ** 2) / np.pi) ** (2 * M)),
                       axis=-1)



class hybrid_optimisation_algorithms:

    def __init__(self,population,fitness):
        self.population=population
        self.fitness = fitness

    # mutation operator
    def mutation(self,bitstring, r_mut):
        """

        :param bitstring:
        :param r_mut:
        :return:
        """
        for i in range(len(bitstring)):
            # check for a mutation
            if np.rand() < r_mut:
                # flip the bit
                bitstring[i] = 1 - bitstring[i]

    def selection(pop, scores, k=3):
        # first random selection
        selection_ix = np.randint(len(pop))
        for ix in np.randint(0, len(pop), k - 1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]

    def crossover(self,p1, p2, r_cross):
        '''

        :param p2:
        :param r_cross:
        :return:
        '''
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if np.rand() < r_cross:
            # select crossover point that is not on the end of the string
            pt = np.randint(1, len(p1) - 2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    def grad(self,f, x):
        '''
        CENTRAL FINITE DIFFERENCE CALCULATION
        '''
        h = np.cbrt(np.finfo(float).eps)
        d = len(x)
        # print(d)
        nabla = []  # np.zeros(d)
        for i in range(d):
            x_for = np.copy(x)
            x_back = np.copy(x)
            x_for = np.array(x_for)
            x_back = np.array(x_back)
            x_for[i] += h
            x_back[i] -= h
            nabl = (f(np.array(x_for)) - f(np.array(x_back))) / (2 * h)
            # print(nabl)
            nabla.append(nabl)
            # nabla=np.array(nabla)
        return nabla

    def line_search(self,f, x, p, nabla):
        '''
        BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
        '''
        a = 1
        c1 = 1e-4
        c2 = 0.9
        fx = f(np.array(x))
        x_new = x + a * p
        nabla_new = self.grad(f, np.array(x_new))
        # while np.logical_or((f(np.array(x_new)) >= np.array(fx) + c1*a*np.array(nabla).T@p),(np.array(nabla_new).T@p <= c2*np.array(nabla).T@p)).any():
        a *= 0.5
        x_new = x + a * p
        nabla_new = self.grad(f, x_new)
        # print(nabla_new)
        return a

    def BFGS(self,f, x0, max_it):
        '''
        DESCRIPTION
        BFGS Quasi-Newton Method, implemented as described in Nocedal:
        Numerical Optimisation.


        INPUTS:
        f:      function to be optimised
        x0:     intial guess
        max_it: maximum iterations
        plot:   if the problem is 2 dimensional, returns
                a trajectory plot of the optimisation scheme.

        OUTPUTS:
        x:      the optimal solution of the function f

        '''
        d = len(x0)  # dimension of problem
        nabla = self.grad(f, x0)  # initial gradient
        H = np.eye(d)  # initial hessian
        x = x0[:]
        it = 2
        while np.linalg.norm(nabla) > 1:  # while gradient is positive
            if it > max_it:
                print('Maximum iterations reached!')
                break
            it += 1
            p = -H @ nabla  # search direction (Newton Method)
            a = self.line_search(f, x, p, nabla)  # line search
            s = a * p
            x_new = x + a * p
            nabla_new = self.grad(f, x_new)
            y = np.array(nabla_new) - np.array(nabla)
            y = np.array([y])
            s = np.array([s])
            y = np.reshape(y, (d, -1))
            s = np.reshape(s, (d, -1))
            r = 1 / (y.T @ s)

            tt = s @ (y.T)

            # mm=np.dot((np.array(r)).reshape(10,10),(np.array(tt)))
            # mmm= (r,np.swapaxes(1,1) * tt).np.swapaxes(1,1)#np.mat(r)*np.mat(tt)
            # mmmm=np.transpose(np.array([r]))*tt
            li = (np.eye(d)) - (((s @ (y.T))))
            ri = (np.eye(d) - (((y @ (s.T)))))
            hess_inter = li @ H @ ri
            H = hess_inter + (((s @ (s.T))))  # BFGS Update
            nabla = nabla_new[:]
            x = x_new[:]

        return x

    def genetic_algorithm(self,population,fitness, mattingPoolSize=None , ub=None, lb=None, offspringSize=None):
        if mattingPoolSize==None:
            self.genes = 2
            self.chromosomes = 10
            self.mattingPoolSize = 6
        if offspringSize==None:
            self.offspringSize = self.chromosomes - self.mattingPoolSize
        if lb==None:
            self.lb = -5
        if ub==None:
            self.ub = 5
            self.populationSize = (self.chromosomes, self.genes)
            self.generations = 100

        #print(np.size(population))
        try:
            for generation in range(generations):
                #print(("Generation:", generation + 1))
                # fitness = mtf.griewank_function(population,population)#numpy.sum(np.array(population)*np.array(population), axis=1)
                #print("\npopulation")
                #print(population)
                #print("\nfitness calcuation")
                #print(fitness)
                # Following statement will create an empty two dimensional array to store parents
                parents = numpy.empty((self.mattingPoolSize, np.array(self.population).shape[1]))

                # A loop to extract one parent in each iteration
                for p in range(self.mattingPoolSize):
                    # Finding index of fittest chromosome in the population
                    fittestIndex = numpy.where(fitness == numpy.max(fitness))
                    # Extracting index of fittest chromosome
                    fittestIndex = fittestIndex[0][0]
                    # Copying fittest chromosome into parents array
                    parents[p, :] = np.array(population)[fittestIndex, :]
                    # Changing fitness of fittest chromosome to avoid reselection of that chromosome
                    # fitness[fittestIndex] = -1
                #print("\nParents:")
                #print(parents)

                # Following statement will create an empty two dimensional array to store offspring
                offspring = numpy.empty((self.offspringSize, np.array(self.population).shape[1]))
                for k in range(self.offspringSize):
                    # Determining the crossover point
                    crossoverPoint = numpy.random.randint(0, self.genes)

                    # Index of the first parent.
                    parent1Index = k % parents.shape[0]

                    # Index of the second.
                    parent2Index = (k + 1) % parents.shape[0]

                    # Extracting first half of the offspring
                    offspring[k, 0: crossoverPoint] = parents[parent1Index, 0: crossoverPoint]

                    # Extracting second half of the offspring
                    offspring[k, crossoverPoint:] = parents[parent2Index, crossoverPoint:]
                #print("\nOffspring after crossover:")
                #print(offspring)

                # Implementation of random initialization mutation.
                for index in range(offspring.shape[0]):
                    randomIndex = numpy.random.randint(1, self.genes)
                    randomValue = numpy.random.uniform(self.lb, self.ub, 1)
                    offspring[index, randomIndex] = offspring[index, randomIndex] + randomValue
                #print("\n Offspring after Mutation")
                #print(offspring)

                population.append(np.array(parents))
                population.append(np.array(offspring))
                #print("\nNew Population for next generation:")
                #print(population)
                return population
        except:
            return population
class fidelity:
    def __init__(self,population,NP,fitness):
        self.population=population
        self.initial_population_size=NP
        self.fitness=fitness
    def fidelity_choice(self,population):
        if np.size(population)<0.5*self.initial_population_size:
            fidelity=self.high_fidelity(population)
            return fidelity
        else:
            fidelity=self.low_fidelity(population)
            return fidelity
    def low_fidelity(self,population):
        try:
            Flf=[]
            A=0.5
            a=np.full(random.randint(0,len(population)),4)
            omega=random.randint(10,1000)
            f2=[]
            for i in range(0,len(a)):
                f = (population[i] - a[i]) ** 2
                f2.append(f)
            total = A * np.sin(omega * (sum(f2)))
            for i in range(len(population)):
                Fl = population[i] + total
                Flf.append(Fl)
            population=Flf
            return population
        except:
            print('Lower fidelity evaluation failed. Original population passeed')
            return self.population

    def high_fidelity(self,population):
        try:
            hyb=hybrid_optimisation_algorithms(population,self.fitness)
            population=self.low_fidelity(population)
            population=hyb.genetic_algorithm(population,self.fitness)
            return population
        except:
            return population

class Individual(ABC):
    def __init__(self, value=None, init_params=None):
        if value is not None:
            self.value = value
        else:
            self.value = self._random_init(init_params)

    @abstractmethod
    def pair(self, other, pair_params):
        pass

    @abstractmethod
    def mutate(self, mutate_params):
        pass

    @abstractmethod
    def _random_init(self, init_params):
        pass


class Optimization(Individual):
    def pair(self, other, pair_params):
        return Optimization(pair_params['alpha'] * self.value + (1 - pair_params['alpha']) * other.value)

    def mutate(self, mutate_params):
        self.value += np.random.normal(0, mutate_params['rate'], mutate_params['dim'])
        for i in range(len(self.value)):
            if self.value[i] < mutate_params['lower_bound']:
                self.value[i] = mutate_params['lower_bound']
            elif self.value[i] > mutate_params['upper_bound']:
                self.value[i] = mutate_params['upper_bound']

    def _random_init(self, init_params):
        return np.random.uniform(init_params['lower_bound'], init_params['upper_bound'], init_params['dim'])


class Population:
    def __init__(self, size, fitness, individual_class, init_params):
        """"
        This is the class that deals with the initialisation, creation and replacecement of polulation
        :param size:This is the size of Ns population
        :param fitness is the function being optimized
        :param individual_class: is the optimization class
        :param init_params:initial parameters
        """
        self.fitness = fitness
        self.individuals = [individual_class(init_params=init_params) for _ in range(size)]
        self.individuals.sort(key=lambda x: self.fitness(x))
        X = np.linspace(-4, 4, 200)

        #size = 100


    def selection(self,size):
        Ns=random.randint(size(self.individuals))

    def replace(self, new_individuals):
        """
        This is a function to replace members of a population after an evaluation or evolution
        :param new_individuals: This are the offsprings of the population
        :return:
        """
        size = len(self.individuals)
        self.individuals.extend(new_individuals)
        self.individuals.sort(key=lambda x: self.fitness(x))
        self.individuals = self.individuals[-size:]

    def get_parents(self, n_offsprings):
        """
        This is the function used to select the members of a population to be used in 'siring' offsprings'
        :param n_offsprings: the number of new members of the population to be created
        :return:
        """
        mothers = self.individuals[-2 * n_offsprings::2]
        fathers = self.individuals[-2 * n_offsprings + 1::2]

        return mothers, fathers
class GA:
    def __init__(self,population, objective,bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
        """
        :param population: population
        :param objective:mathematical test function
        :param bounds: range for inputs
        :param n_bits: bits per variable
        :param n_iter: define the total iterations
        :param n_pop: define the population size
        :param r_cross: crossover rate
        :param r_mut: mutation rate
        """

        # initial population of random bitstring
        if population is None:
            pop = [randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
        else:
            pop =population
        # keep track of best solution
        best, best_eval = 0, objective(self.decode(bounds, n_bits, pop[0]))
        # enumerate generations
        for gen in range(n_iter):
            # decode population
            decoded = [self.decode(bounds, n_bits, p) for p in pop]
            # evaluate all candidates in the population
            scores = [objective(d) for d in decoded]
            # check for new best solution
            for i in range(n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    print(">%d, new best f(%s) = %f" % (gen, decoded[i], scores[i]))
            # select parents
            selected = [self.selection(pop, scores) for _ in range(n_pop)]
            # create the next generation
            children = list()
            for i in range(0, n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]
                # crossover and mutation
                for c in self.crossover(p1, p2, r_cross):
                    # mutation
                    self.mutation(c, r_mut)
                    # store for next generation
                    children.append(c)
            # replace population
            self.pop = children

    def __repr__(self):
        return self.pop

    def decode(self,bounds, n_bits, bitstring):
        decoded = list()
        largest = 2 ** n_bits
        for i in range(len(bounds)):
            # extract the substring
            start, end = i * n_bits, (i * n_bits) + n_bits
            substring = bitstring[start:end]
            # convert bitstring to a string of chars
            chars = ''.join([str(s) for s in substring])
            # convert string to integer
            integer = int(chars, 2)
            # scale integer to desired range
            value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
            # store
            decoded.append(value)
        return decoded

    # tournament selection
    def selection(self,pop, scores, k=3):
        # first random selection
        selection_ix = randint(len(pop))
        for ix in randint(0, len(pop), k - 1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]

    # crossover two parents to create two children
    def crossover(self,p1, p2, r_cross):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if randint(0,r_cross) < r_cross:
            # select crossover point that is not on the end of the string
            pt = randint(1, len(p1) - 2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    # mutation operator
    def mutation(self,bitstring, r_mut):
        for i in range(len(bitstring)):
            # check for a mutation
            if randint(0,r_mut) < r_mut:
                # flip the bit
                bitstring[i] = 1 - bitstring[i]

genes = 2
chromosomes = 10
mattingPoolSize = 6
offspringSize = chromosomes - mattingPoolSize
lb = -5
ub = 5
populationSize = (chromosomes, genes)
generations = 100


import numpy
from main import mathematical_test_functions as mtf
fit=mtf()


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
    hyb = hybrid_optimisation_algorithms(f, fitness)
    s=np.size(f)
    print(s)
    new_population=[]
    for i in range(generations):
        fid=fidelity(f,s,fitness)
        new_population=fid.fidelity_choice(f)
        determiner=random.randint(0,1)
        if determiner==1:
            new_population=hyb.genetic_algorithm(new_population,fitness)#mattingPoolSize,,ub,lb,offspringSize)
            new_population1 = rng.choice(new_population, 1)
            if np.size(new_population1)==np.size(new_population):
                new_population=new_population1
            else:
                s1=np.size(new_population)
                for i in range(len(new_population1)):
                    hyb=hybrid_optimisation_algorithms(new_population1,fitness)
                    new_population=hyb.BFGS(fitness,new_population1,100)
                    fid = fidelity(new_population, s1,fitness)
                    new_population = fid.fidelity_choice(new_population)
        else:
            new_population=new_population
            determiner1 = random.randint(0, 1)
            if determiner1 == 1:
                new_population_indices = np.random.choice(len(new_population),size=round(0.5*np.size(new_population)))
                new_population=hyb.genetic_algorithm(new_population, fitness)#(new_population), mattingPoolSize ub=ub, lb=lb, offspringSize=offspringSize)
            else:
                new_population=hyb.genetic_algorithm(new_population,fitness)#=fitness(new_population) mattingPoolSize, , ub=ub, lb=lb, offspringSize=offspringSize)
    fitness = fitness(new_population)
    #fittestIndex=[]
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
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        ==========================================================
        ==========================================================
        ==========================================================
        """)
        # print("The algorith did not converge")


