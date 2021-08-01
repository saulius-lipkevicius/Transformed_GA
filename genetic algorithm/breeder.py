from __future__ import division
from random import randint, choice
import random, decimal
from numpy import mean
import numpy as np
import pandas as pd


# parents are in the format of ['ACGTCGT', fitness score]
def crossover(parent1, parent2): #prideti iteracija ir generacija
    crossPosition = randint(1,14) # ar galime taikyti distribucija ar kita randomizacija
    initialParent = randint(0,1)

    if initialParent == 0:
        childSeq = parent1[:crossPosition] + parent2[crossPosition:]
    else:
        childSeq = parent2[:crossPosition] + parent1[crossPosition:]
    childSeq = mutate(childSeq)

    return str(childSeq)


#for each base and not once
def mutate(aptamer, mutation_probability = 0.002): #mutacijos
    apt = [char for char in aptamer]
    for i in range(0,15):
        if random.random() <= mutation_probability:
            notList = ['A', 'G', 'C', 'T']
            notList.remove(apt[i])
            apt[i] = random.choice(notList) 
    
    aptamer = "".join(apt) 
    
    return aptamer


def breed(dataset): #prideti kad geresni parent poruotusi dazniau
    df = list(dataset['Sequence'])
    aptamers = []
    for child in range(0,900): # tiesiog while loop padarom
        newAptamer = crossover(choice(df), choice(df))
        aptamers.append(newAptamer)

    aptamers = aptamers + df
    aptamers = pd.DataFrame(aptamers)
    aptamers.columns = ['Sequence']
    return aptamers

def main():
    dataset = pd.read_csv('./datasets/GA_iterations/top100_1')
    afterBreed = breed(dataset)

    location = './datasets/GA_iterations/breeding_{}'.format(1)
    afterBreed.to_csv(location, encoding='utf-8', index=False)


if __name__=="__main__":
    main()
