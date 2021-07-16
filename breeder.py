from __future__ import division
from random import randint, choice
from numpy import mean
import numpy as np
import pandas as pd
#ikisti iverti

aptLength = 40 # automatizuoti?

#made by sid for gluedtogether.py
def readDatabase(readfile):
    databaseSize = len(aptamers)
    return aptamers


def generateAptamer():
     return aptamer


def getFitness(sequences):
    return fitnessScore


def getBest(aptamerList, cutOff):
    top_aptamers = []
    top_aptamers.append(aptamerList[:int(len(aptamerList) * cutOff)])

    return  top_aptamers


def genPool(pool_size):
     return aptamerList


# parent1 and parent 2 are in the format [aptamer_1, 'asdasdasdasdasda', 47]
def crossover(parent1, parent2, iteration, generationNumber):
    crossPosition = randint(1,2) # ar galime taikyti distribucija ar kita randomizacija
    if crossPosition%2 == 0:
        childseq = parent1[1][:crossPosition] + parent2[1][crossPosition:]
    else:
        childseq = parent2[1][:crossPosition] + parent1[1][crossPosition:]

    child = ["gen_" + str(generationNumber) + "_offspring_" + str(iteration), childseq, getFitness(childseq)]
    return child


def breed(aptamers, generationNumber, top_cutoff=0.10):
    top_aptamers = []
    top_parents = getBest(aptamers)

    for child in range(int(databaseSize * 0.98)): # koki procenta norime numesti
        top_aptamers.append(crossover(choice(top_parents), choice(top_parents), child, generationNumber))

    return top_aptamers


def mutate(aptamer, mutation_probability = 0.02): #mutacijos
    input_sequence = aptamer[1]
    if random.random() <= mutation_probability:
        site = random.randint(0, aptLength - 2)
        listedSeq = list(input_sequence)
        notList = ['A', 'G', 'C', 'T']

        notList[site] == 'A':
        notList[site] = random.choice(notList.remove(notList[site]))
        
        return [aptamer[0], "".join(notList), ab.getFitness("".join(notList))] 