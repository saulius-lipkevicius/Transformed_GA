from __future__ import division
from random import randint, choice
import random, decimal
from numpy import mean
import numpy as np
import pandas as pd #reading csv


#Read csv sheet, outputs already the best
def readCSV(readfile):
    df = pd.read_csv(readfile)
    df.sort_values(by=['Fitness'], inplace=True, ascending=False, ignore_index=True)
    df = pd.DataFrame(df)
    df = df.iloc[:int(len(df)*0.05),:]

    return df


# parents are in the format of ['ACGTCGT', fitness score]
def crossover(parent1, parent2): #prideti iteracija ir generacija
    crossPosition = randint(1,38) # ar galime taikyti distribucija ar kita randomizacija
    initialParent = randint(0,1)

    if initialParent == 0:
        childSeq = parent1[:crossPosition] + parent2[crossPosition:]
    else:
        childSeq = parent2[:crossPosition] + parent1[crossPosition:]

    return str(childSeq)


def breed(aptamers): #prideti kad geresni parent poruotusi dazniau
    top_parents = aptamers.iloc[:,0]
    for child in range(0,190): # 1 - kiek paliekame, lyginti su getBest funkcija
        newRow = {"Aptamer": crossover(choice(top_parents), choice(top_parents)), "Fitness": random.randint(9000, 10000)/10000}
        aptamers = aptamers.append(newRow, ignore_index=True)
    return aptamers


#for each base and not once
def mutate(aptamers, mutation_probability = 0.1): #mutacijos
  
    for row in range(0,len(aptamers) - 1):
        print(row)
        aptamer = list(aptamers.iloc[row,0])
        for i in range(0,39):
            if random.random() <= mutation_probability:
                notList = ['A', 'G', 'C', 'T']
                notList.remove(aptamer[i])
                aptamer[i] = random.choice(notList)
                print("located at: ", row, i)   
            aptamers.iloc[row,0] = "".join(aptamer)
    return aptamers


data = readCSV('AptamersList.csv')
data = breed(data)
print(len(data))
#data = mutate(data.iloc[0,0])
print(data)
print(mutate(data))
#print(data)

