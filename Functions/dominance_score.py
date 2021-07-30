import pandas as pd
import numpy as np

def dominanceScore(dataset, initialAptamers):
    power = {}

    #initial aptamers, pataisyti kad imtu is kitos funkcijos
    for i in range(0,len(initialAptamers)):
        power.update({aptList.loc[i,'Sequence']: 0})

    #dont have to normalize the score
    for t in range(0, len(df)):
        if dataset.loc[t,'Label'] == 1:
            power[dataset.loc[t,'Sequence1']] += 1/len(initialAptamers)
        else:
            power[dataset.loc[t,'Sequence2']] += 1/len(initialAptamers)

    return power
#print(sorted(power.items(), key=lambda x:x[1], reverse=True) )

if __name__ == "__main__":

    dataPath = './datasets/full_comparison.csv'
    df = pd.read_csv(dataPath)

    aptamerPath = './datasets/scored_sequences.csv'
    aptList = pd.read_csv(aptamerPath)

    power = dominanceScore(df, aptList)
    tocsv = pd.DataFrame(power.items())
    print(tocsv)