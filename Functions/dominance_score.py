import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Calculates top N'%' aptamers")
parser.add_argument("--p", "--path"
                        , help="Path to paired aptamers"
                        , type=str)
parser.add_argument("--l", "--output_location"
                        , help="Location of new breed"
                        , type=str)
args = parser.parse_args()


def dominanceScore(dataset, initialAptamers):
    power = {}

    #initial aptamers, pataisyti kad imtu is kitos funkcijos
    for i in range(0,len(initialAptamers)):
        power.update({initialAptamers.loc[i,'Sequence']: 0})

    #dont have to normalize the score
    for t in range(0, len(dataset)):
        if dataset.loc[t,'Label'] == 1:
            power[dataset.loc[t,'Sequence1']] += 1/len(initialAptamers)
        else:
            power[dataset.loc[t,'Sequence2']] += 1/len(initialAptamers)

    n = int(len(power) * 0.10)  # floor float result, as you must use an integer
    power = sorted(power.items(), key=lambda x:x[1], reverse=True)[:n]

    return power

def main():
    dataPath = './datasets/full_comparison.csv'
    df = pd.read_csv(dataPath)

    aptamerPath = './datasets/scored_sequences.csv'
    aptList = pd.read_csv(aptamerPath)

    power = dominanceScore(df, aptList)

    preprocessedToGA = pd.DataFrame(power)
    preprocessedToGA.columns = ['Sequence', 'Power'] 
    preprocessedToGA['Power'] = preprocessedToGA['Power'].round(decimals=3)

    try:
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    #prideti saving excelio

    location = './datasets/GA_iterations/top100_{}'.format(1)
    preprocessedToGA.to_csv(location, encoding='utf-8', index=False)

if __name__=="__main__":
    main()

