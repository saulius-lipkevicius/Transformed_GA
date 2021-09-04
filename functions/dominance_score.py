import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Calculates top N'%' aptamers")
parser.add_argument("--p", "--path"
                        , help="Path to paired aptamers"
                        , type=str)
parser.add_argument("--f", "--full_apt"
                        , help="Full aptamer list"
                        , type=str)
parser.add_argument("--o", "--output_location"
                        , help="Location of new breed"
                        , type=str)
parser.add_argument("--i", "--iter"
                        , help="What iteration it is"
                        , type=int)
parser.add_argument("--l", "--apt_len"
                        , help="How many aptamers we have in total"
                        , type=int)
args = parser.parse_args()


def dominanceScore(dataset):
    dataset = pd.read_csv(args.p)
    power = {}
    initialAptamers = pd.read_csv(args.f)

    #initial aptamers, pataisyti kad imtu is kitos funkcijos
    for i in range(0,args.l):
        power.update({initialAptamers.loc[i,'Sequence']: 0})

    #dont have to normalize the score
    for t in range(0, len(dataset)):
        if dataset.loc[t,'Label'] == 1:
            power[dataset.loc[t,'Sequence1']] += 1/args.l
        else:
            power[dataset.loc[t,'Sequence2']] += 1/args.l

    n = int(len(power) * 0.10)  # floor float result, as you must use an integer
    power = sorted(power.items(), key=lambda x:x[1], reverse=True)[:n]

    return power

def main():
    power = dominanceScore()

    preprocessedToGA = pd.DataFrame(power)
    preprocessedToGA.columns = ['Sequence', 'Power'] 
    preprocessedToGA['Power'] = preprocessedToGA['Power'].round(decimals=3)
    print("Best aptamers have been chosen and located to {}".format(args.o))
    preprocessedToGA.to_csv('{}'.format(args.o, args.i), encoding='utf-8', index=False)

if __name__=="__main__":
    main()

