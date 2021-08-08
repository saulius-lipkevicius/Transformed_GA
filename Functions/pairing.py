import pandas as pd
import numpy as np
from sklearn.utils import shuffle


parser = argparse.ArgumentParser(description='Breeder acquires new aptamers...')
parser.add_argument("--p", "--path_initial_aptamers"
                        , help="Path to fittest aptamers CSV"
                        , type=str)
parser.add_argument("--l", "--labeled"
                        , help="Is data labeled"
                        , action="store_true")
args = parser.parse_args()


def pairLabel(df):
    apt1 = pd.DataFrame(columns=['Sequence1', 'Entropy1'])
    apt2 = pd.DataFrame(columns=['Sequence2', 'Entropy2'])

    for first in range(0,len(df)):
        x = pd.DataFrame({'Sequence1':df.loc[first,'Sequence'],'Entropy1':df.loc[first,'Entropy']},index = range(1))
        apt1 = apt1.append([x]*(len(df) - first - 1), ignore_index=True)
            
    for second in range(1,len(df) + 1):
        y = pd.DataFrame({'Sequence2':df.loc[second:,'Sequence'],'Entropy2':df.loc[second:,'Entropy']})
        apt2 = apt2.append(y, ignore_index=True)

    dataset = apt1.join(apt2)
    dataset["Label"] = np.where(dataset.eval("Entropy1 >= Entropy2"), 1, 0)
    dataset = shuffle(dataset)

    return dataset[['Sequence1', 'Sequence2', 'Label']]


def balanceData(dataset):
    counts = dataset['Label'].value_counts()

    if counts.loc[0] >= counts.loc[1]:
        change = int((counts.loc[0] - counts.loc[1])/2)
        value = 0
    else:
        change = int((counts.loc[1] - counts.loc[0])/2)
        value = 1

    zeros = dataset[(dataset['Label'] == value)]
    zerosIndexes = zeros.sample(change).index
    zeros.loc[zerosIndexes,'Label'] = int(abs(value - 1))
    
    dataset.update(zeros)
    dataset['Label'] = dataset['Label'].astype(int)

    return dataset


def main():
    path = './datasets/'
    dataPath = './datasets/scored_sequences.csv'
    df = pd.read_csv(dataPath)
    data = pairLabel(df)
    dataset = balanceData(data)
    
    train, test, val = np.split(data, [int(.8*len(data)), int(.9*len(data))]) #80% training, 10% validating, 10% testing

    print("Migrating datasets to CSV to {}".format(path))
    data.to_csv('./datasets/full_comparison.csv', encoding='utf-8', index=False)
    train.to_csv('./datasets/train.csv', encoding='utf-8', index=False)
    test.to_csv('./datasets/test.csv', encoding='utf-8', index=False)
    val.to_csv('./datasets/val.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    main()