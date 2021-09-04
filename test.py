import pandas as pd

df = pd.read_csv('./datasets/training/breed_1.csv')
df = df
apt1 = pd.DataFrame(columns=['Sequence1'])
apt2 = pd.DataFrame(columns=['Sequence2'])

for first in range(0,len(df)):
    x = pd.DataFrame.from_records({'Sequence1':[df.loc[first, 'Sequence']]})


    apt1 = apt1.append([x]*(len(df) - first - 1), ignore_index = True)
    
    y = pd.DataFrame({'Sequence2':df.loc[first+1:,'Sequence']})
    apt2 = apt2.append(y, ignore_index=True)
    

dataset = apt1.join(apt2)
print(dataset)
dataset.to_csv('./images/test.csv')
#print(dataset)