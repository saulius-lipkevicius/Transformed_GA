import pandas as pd #reading csv


#Read csv sheet, outputs already the best
def readCSV(readfile):
    df = pd.read_csv(readfile)
    df.sort_values(by=['Fitness'], inplace=True, ascending=False, ignore_index=True)
    df = pd.DataFrame(df)
    df = df.iloc[:int(len(df)*0.025),:] #testui

    return df


#Label every data row
def labelClass(df):
    listLength = len(df)
    dataframe = pd.DataFrame(columns=['Aptamer1', 'Aptamer2', 'Label'])
    for aptamer1 in range(0, listLength):
        for aptamer2 in range(0, listLength):
            if aptamer1 != aptamer2:
                if df.iloc[1,aptamer1] >= df.iloc[1, aptamer2]:
                    label = 1
                else:
                    label = 0

                newRow = {'Aptamer1':  df.iloc[1,aptamer1], 'Aptamer2': df.iloc[1, aptamer2] , 'Label': label}


    return dataframe


df = readCSV('AptamersList.csv')

print(df.head())
print(len(df))

df2 = labelClass(df)
print(df2)
