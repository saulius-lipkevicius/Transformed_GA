from itertools import combinations_with_replacement

alphabets = ['A', 'C', 'G', 'T']
L3 = ''
L4 = ''

for (a,b,c) in combinations_with_replacement(alphabets, 3):
    L3 = L3+a+b+c+' '

for (a,b,c,d) in combinations_with_replacement(alphabets, 4):
    L4 = L4+a+b+c+d+' '

dataset = {}
dataset[0] = { 'id': 1,
    'text': "A G C T"
}
dataset[1] = {'id': 2,
    'text': "AA AG AC AT GA GG GC GT CA CG CC CT TA TG TC TT"
    }
dataset[2] = {'id': 3,
    'text': L3
    }
dataset[3] = {'id': 4,
    'text': L4
    }

text_data = []
for i in range(1,4):
  text_data.append(dataset[i]['text'])
with open(f'1.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))