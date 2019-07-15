import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_books = pd.read_csv('Dataset_2/books.csv')
df_ratings = pd.read_csv('Dataset_2/ratings.csv')

def kriteria(i):
    return str(i['authors'])+' '+str(i['original_title'])+' '+str(i['title'])+' '+str(i['language_code'])
df_books['Kriteria'] = df_books.apply(kriteria,axis='columns')
# print(df_books.head())

# count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(tokenizer=lambda x:x.split(' '))
matrixFeature = model.fit_transform(df_books['Kriteria'])

Kriteria = model.get_feature_names()
jml_fitur = len(Kriteria)

# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
skor = cosine_similarity(matrixFeature)

# constraint just input book ratings with 3 stars higher
andi_1 = df_books[df_books['original_title']=='The Hunger Games']['book_id'].tolist()[0]-1 
andi_2 = df_books[df_books['original_title']=='Catching Fire']['book_id'].tolist()[0]-1 
andi_3 = df_books[df_books['original_title']=='Mockingjay']['book_id'].tolist()[0]-1 
andi_4 = df_books[df_books['original_title']=='The Hobbit or There and Back Again']['book_id'].tolist()[0]-1 
suka_andi = [andi_1,andi_2,andi_3,andi_4]

budi_1 = df_books[df_books['original_title']=='Harry Potter and the Philosopher\'s Stone']['book_id'].tolist()[0]-1 
budi_2 = df_books[df_books['original_title']=='Harry Potter and the Chamber of Secrets']['book_id'].tolist()[0]-1 
budi_3 = df_books[df_books['original_title']=='Harry Potter and the Prisoner of Azkaban']['book_id'].tolist()[0]-1 
suka_budi = [budi_1,budi_2,budi_3]

ciko_1 = df_books[df_books['original_title']=='Robots and Empire']['book_id'].tolist()[0]-1 
suka_ciko = [ciko_1]

dedi_1 = df_books[df_books['original_title']=='Nine Parts of Desire: The Hidden World of Islamic Women']['book_id'].tolist()[0]-1 
dedi_2 = df_books[df_books['original_title']=='A History of God: The 4,000-Year Quest of Judaism, Christianity, and Islam']['book_id'].tolist()[0]-1 
dedi_3 = df_books[df_books['original_title']=='No god but God: The Origins, Evolution, and Future of Islam']['book_id'].tolist()[0]-1 
suka_dedi = [dedi_1,dedi_2,dedi_3]

ello_1 = df_books[df_books['original_title']=='Doctor Sleep']['book_id'].tolist()[0]-1 
ello_2 = df_books[df_books['original_title']=='The Story of Doctor Dolittle']['book_id'].tolist()[0]-1 
ello_3 = df_books[df_books['title']=='Bridget Jones\'s Diary (Bridget Jones, #1)']['book_id'].tolist()[0]-1 
suka_ello = [ello_1,ello_2,ello_3]

list_skor_andi_1 = list(enumerate(skor[andi_1]))
list_skor_andi_2 = list(enumerate(skor[andi_2]))
list_skor_andi_3 = list(enumerate(skor[andi_3]))
list_skor_andi_4 = list(enumerate(skor[andi_4]))

list_skor_budi_1 = list(enumerate(skor[budi_1]))
list_skor_budi_2 = list(enumerate(skor[budi_2]))
list_skor_budi_3 = list(enumerate(skor[budi_3]))

list_skor_ciko = list(enumerate(skor[ciko_1]))

list_skor_dedi_1 = list(enumerate(skor[dedi_1]))
list_skor_dedi_2 = list(enumerate(skor[dedi_2]))
list_skor_dedi_3 = list(enumerate(skor[dedi_3]))

list_skor_ello_1 = list(enumerate(skor[ello_1]))
list_skor_ello_2 = list(enumerate(skor[ello_2]))
list_skor_ello_3 = list(enumerate(skor[ello_3]))

list_skor_andi = []
for i in list_skor_andi_1:
    list_skor_andi.append((i[0],0.25*(list_skor_andi_1[i[0]][1]+list_skor_andi_2[i[0]][1]+list_skor_andi_3[i[0]][1]+list_skor_andi_4[i[0]][1])))
list_skor_budi = []
for i in list_skor_andi_1:
    list_skor_budi.append((i[0],(list_skor_budi_1[i[0]][1]+list_skor_budi_2[i[0]][1]+list_skor_budi_3[i[0]][1])/3))
list_skor_dedi = []
for i in list_skor_andi_1:
    list_skor_dedi.append((i[0],(list_skor_dedi_1[i[0]][1]+list_skor_dedi_2[i[0]][1]+list_skor_dedi_3[i[0]][1])/3))
list_skor_ello = []
for i in list_skor_andi_1:
    list_skor_ello.append((i[0],(list_skor_ello_1[i[0]][1]+list_skor_ello_2[i[0]][1]+list_skor_ello_3[i[0]][1])/3))

sort_andi = sorted(
    list_skor_andi,
    key=lambda j:j[1],
    reverse=True
)
sort_budi = sorted(
    list_skor_budi,
    key = lambda j:j[1],
    reverse = True
)
sort_ciko = sorted(
    list_skor_ciko,
    key = lambda j:j[1],
    reverse = True
)
sort_dedi = sorted(
    list_skor_dedi,
    key = lambda j:j[1],
    reverse = True
)
sort_ello = sorted(
    list_skor_ello,
    key = lambda j:j[1],
    reverse = True
)

# top 5 recommendation
sama_andi = []
for i in sort_andi:
    if i[1]>0:
        sama_andi.append(i)
sama_budi = []
for i in sort_budi:
    if i[1]>0:
        sama_budi.append(i)
sama_ciko = []
for i in sort_ciko:
    if i[1]>0:
        sama_ciko.append(i)
sama_dedi = []
for i in sort_dedi:
    if i[1]>0:
        sama_dedi.append(i)
sama_ello = []
for i in sort_ello:
    if i[1]>0:
        sama_ello.append(i)

print('1. Buku bagus untuk Andi:')
for i in range(0,5):
    if sama_andi[i][0] not in suka_andi:
        print('-',df_books['original_title'].iloc[sama_andi[i][0]])
    else:
        i+=5
        print('-',df_books['original_title'].iloc[sama_andi[i][0]])

print(' ')
print('2. Buku bagus untuk Budi:')
for i in range(0,5):
    if sama_budi[i][0] not in suka_budi:
        print('-',df_books['original_title'].iloc[sama_budi[i][0]])
    else:
        i+=5
        print('-',df_books['original_title'].iloc[sama_budi[i][0]])

print(' ')
print('3. Buku bagus untuk Ciko:')
for i in range(0,5):
    if sama_ciko[i][0] not in suka_ciko:
        print('-',df_books['original_title'].iloc[sama_ciko[i][0]])
    else:
        i+=5
        print('-',df_books['original_title'].iloc[sama_ciko[i][0]])

print(' ')
print('4. Buku bagus untuk Dedi:')
for i in range(0,5):
    if sama_dedi[i][0] not in suka_dedi:
        print('-',df_books['original_title'].iloc[sama_dedi[i][0]])
    else:
        i+=5
        print('-',df_books['original_title'].iloc[sama_dedi[i][0]])

print(' ')
print('5. Buku bagus untuk Ello:')
for i in range(0,5):
    if sama_ello[i][0] not in suka_ello:
        if str(df_books['original_title'].iloc[sama_ello[i][0]])=='nan':
            print('-',df_books['title'].iloc[sama_ello[i][0]])
        else:
            print('-',df_books['original_title'].iloc[sama_ello[i][0]])  
    else:
        i+=5
        if str(df_books['original_title'].iloc[sama_ello[i][0]])=='nan':
            print('-',df_books['title'].iloc[sama_ello[i][0]])
        else:
            print('-',df_books['original_title'].iloc[sama_ello[i][0]])  