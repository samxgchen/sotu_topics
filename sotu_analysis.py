import gensim
from gensim import models
import os
import re
from gensim import corpora, utils
import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import *

project_dir = '/Users/Chen/projects/sotu_topics/'
data_dir = '/Users/Chen/projects/sotu_topics/data/'
models_dir = '/Users/Chen/projects/sotu_topics/models/'
files = os.listdir(data_dir)

class folder_corpus_for_dct:
    """
    Iterator over a folder where each txt file is returned
    """
    def __init__(self, data_dir, dct=None, file_suffix = ".txt"):
        self.data_dir = data_dir
        self.file_suffix = file_suffix
        self.dct = dct

    def get_files(self):
        """
        Get files with file_suffix from data_dir
        """
        files = os.listdir(self.data_dir)
        files = [file for file in files if self.file_suffix in file]
        return files

    def __iter__(self):
        """
        For each file in folder, read it in and yield it
        """
        files = self.get_files()
        for file in files:
            curr_file = open(data_dir + file, "r")
            text = curr_file.read()
            #text = text.lower()
            #text = dct.doc2bow(preprocess_string(text))  # this is for creating lda
            text = preprocess_string(text)  # this is for creating dict
            print(file)
            yield text

class folder_corpus:
    """
    Iterator over a folder where each txt file is returned
    """
    def __init__(self, data_dir, dct, file_suffix = ".txt"):
        self.data_dir = data_dir
        self.file_suffix = file_suffix
        self.dct = dct

    def get_files(self):
        """
        Get files with file_suffix from data_dir
        """
        files = os.listdir(self.data_dir)
        files = [file for file in files if self.file_suffix in file]
        return files

    def __iter_old__(self):
        """
        For each file in folder, read it in and yield it
        """
        files = self.get_files()
        for file in files:
            curr_file = open(data_dir + file, "r")
            text = curr_file.read()
            #text = text.lower()
            text = dct.doc2bow(preprocess_string(text))  # this is for creating lda
            # text = preprocess_string(text)  # this is for creating dict
            print(file)
            yield text

    def __iter__(self):
        """
        For each file in folder, read it in
        Then return each sentence in a doc
        """
        files = self.get_files()
        for file in files:
            print(file)
            curr_file = open(data_dir + file, "r")
            text = curr_file.read()
            text_sentences = text.split('.')
            # clean the list of sentences
            text_sentences = [re.sub('\n', ' ', sen) for sen in text_sentences]  # replace '/n' with ' '
            for sen in text_sentences:
                text = dct.doc2bow(preprocess_string(sen))  # this is for creating lda
                # text = preprocess_string(text)  # this is for creating dict
                yield text

sotu_corpus = folder_corpus_for_dct("/Users/Chen/projects/sotu_topics/data/")
#for speech in sotu_corpus.__iter__(): print(speech)

# build dictionary
# preprocess_string()  returns a list of tokens
dct_path = models_dir + 'sotudict.dct'
#dct = corpora.Dictionary(sotu_corpus)
#dct.save(dct_path)
dct = corpora.Dictionary.load(dct_path)

# train lda on raw data
from gensim import models
sotu_corpus = folder_corpus("/Users/Chen/projects/sotu_topics/data/", dct = dct)
#sotu_corpus = folder_corpus("/Users/Chen/projects/sotu_topics/data_small/", dct = dct)

### testing alpha and eta  ###
lda = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=2000, passes=5)
#lda2 = models.LdaModel(sotu_corpus, id2word=dct, num_topics=10, chunksize=1)
lda.show_topics(10, 3)

# diff alphas
lda2 = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=2000, passes=5, alpha=.1)
lda2.show_topics(10, 3)

lda3 = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=2000, passes=5, alpha=.01)
lda3.show_topics(10, 3)

# diff etas
lda4 = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=2000, passes=5, eta=.1)
lda4.show_topics(10, 3)

lda5 = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=2000, passes=5, eta=.01)
lda5.show_topics(10, 3)

lda5 = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=2000, passes=5, eta=.001)
lda5.show_topics(10, 3)

lda6 = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=2000, passes=5, eta=.0001)
lda6.show_topics(10, 3)

lda6 = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=10000, passes=5, eta=.00001)
lda6.show_topics(10, 3)

# diff chunk chunksize
# 2k has ~60 utiliziation in cores 2-4
# 10k got ~85% but seemed to intorduce errors-divide by 0
lda6 = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=2000, passes=5, eta=.0001)
lda6.show_topics(10, 3)

lda6 = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=10000, passes=5, eta=.00001)
lda6.show_topics(10, 3)
lda6 = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=20000, passes=5, eta=.00001)
lda6.show_topics(10, 3)
### end testing ###


### time multicore vs single thread ###
import time
t1 = time.time()
t = models.LdaModel(sotu_corpus, id2word=dct, num_topics=10, chunksize=10000, passes=1)
t2 = time.time()
t = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=10000, passes=1, workers=2)
t3 = time.time()
t = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=10000, passes=1, workers=3)
t4 = time.time()

t2-t1 # 74.7
t3-t2 # 56.2
t4-t3 # 56.1
### end ###

### train many passes on LdaMulticore ###
t5 = time.time()
lda_big = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=20, chunksize=10000, passes=30)
t6=time.time()
t6-t5 # 54.5 for 1 pass
lda_big.show_topics(20,3)
lda_big.save(models_dir + 'lda_big')

### end ###

lda = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, chunksize=2000, passes=5)
#lda2 = models.LdaModel(sotu_corpus, id2word=dct, num_topics=10, chunksize=1)
lda.show_topics(10, 3)

lda.update(sotu_corpus)  # no passes argument
lda.show_topics(10, 3)

lda = models.LdaMulticore(sotu_corpus, id2word=dct, num_topics=10, passes=20)
lda.show_topics(10, 3)

#lda.save(project_dir + "/models/sotu_lda")
#lda.save(project_dir + "/models/sotu_sen_lda")

# next steps:
# break documents into individual sentences and do topics at that level
# parse the year out of the files so can do topics over times
# add predictions over time

# predictions
# need a corpus that generates a stream based on a file
class file_corpus:
    """
    Iterator over a file and return each sentence as generator
    """
    def __init__(self, file, dct):
        self.file = file
        self.dct = dct

    def __iter__(self):
        """
        For each file in folder, read it in
        Then return each sentence in a doc
        """
        curr_file = open(self.file, "r")
        text = curr_file.read()
        text_sentences = text.split('.')
        # clean the list of sentences
        text_sentences = [re.sub('\n', ' ', sen) for sen in text_sentences]  # replace '/n' with ' '
        for sen in text_sentences:
            text = dct.doc2bow(preprocess_string(sen))  # this is for creating lda
            # text = preprocess_string(text)  # this is for creating dict
            if text != []:
                yield text

test = file_corpus(file=data_dir + "Washington_1791.txt", dct=dct)

lda.get_document_topics([(92, 1), (286, 1), (363, 1), (637, 1), (809, 1), (878, 1), (928, 2), (936, 1), (970, 1), (1024, 1)], minimum_probability=0)

def write_doc_topics(export_file, lda_model, corpus):
    with open(export_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter = '|')
        headers = ['filename', 'year'] + ['topic' + str(topic) for topic in range(0, lda_model.num_topics)]
        writer.writerow(headers)

        #for sentence in corpus:
        files = os.listdir(data_dir)
        files = [file for file in files if '.txt' in file]
        for file in files:
            curr_doc_corpus = file_corpus(file=data_dir + file, dct=dct)
            for sen in curr_doc_corpus:
                output = lda_model.get_document_topics(sen, minimum_probability=0)
                output = [file, re.findall('[0-9]+', file)[0]] + [tuple[1] for tuple in output]
                print(output)
                writer.writerow(output)

write_doc_topics(export_file = project_dir + "data/topic_assignments.csv", lda_model = lda, corpus=test)

export = pd.read_csv(project_dir + "data/topic_assignments.csv", sep='|')
! head /Users/Chen/projects/sotu_topics/data/topic_assignments.csv
! tail /Users/Chen/projects/sotu_topics/data/topic_assignments.csv

threshold = .3
year_df = export[['year']]
data_df = export.filter(regex=("topic*"))
data_df = data_df.mask(data_df < .3, 0).mask(data_df >= .3, 1)
result = pd.concat([year_df, data_df], axis=1).groupby('year').sum()
result = result.reset_index().set_index('year')
# this is how many sentences a president dedicated to each topic

result.columns = ['public_finance', 'state', 'industrial_dev', 'congressional_legislation', 'nation_peace', 'taxes', 'world', 'government_power', 'america_tonight', 'war']
print(result.to_string())
result.to_csv(project_dir + 'result.csv', index=True)

import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(data=result, dashes=False)
plt.show()

# sen lda results:
[(0,
  '0.023*"public" + 0.014*"bank" + 0.013*"govern" + 0.012*"revenu" + 0.012*"monei"'),
 (1,
  '0.058*"state" + 0.040*"unit" + 0.018*"govern" + 0.012*"countri" + 0.011*"relat"'),
 (2,
  '0.013*"program" + 0.012*"need" + 0.012*"develop" + 0.011*"product" + 0.011*"industri"'),
 (3,
  '0.056*"congress" + 0.025*"recommend" + 0.018*"report" + 0.018*"depart" + 0.015*"legisl"'),
 (4,
  '0.029*"nation" + 0.018*"peopl" + 0.015*"peac" + 0.013*"govern" + 0.011*"countri"'),
 (5,
  '0.121*"year" + 0.021*"increas" + 0.017*"tax" + 0.017*"fiscal" + 0.016*"million"'),
 (6,
  '0.024*"world" + 0.022*"american" + 0.019*"peopl" + 0.016*"america" + 0.015*"work"'),
 (7,
  '0.024*"govern" + 0.023*"state" + 0.023*"law" + 0.012*"constitut" + 0.011*"power"'),
 (8,
  '0.020*"america" + 0.016*"american" + 0.014*"tonight" + 0.011*"countri" + 0.011*"thing"'),
 (9,
  '0.017*"war" + 0.016*"servic" + 0.015*"forc" + 0.014*"offic" + 0.012*"militari"')]








# try split sentences using csv and .
import csv

with open(data_dir + "Washington_1791.txt", "r") as f:
    reader = csv.reader(f, delimiter=".")
    for i, line in enumerate(reader):
        print('line[{}] = {}'.format(i, line))

test_file = open(data_dir + "Washington_1791.txt", "r")
temp = test_file.read().splitlines()

test_file = open(data_dir + "Washington_1791.txt", "r")
temp2 = test_file.read().split('.') # this works but need to remove '\n'

test_sentences = [re.sub('\n', ' ', sen) for sen in temp2]  # replace '/n' with ' '
