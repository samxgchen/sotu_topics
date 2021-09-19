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

import logging
logging.basicConfig(filename=project_dir + 'gensim.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logging.setLevel(logging.DEBUG)

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

# choose k topics
k_topics = {'m10': 10, 'm20':20, 'm30':30, 'm40':40, 'm50':50}
k_topics = {'m10': 10, 'm30':30, 'm50':50}
k_topics = {'m30':30, 'm50':50}
model_storage = {}
coherence_storage = {}
for model in k_topics:
    k = k_topics[model]
    lda_model = models.LdaMulticore(sotu_corpus, id2word=dct,
                    num_topics=k_topics[model], chunksize=10000,
                    passes=1, workers=3)
    model_storage[model] = lda_model
    lda_model.save(models_dir + 'lda_' + str(k) + 'topics.model')
    cm = models.CoherenceModel(model=lda_model, corpus=sotu_corpus,
                    dictionary=dct, coherence='u_mass')
    coherence = cm.get_coherence()
    coherence_storage[model] = coherence
coherence_storage
{'m10': -3.0643356890368829,
 'm30': -3.0682504720634309,
 'm50': -3.0878282871620168}

# train model a little longer

# run predictions



# generate output dataset with metadata (year and president)

# get file metadata
files = os.listdir(data_dir)
# split file name into sections and turn into df
metadata = pd.DataFrame([no_header.split('_') for no_header in [file.split('.')[0] for file in files]])
metadata.columns = ['president', 'year']
metadata['year'] = pd.to_numeric(metadata.year)

# cluster data
# this is at sentence level
export = pd.read_csv(project_dir + "data/topic_assignments.csv", sep='|')
export.head()
# avg sentences to doc-level
avg_df = export.groupby('year').mean()
from sklearn.cluster import KMeans
km = pd.DataFrame(KMeans(n_clusters=10, random_state=0).fit_predict(avg_df))

# join against metadata
result = pd.DataFrame(avg_df.index)
result['year'] = pd.to_numeric(result.year)
result = result.merge(metadata, on='year', how='left')
result = pd.concat([result, km], axis=1)
result.columns = ['year', 'president', 'cluster']
print(result.to_string())

# visualize
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(avg_df)
X_embedded.shape
