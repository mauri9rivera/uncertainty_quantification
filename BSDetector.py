import re
import numpy as np
import json  
import os
import pandas as pd
import random
import seaborn as sns
from openai import OpenAI
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import scripts.other_experiments.reliability_diagrams as reldiag

from betacal import BetaCalibration
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, brier_score_loss, jaccard_score
from collections import Counter
import itertools

from math import sqrt, pow, exp
import spacy
from sentence_transformers import SentenceTransformer, util
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer




# ----------- Trimmer class ------------------------ #


class Trimmer():
    '''
    Patterns log:

        - pattern = r'\[.*\]' ------> Gets everything between squared brackets
        - pattern - r'\|.*\|' ------> Gets everything between vertical brackets
        - pattern = [0-5] ------> Gets number between 0-5 (for politifact scale)
        - pattern = \([A-Z]\) -----> Gets any capital letter between parentheses (for decomposition)
        - pattern = r'\b(0|[1-9][0-9]?|100)\b' ---> Gets any 0-100 number after vertical bracket (self.separator has to be '|')
    '''

    def __init__(self, pattern, separator):
        self.pattern = pattern
        self.separator = separator

    
    def __str__(self):
        print(f'The separator used by this Trimmer is: {self.separator} . \n')

    
    def get_split(self, answer):

        try:
            if self.separator == '|':
                tmp = answer.split('|')[1]
                match = re.search(self.pattern, tmp).group(0)
                return int(match)

            tmp = answer.split(self.separator)[0]
            match = re.search(self.pattern, tmp).group(0)
            return int(match.strip(self.separator))
        except: 
            return random.randint(0, 100) #return 'notnumber'
        
    def direct_split(self, answer):

        try:
            return int(answer)
        except:
            return random.randint(0, 100)

    def get_decompositions(self, answer):

        p = re.compile(self.pattern)
        try:
            segments = re.split(p, answer)
            segments = [segment.strip() for segment in segments if segment]
            segments = segments[1:]

            l = []
            for segment in segments:

                question = segment.split('?')[0]
                analysis = segment.split('?')[1]

                match = re.search(r'\[.*\]', analysis).group(0)
                
                try:
                    score = int(match.strip('[]'))
                
                except:
                    score = 50

                #print(f'This is question: {question} and the score: {score}')

                l.append((question, score))

            return l

        except:
            return []
        


# ---------------- Scale converter functions ------------------- #


def to_binary(label, threshold):
    try:
        return 1 if label >= threshold else 0
    except:
        return random.randint(0, 1)

def to_pants_on_fire(label):

    if label == 'notnumber':
        return random.randint(0,5)
    elif label < 16:
        return 0
    elif 16 < label < 33:
        return 1
    elif 33 < label < 50:
        return 2
    elif 50 < label < 67:
        return 3
    elif 67 < label < 83:
        return 4
    elif 83 < label:
        return 5
    else:
        return random.randint(0,5)

def to_number(input_str):
    try:
        input_str = float(input_str)
        return int(input_str)
    except:
        return random.randint(0, 100)

def to_terciary(label):    
    if label == 0 or label == 1:
        return 0
    elif label == 2 or label == 3:
        return 0.5
    else:
        return 1
    
def from_pants_on_fire(label):

    if label == 'pants-fire':
        return 0
    elif label == 'false':
        return 0
    elif label == 'half-true':
        return 0
    elif label == 'barely-true':
        return 1
    elif label == 'mostly-true':
        return 1
    elif label == 'true':
        return 1

# -------------- Normalization functions -------------------- #


def id_scaling(df_col):
    return df_col

def min_max_scale(df_col):

    max_val = max(df_col.values)
    min_val = min(df_col.values)
    df_col = (df_col - min_val) / (max_val - min_val)

    return df_col

def z_scale(df_col):

    u = np.mean(df_col.values)
    sigma = np.std(df_col.values)
    df_col = (df_col - u) / sigma

    from scipy.stats import norm

    def zscore_to_percentile(z_score):
        percentile_score = norm.cdf(z_score)
        return percentile_score

    df_col = df_col.apply(zscore_to_percentile)
    return df_col

def percentalize(df_col):
    percentiles = np.linspace(0, 100, 101)
    percentile_values = np.percentile(df_col.values, percentiles)
    bin_indices = np.searchsorted(percentile_values, df_col.values, side='right')
    df_col = (bin_indices - 1) / (len(percentile_values) - 1)

    return df_col


# ------- Observed consistency methods -------------------- #


def self_consistency(df, scaling=id_scaling):
    
    df['consistency'] = None

    for index, row in df.iterrows():
        scores_list = df.loc[index, 'observed-answers']
        # Count the occurrences of each element in the list
        element_counts = Counter(scores_list)
        counter_list = list(element_counts.items())

        # Get the first key-value pairs
        first_element = counter_list[0]

        first_key, first_value = first_element

        df.loc[index, 'self-consistency'] = first_value

    df['consistency'] /= len(scores_list)

    return scaling(df['consistency'])

def selfcheckGPT(df, scaling=id_scaling):
    
    df['consistency'] = df['observed-answers'].apply(lambda x: sum(
        [1 if to_binary(item, 60) == to_binary(df.at[index, 'reference-answer'], 60) else 0 for index, item in
         enumerate(x)]) / len(x))

    return scaling(df['consistency'])

def prediction_class_margin(df, scaling=id_scaling):

    df['consistency'] = None

    for index, row in df.iterrows():
        scores_list = df.loc[index, 'observed-answers']
        # Count the occurrences of each element in the list
        element_counts = Counter(scores_list)
        counter_list = list(element_counts.items())

        # Get the first and last key-value pairs
        first_element = counter_list[0]
        last_element = counter_list[-1]

        first_key, first_value = first_element
        last_key, last_value = last_element

        df.loc[index, 'consistency'] = abs(first_value - last_value) / len(scores_list)


    return scaling(df['consistency'])

def sample_avg_dev(df, scaling=id_scaling):

    df['consistency'] = df['observed-answers'].apply(lambda x: abs(sum(x) - 50) / len(x))

    return scaling(df['consistency'])

def std_consistency(df, scaling=id_scaling):

    df['consistency'] = df['observed-answers'].apply(lambda x: np.std(x))

    return scaling(df['consistency'])

def deviation_sum(df, scaling=id_scaling):

    df['mean'] = df['observed-answers'].apply(lambda x: np.mean(x))

    def calculate_deviation_sum(row):
        return np.sum(np.abs(np.array(row['observed-answers']) - row['mean']))

    df['consistency'] = df.apply(calculate_deviation_sum, axis=1)

    return scaling(df['consistency'])

    
# --------- BSDetector Framework ----------------------- #

def calculate_ece(y_true, y_pred, confidences):

    pos_label_confidence = []
    for prediction, confidence in zip(y_pred, confidences):
        if prediction == 0:
            pos_label_confidence.append(1 - confidence)
        else:
            pos_label_confidence.append(confidence)

    return calibration_curve(y_true, pos_label_confidence, n_bins=10, pos_label=1, strategy='quantile')    



class BSDetector():

    def __init__(self, ref_file, verb_file, observed_folder, consistency_method, scaling, trimmer, test=False):
        self.ref_file = ref_file
        self.verb_file = verb_file
        self.observed_folder = observed_folder
        self.consistency_method = consistency_method
        self.test = test
        self.scaling = scaling
        self.df = None
        self.best_alpha = None
        self.trimmer = trimmer

    def get_df(self):

        base_df = pd.read_json(self.ref_file, lines=True, orient='records')
        base_df.rename(columns={'gpt4-answer': 'reference-answer'}, inplace=True)
        verbalized_df = pd.read_json(self.verb_file, lines=True, orient='records')
        verbalized_df.rename(columns={'gpt-answer': 'verbalized-answer'}, inplace=True)
        if self.test:
            verbalized_df.rename(columns={'gpt-answer_y': 'verbalized-answer'}, inplace=True)

        base_df = pd.merge(base_df, verbalized_df[['id', 'verbalized-answer']], on='id', how='left')


        def add_sample_answer(row):
            return row['observed-answers'].append(self.trimmer.get_split(row['sample-answer']))
        
        base_df['observed-answers'] = base_df['reference-answer'].apply(lambda x: [self.trimmer.get_split(x)])
        
        for filename in os.listdir(self.observed_folder):
            file_path = os.path.join(self.observed_folder, filename)

            curr_df = pd.read_json(file_path, lines=True, orient='records')
            curr_df.rename(columns={'gpt4-answer': 'sample-answer'}, inplace=True)
            if self.test:
                curr_df.rename(columns={'gpt-answer_y': 'sample-answer'}, inplace=True)
            base_df = pd.merge(base_df, curr_df[['id', 'sample-answer']], on='id', how='left')
            base_df.apply(add_sample_answer, axis=1)
            base_df = base_df.drop('sample-answer', axis=1)
        
        base_df['consistency-score'] = self.consistency_method(base_df, self.scaling)
        base_df['verbalized-answer'] = base_df['verbalized-answer'].apply(lambda x: self.trimmer.direct_split(x))
        base_df['reference-answer'] = base_df['reference-answer'].apply(lambda x: self.trimmer.get_split(x))
        base_df.drop(columns=['subject', 'speaker', 'job_title', 'state_info', 'party_affiliation',
                              'barely_true_counts', 'false_counts', 'half_true_counts',
                              'mostly_true_counts', 'pants_onfire_counts', 'max_tokens', 
                              'prompt_tokens', 'completion_tokens'], axis=1)
        self.df = base_df

    def set_alpha(self, k):
        
        X1 = self.df['consistency-score'].values
        X2 = self.df['verbalized-answer'].values
        X = np.vstack([X1, X2])
        y = self.df['label'].apply(lambda x: to_binary(x, 3)).values

        def kfold_indices(data, k):
            fold_size = data.shape[1] // k
            shuffled_indices = np.arange(data.shape[1])
            np.random.shuffle(shuffled_indices)
            folds = [shuffled_indices[i * fold_size:(i + 1) * fold_size] for i in range(k)]

            fold_combinations = []
            indices = list(range(0, k))
            for comb in itertools.combinations(indices, k-1):
                test_indices = folds[list(set(indices) - set(comb))[0]]
                train_indices = list(itertools.chain(*[folds[i] for i in comb]))
                fold_combinations.append((train_indices, test_indices))
                
            return fold_combinations


        fold_indices = kfold_indices(X, k)
        beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_beta = 0.0
        best_score = 100

        for beta_candidate in beta_values:

            scores = []

            for train_indices, test_indices in fold_indices:

                X_train, y_train = X[:, train_indices], y[train_indices]

                hybrid_scores = hybrid_scores = X_train[0,] * beta_candidate + X_train[1,] * (1 - beta_candidate) / 100

                reliability = calibration_curve(y_train, list(hybrid_scores), n_bins=10, pos_label=1,
                                        strategy='quantile')

                ece_score = np.mean(np.abs(reliability[0] - reliability[1]))

                scores.append(ece_score)

            if np.mean(scores) < best_score:
                best_score = np.mean(scores)
                best_beta = beta_candidate

        #X_test, y_test = X[:, fold_indices[1:]], y[fold_indices[1:]]

        hybrid_scores = X1 * best_beta + X2 * (1 - best_beta) / 100
        reliability = calibration_curve(list(y), list(hybrid_scores), n_bins=10, pos_label=1,
                                        strategy='quantile')
        ece_score = np.mean(np.abs(reliability[0] - reliability[1]))
        print(f'This is the ece_score: {round(ece_score, 4)} for beta {best_beta}')

        self.df['overall-confidence'] = self.df['consistency-score'] * best_beta + (1 - best_beta) * self.df['verbalized-answer']/100
        self.df['overall-confidence'] = pd.to_numeric(self.df['overall-confidence'], errors='coerce').round(decimals=4)

        self.best_alpha = best_beta

    def semantic_analysis(self, embeddor):
        
        base_df = pd.read_json(self.ref_file, lines=True, orient='records')
        base_df.rename(columns={'gpt4-answer': 'reference-answer'}, inplace=True)
        
        def add_sample_answer(row):
                row['observed-answers'].append(row['sample-answer'])
                return row

        base_df['observed-answers'] = base_df['reference-answer'].apply(lambda x: [x])
        for filename in os.listdir(self.observed_folder):
            file_path = os.path.join(self.observed_folder, filename)

            curr_df = pd.read_json(file_path, lines=True, orient='records')
            curr_df.rename(columns={'gpt4-answer': 'sample-answer'}, inplace=True)
            if self.test:
                curr_df.rename(columns={'gpt-answer_y': 'sample-answer'}, inplace=True)
            base_df = pd.merge(base_df, curr_df[['id', 'sample-answer']], on='id', how='left')
            base_df.apply(add_sample_answer, axis=1)
            base_df = base_df.drop('sample-answer', axis=1)

        base_df['semantic-analysis'] = None
        for index, row in base_df.iterrows():

            analysis_list = row['observed-answers']

            sim_scores = []
            for i, s1 in enumerate(analysis_list):
                for j, s2 in enumerate(analysis_list):
                    if i != j and isinstance(s1, str) and isinstance(s2, str):
                        sim_scores.append(semantic_sim([s1], [s2], embeddor))

            base_df.loc[index, 'semantic-analysis'] = np.mean(sim_scores)

        self.df['semantic-analysis'] = base_df['semantic-analysis']

    def show_calib(self):

        true_labels = self.df['label'].apply(lambda x: to_binary(x, 3)).values
        confidences = self.df['overall-confidence'].values
        pred_labels = self.df['reference-answer'].apply(lambda x: to_binary(x, 60)).values

        reliability = calibration_curve(true_labels, confidences, n_bins=10, pos_label=1, strategy='quantile')
        plt.xlabel('BSDetector Predicted Confidence', size=12)
        plt.ylabel('Actual Confidence', size=12)
        line1, = plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2)
        line2, = plt.plot(reliability[1], reliability[0], linewidth=2)
        legend = plt.legend(handles=[line1, line2], labels=['Perfect Prediction', 'BSDetector Prediction'], fontsize='x-large')
        plt.show()
        print('Expected Calibration Error (ECE)', np.mean(np.abs(reliability[0] - reliability[1])))
        print(f'The brier score: {brier_score_loss(true_labels, pred_labels, sample_weight=confidences)}')

    def show_rel_dig(self):

        uncertainty_filtered = np.array(self.df['overall-confidence'].values)
        estimated_labels = np.array(self.df['reference-answer'].apply(lambda x: to_binary(x, 60)).values)
        true_labels = np.array(self.df['label'].apply(lambda x: to_binary(x, 3)).values)
        fig = reldiag.reliability_diagram(true_labels, estimated_labels, uncertainty_filtered, return_fig=True)


# ---------- Calibration methods ----------------------- #
       
        
class Calibrator():

    def __init__(self, type='default'):
        self.classes_ = [0,1]
        self.type = type
        if type == 'default':
            self.regressor = CalibratedClassifierCV(self, cv='prefit', method='sigmoid')
        elif type == 'beta':
            self.regressor = BetaCalibration(parameters="abm")
        elif type== "platt":
            self.regressor = LogisticRegression()
        elif type == "isotonic":
            self.regressor = IsotonicRegression(y_min=0.0, y_max = 1.0)

    def predict_proba(self, X):
            print(np.vstack([X, 1 - X]).T.shape)
            return np.vstack([X, 1 - X]).T

    def fit(self, X_probs):
        return self
    
    def calibrating(self, bsdetector):

        gpt4_val_probs = np.array(bsdetector.df['overall-confidence'].values)
        y_val_true_binary = np.array(bsdetector.df['label'].apply(lambda x: to_binary(x, 3)).values)

        if self.type == 'platt':
            self.regressor.fit(gpt4_val_probs.reshape(-1, 1), y_val_true_binary)
            calibrated_probs_platt = self.regressor.predict_proba(gpt4_val_probs.reshape(-1, 1))[:, 1]
        elif self.type == 'isotonic':
            calibrated_probs_iso = self.regressor.fit_transform(gpt4_val_probs, y_val_true_binary)
        else:
            self.regressor.fit(gpt4_val_probs, y_val_true_binary)


# --------- Similarity Measure methods ------------------------- #

def semantic_sim(list1, list2, embeddor):

    similarity_scores = embeddor(list1, list2)

    final_score = []
    for score in similarity_scores:
        final_score.append(max(score))

    return np.mean(final_score)

def semantic_sim2(list1, list2, embeddor):

    similarity_scores = embeddor(list1, list2)

    #max_idx_val = [(i, max(sublist)) for i,sublist in enumerate(similarity_scores)]

    final_scores = {}
    for scores in similarity_scores:

        match = scores.tolist().index(max(scores))

        if match not in final_scores:
            final_scores[match] = [max(scores)]
        else:
            final_scores[match].append(max(scores))

    for k, v in final_scores.items():

        avg = sum(v) / len(v)
        final_scores[k] = avg

    return sum(final_scores.values()) / len(final_scores)

def transformer_embed(list1, list2):

    '''
    documents = list1 + list2
    doc1 = " ".join(list1)
    doc2 = " ".join(list2)

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    #embeddings = model.encode(documents, convert_to_tensor=True)

    embedding1 = model.encode(doc1)
    embedding2 = model.encode(doc2)
    # Extract embeddings for sentences in list1 and list2
    #embeddings_list1 = embeddings[:len(list1)]
    #embeddings_list2 = embeddings[len(list1):]

    # Calculate cosine similarity using sklearn's cosine_similarity
    #similarity_matrix = cosine_similarity(embeddings_list1, embeddings_list2)
    similarity_matrix = util.pytorch_cos_sim(embedding1, embedding2)

    return similarity_matrix
    '''
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings1 = model.encode(list1, convert_to_tensor=True)
    embeddings2 = model.encode(list2, convert_to_tensor=True)
    
    return util.cos_sim(embeddings1, embeddings2)

def word2vec_embed(list1, list2):

    # Preprocess sentences (lowercase and split into words)
    
    tokenized_list1 = [word_tokenize(sentence.lower()) for sentence in list1]
    tokenized_list2 = [word_tokenize(sentence.lower()) for sentence in list2]

    #tokenized_list1 = [sentence.lower().split() for sentence in list1]
    #tokenized_list2 = [sentence.lower().split() for sentence in list2]

    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    for index, token_sentence in enumerate(tokenized_list1):
        filtered = [lemmatizer.lemmatize(token) for token in token_sentence if token not in stop_words]
        tokenized_list1[index] = filtered
    for index, token_sentence in enumerate(tokenized_list2):
        filtered = [lemmatizer.lemmatize(token) for token in token_sentence if token not in stop_words]
        tokenized_list2[index] = filtered

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=tokenized_list1 + tokenized_list2, vector_size=100, window=5, min_count=1, workers=4)

    # Function to calculate average Word2Vec vector for a sentence
    def average_word2vec(sentence):
        vectors = [word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec_model.vector_size)

    # Get average Word2Vec vectors for sentences in list1 and list2
    vectors_list1 = [average_word2vec(sentence) for sentence in tokenized_list1]
    vectors_list2 = [average_word2vec(sentence) for sentence in tokenized_list2]

    # Calculate cosine similarity for all pairwise combinations
    similarity_scores = cosine_similarity(vectors_list1, vectors_list2)

    return similarity_scores    

def tfid_embed(list1, list2):

    vectorizer = TfidfVectorizer(stop_words='english')
    vector1 = vectorizer.fit_transform(list1)
    vector2 = vectorizer.transform(list2)

    return cosine_similarity(vector1, vector2)

def jaccard(list1, list2):
    tokenized_list1 = [set(sentence.split()) for sentence in list1]
    tokenized_list2 = [set(sentence.split()) for sentence in list2]

    # Calculate Jaccard similarity index for each pair of sentences
    jaccard_similarities = []
    for set1 in tokenized_list1:
        for set2 in tokenized_list2:
            jaccard_similarities.append(jaccard_score(set1, set2))

    # Reshape the jaccard_similarities into a matrix
    num_sentences_list1 = len(tokenized_list1)
    num_sentences_list2 = len(tokenized_list2)
    similarity_scores = np.array(jaccard_similarities).reshape((num_sentences_list1, num_sentences_list2))

    final_score = []
    for cos_sim in similarity_scores:
        final_score.append(max(cos_sim))

    #print(final_score)

    return np.mean(final_score)

def openai_embeddor(list1, list2, openai_model="text-embedding-ada-002"):

    client = OpenAI(api_key='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')

    doc1 = "?".join(list1)
    doc2 = "?".join(list2)

    def get_embedding(text):

        response = client.embeddings.create(input=[text], model= openai_model)

        #return response["data"][0]["embedding"]
        return response.data[0].embedding

    embedding1 = get_embedding(doc1)
    embedding2 = get_embedding(doc2)

    # Calculate cosine similarity for all pairwise combinations

    def cos_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    similarity_scores = cos_similarity(embedding1, embedding2)

    return [[similarity_scores]]

    



    


    



    

    







    

    

