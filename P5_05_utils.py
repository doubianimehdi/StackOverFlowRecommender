import numpy as np
import pandas as pd
import pickle 
import re, spacy, nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import ToktokTokenizer
from nltk.stem import wordnet
import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
token = ToktokTokenizer()
punct = punctuation

def avg_jaccard(y_true, y_pred):
    ''' It calculates Jaccard similarity coefficient score for each instance,and
    it finds their averange in percentage

    Parameters:

    y_true: truth labels
    y_pred: predicted labels
    '''
    jacard = np.minimum(y_true, y_pred).sum(axis=1) / \
        np.maximum(y_true, y_pred).sum(axis=1)

    return jacard.mean()*100

def clean_text(text):
        ''' Lowering text and removing undesirable marks

        Parameter:

        text: document to be cleaned    
        '''

        text = text.lower()
        text = re.sub(r"\'\n", " ", text)
        text = re.sub(r"\'\xa0", " ", text)
        text = re.sub('\s+', ' ', text)  # matches all whitespace characters
        text = text.strip(' ')
        return text
	
def strip_list_noempty(mylist):

    newlist = (item.strip() if hasattr(item, 'strip')
               else item for item in mylist)

    return [item for item in newlist if item != '']

def clean_punct(text, top_tags):
    ''' Remove all the punctuation from text, unless it's part of an important 
    tag (ex: c++, c#, etc)

    Parameter:

    text: document to remove punctuation from it
    '''

    words = token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct)
    top_tags = top_tags

    for w in words:
        if w in top_tags:
            punctuation_filtered.append(w)
        else:
            w = re.sub('^[0-9]*', " ", w)
            punctuation_filtered.append(regex.sub('', w))

    filtered_list = strip_list_noempty(punctuation_filtered)

    return ' '.join(map(str, filtered_list))
	
def stopWordsRemove(text):
    ''' Removing all the english stop words from a corpus

    Parameter:

    text: document to remove stop words from it
    '''

    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]

    return ' '.join(map(str, filtered))

def lemmatization(texts, allowed_postags, top_tags,stop_words=stop_words):
        ''' It keeps the lemma of the words (lemma is the uninflected form of a word),
        and deletes the underired POS tags

        Parameters:

        texts (list): text to lemmatize
        allowed_postags (list): list of allowed postags, like NOUN, ADL, VERB, ADV
        '''

        lemma = wordnet.WordNetLemmatizer()
        doc = nlp(texts)
        texts_out = []
        top_tags = top_tags
		
        for token in doc:

            if str(token) in top_tags:
                texts_out.append(str(token))

            elif token.pos_ in allowed_postags:

                if token.lemma_ not in ['-PRON-']:
                    texts_out.append(token.lemma_)

                else:
                    texts_out.append('')

        texts_out = ' '.join(texts_out)

        return texts_out

def pred_nwords_unsupervised(text, tfidf, lda, n_words):
    ''' Recommend n_words tags by detecting latent topics in a corpus
    Parameters:    
    text: cleaned text on which recommendations are based
    tfidf: tfidf transformer
    lda: lda model
    n_words: number of words retrieved
    '''

    document_tfidf = tfidf.transform(text)
    proba_topic_sachant_document = lda.transform(document_tfidf)
    words_label = []
    for word in tfidf.get_feature_names():
        words_label.append(word)
    proba_word_sachant_topic = lda.components_ / \
        lda.components_.sum(axis=1)[:, np.newaxis]  # normalization

    # proba_topic_sachant_document est de dimension d x t
    # proba_word_sachant_topic est de dimension t x w
    # je peux donc opérer un produit matriciel entre les 2 matrices pour calculer pour chaque document : proba(wordn)
    # j'obtiendrai une matrice proba_word_sachant_document de dimension d x w
    # il ne me restera plus qu'à choisir les "n_words" mots les plus probables
    proba_word_sachant_document = proba_topic_sachant_document.dot(
        proba_word_sachant_topic)

    # je transforme la matrice en dataframe :
    # data = les proba des mots pour chaque document
    # index = l'index des données en entrée
    # columns = les labels des mots sélectionnés en sortie du LDA
    df_wd = pd.DataFrame(data=proba_word_sachant_document,
                         index=text.index,
                         columns=words_label)

    values = df_wd.columns.values[np.argsort(
        -df_wd.values, axis=1)[:, :n_words]]
    values = [", ".join(item) for item in values.astype(str)]
    pred_unsupervised = pd.DataFrame(values,
                                     index=df_wd.index,
                                     columns=['Unsupervised'])

    return pred_unsupervised

def recommend_tags(text_ori, n_words, seuil=0.5, clean=False):
    ''' Recommendation system for StackOverflow posts based on a unsupervised AND supervised model which returns up to 5 words.
    Parameters:
    text_ori: the stackoverflow post of user
    n_words: number of tags to recommend
    seuil: threshold for decision
    clean: True if data preparation is needed
    '''
    # CHARGEMENT
    with open('vectorizer_dfText.pkl', 'rb') as f:
        vectorizer_dfText = pickle.load(f)    
    with open('lda_model.pkl', 'rb') as f:
        best_lda = pickle.load(f)    
    with open('lr_ovr.pkl', 'rb') as f:
        lr_ovr = pickle.load(f) 
    with open('multilabel_binarizer.pkl', 'rb') as f:
        multilabel_binarizer = pickle.load(f)
    with open('top_tags.pkl', 'rb') as f:
        top_tags = pickle.load(f) 
      
    if type(text_ori) in (str, pd.Series):
        if type(text_ori) is str:
            text_ori = pd.Series(text_ori)
        text = text_ori
        text_ori = text_ori.rename("Texte d'origine")
        text = text.rename("Texte modifié")
    else:
        return 'Type should be str or pd.Series'

    if clean == True:
        text = text.apply(lambda s: clean_text(s))
        text = text.apply(lambda s: BeautifulSoup(s,features="lxml").get_text())
        text = text.apply(lambda s: clean_punct(s,top_tags))
        text = text.apply(lambda s: stopWordsRemove(s))
        text = text.apply(lambda s: lemmatization(s,['NOUN', 'ADV'],top_tags))

    pred_unsupervised = pred_nwords_unsupervised(
        text, vectorizer_dfText, best_lda, n_words)
    pred_supervised = pd.DataFrame(lr_ovr.predict_proba(vectorizer_dfText.transform(
        text))).applymap(lambda x: 1 if x > seuil else 0).to_numpy()
    pred_supervised = pd.Series(multilabel_binarizer.inverse_transform(
        pred_supervised), name='Supervised', index=text.index)
    pred_supervised = pred_supervised.apply(lambda row: ', '.join(row))
    result = pd.concat(
        [pred_supervised, pred_unsupervised, text_ori, text], axis=1)

    return result