from keras import backend as K
import tensorflow as tf
import pickle
import pandas as pd
import os
from string import punctuation
import re
import string
from itertools import groupby
import numpy as np
import time

start = time.time()

class hatespeech:
    label_encoder = pickle.load(open("Hatespeech/label_encoder.pkl", "rb"))
    tokenizer = pickle.load(open("Hatespeech/tokenizer.pkl", "rb"))

class imageCaption:
    transform = pickle.load(open('Image Caption/transform.pkl','rb'))
    model = pickle.load(open('Image Caption/model.pkl', 'rb'))
    model.eval()

def test_re(s):
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        return regex.sub(' ', s)

# clean words in the sentences:
def search(array, element):
    if element in array:
        return True
    else:
        return False
    
def diff_pos_neg_check(pat,arr):
    result_final=False
    result_binary = False
    result_kmp = False
    string=''
    result = search(arr,pat)
    if result != False:
        result_binary = True
    else:
        pass;
    if result_binary or result_kmp:
        result_final=True
    else:
        result_final=False
    if result_final:
        string='found'
        return string
    else:
        string='not found'
        return string

class islamophobia:

    def __init__(self,text=None,nlp=None,model=None):
        self.text = text
        self.nlp = nlp
        self.model = model

    def clean_tweet(self):
        stop = pickle.load(open("Islamophobia/stop.pkl", "rb"))
        preprocess_text = []
        for tweet in self.text:
            doc = self.nlp(tweet)
            tweet = str(tweet).lower()
            tweet = re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', tweet)
            tweet = re.sub('\$[a-zA-Z0-9]*', ' ', tweet)
            tweet = re.sub('\@[a-zA-Z0-9]*', ' ', tweet)
            tweet = re.sub('[^a-zA-Z\']', ' ', tweet)
            tweet = re.sub(r'\b(?:\d+|\w)\b\s*', ' ', tweet)
            tweet = re.sub(r"won't", "will not", tweet)
            tweet = re.sub(r"can\'t", "can not", tweet)
            tweet = re.sub(r"n\'t", " not", tweet)
            tweet = re.sub(r"\'re", " are", tweet)
            tweet = re.sub(r"\'s", " is", tweet)
            tweet = re.sub(r"\'d", " would", tweet)
            tweet = re.sub(r"\'ll", " will", tweet)
            tweet = re.sub(r"\'t", " not", tweet)
            tweet = re.sub(r"\'ve", " have", tweet)
            tweet = re.sub(r"\'m", " am", tweet)
            tweet = [token.lemma_ for token in doc if token.text not in stop]
            preprocess_text.append(tweet)
        return preprocess_text

    def sim_check(self):
        rel = pickle.load(open("Islamophobia/rel.pkl", "rb"))
        terms = pickle.load(open("Islamophobia/terms.pkl", "rb"))
        eng = pickle.load(open("Islamophobia/eng.pkl", "rb"))
        stop = pickle.load(open("Islamophobia/stop.pkl", "rb"))
        abusive=False
        religious=False
        ban=False
        clean=False
        check = False
        result_=False
        text_lemma = ''
        result_rel=[]
        result_abusive=[]
        text_clean = []
        
        text = [self.text]
        for t in text:
            doc = self.nlp(t)
        for tokens in doc:
            text_lemma = text_lemma+' '+tokens.lemma_
        text_lemma = text_lemma.strip()
        text_lemma = test_re(text_lemma)
        splitted_text = text_lemma.lower().split()
        text_clean.append([word for word in splitted_text if word not in stop])
        text_clean = [x for sublist in text_clean for x in sublist]
        for word in range(len(text_clean)):
            rep_rem = "".join(c for c, _ in groupby(text_clean[word]))
            if eng.check(rep_rem) or (rep_rem in terms) or (rep_rem in rel):
                text_clean[word] = rep_rem
            else:
                pass;
            
        text = " ".join(text_clean)
        for s in text.split():
            try:
                result_abusive.append(diff_pos_neg_check(s, terms))
                result_rel.append(diff_pos_neg_check(s, rel))
            except:
                print('Word length is less')
        
        if ('found' in result_abusive) and ('found' in result_rel):
            ban=True
            abusive=False
            religious=False
        else:
            if 'found' in result_rel:
                religious=True
            else:
                clean=True
            if 'found' in result_abusive:
                abusive=True
            else:
                clean=True

        if ban:
            return 'Strong Islamophobia'
        else:
            if abusive and religious:
                result_=True
            else:
                if religious:
                    check=True
                else:
                    if abusive:
                        return 'Weak Islamophobia'
                    else:
                        return 'No-Islamophobia'    
        if check:
            doc = self.nlp(text)
            for token in doc:
                if (token.lemma_ in rel and token.dep_ in ('nsubj','dobj')):
                    try:
                        if doc[token.i+2].tag_ in ('JJ','VBG','VBZ','VB','VBP','RB','VBD') and doc[token.i+2].text in terms:
                            result_=True
                    except:
                        try:
                            if doc[token.i+1].dep_ == 'neg' or doc[token.i+2].dep_ in 'neg':
                                if doc[token.i+1].lemma_ == 'not' or doc[token.i+2].lemma_ == 'not':
                                    result_=True
                        except:
                            if doc[token.i-1].tag_ in ('VBG','VBD','VB','VBP','VBZ') and doc[token.i-2].dep_ in ('prep','aux') and (doc[token.i-3].tag_ in ('JJ','RB','NNS') or doc[token.i-2].tag_ in ('JJ','RB','NNS')):
                                result_=True
                            elif doc[token.i-1].tag_ in ('DT') and doc[token.i-1].dep_ in ('dobj','det') and doc[token.i-2].tag_ in ('VBG','VBD'):
                                result_=True
                            elif (doc[token.i-1].tag_ in ('VBG','VBD','VB','VBP','VBZ') or doc[token.i-2].dep_ in ('prep','aux')) and (doc[token.i-3].tag_ in ('JJ','RB') or doc[token.i-2].tag_ in ('JJ','RB')):
                                result_=True
            if result_:
                return 'Strong Islamophobia'
            else:
                return 'No-Islamophobia'
        else:
            pass;

    def prep_data_model(self):
        K.clear_session()
        # localhost_save_option = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")
        # model_load = load_model('Islamophobia/BERT_ISLAMOPHOBIA',options=localhost_save_option)
        tokenizer = pickle.load(open("Islamophobia/tokenizer.pkl", "rb"))
        label_encoder = pickle.load(open("Islamophobia/label_encoder.pkl", "rb"))
        tokens = tokenizer(self.text, max_length=150, truncation=True, 
                        padding='max_length', 
                        add_special_tokens=True, 
                        return_tensors='tf')
        tokens = {'input_ids': tf.cast(tokens['input_ids'], tf.float64), 'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}
        probs = self.model.predict(tokens)[0]
        pred = np.argmax(probs)
        pred = label_encoder.inverse_transform([pred])
        return np.round(probs,3),pred

class nudity:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "circular-cubist-339716-746fff6a0d6c.json"

class tags:
    # nlp = spacy.load("en_core_web_lg")
    # pickle.dump(nlp, open('nlp_spacy.pkl','wb'))
    def __init__(self, sentence=None, nlp=None, token=None):
        self.sentence = sentence
        self.nlp = nlp
        self.token = token

    def tokens(self):
        sentence_nlp = self.nlp(self.sentence)
        return [(word.lemma_, word.ent_type_) for word in sentence_nlp if word.ent_type_]

    def get_token(self):
        words = []
        for word, ner in self.token:
            if ner in ['WORK_OF_ART','LAW','PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT']:
                if word not in punctuation:
                    words.append(word)
        return words

    def get_DT(self):
        words = []
        for word, ner in self.token:
            if ner in ['DATE','TIME']:
                words.append(word)
        return words

    # def getdict(self):
    #     return {"entities": self.get_token,
    #             "dates": self.get_DT}

class translation:
    translator=pickle.load(open('translation/translator.pkl', 'rb'))
    translator=translator()

class similarity:
    X = pickle.load(open('Hadees/tfidfFeaturesX.pkl','rb'))
    v = pickle.load(open('Hadees/tfidfVectorsV.pkl','rb'))

    data = pd.read_csv('Hadees/data_of_ahadees.csv')
    data = data[~data['Text'].isnull()].reset_index(drop=True)
    data['Text'] = data['Text'].apply(lambda x: str(x).strip())

end = time.time()
print(end-start)