from lib2to3.pgen2 import token
import re
import nltk
import pickle
import string
import spacy
from itertools import groupby
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from nltk.stem import WordNetLemmatizer

# lemma = pickle.load(open("lemma.pkl", "rb"))
lemma = WordNetLemmatizer()
# nlp = spacy.load("Islamophobia")
stop = pickle.load(open("Islamophobia/stop.pkl", "rb"))
rel = pickle.load(open("Islamophobia/rel.pkl", "rb"))
terms = pickle.load(open("Islamophobia/terms.pkl", "rb"))
eng = pickle.load(open("Islamophobia/eng.pkl", "rb"))
localhost_save_option = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")
model_load = load_model('Islamophobia/BERT_ISLAMOPHOBIA',options=localhost_save_option)
tokenizer = pickle.load(open("Islamophobia/tokenizer.pkl", "rb"))
label_encoder = pickle.load(open("Islamophobia/label_encoder.pkl", "rb"))

def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def clean_tweet(tweets):
    preprocess_text = []
    for tweet in tweets:
        doc = nlp(tweet)
        tweet = str(tweet).lower()
        tweet = re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', tweet)
        tweet = re.sub('\$[a-zA-Z0-9]*', ' ', tweet)
        tweet = re.sub('\@[a-zA-Z0-9]*', ' ', tweet)
        tweet = re.sub('[^a-zA-Z\']', ' ', tweet)
        tweet = re.sub(r'\b(?:\d+|\w)\b\s*', ' ', tweet)
        tweet = decontracted(tweet)
        tweet = [token.lemma_ for token in doc if token.text not in stop]
        preprocess_text.append(tweet)
    return preprocess_text

list_remove = ['islam','muslim','sex','threesome','exmuslims','fuck','fck','ex-muslim','ex muslim','musalman','ex','jerusalem',
               'iran','makkah','madina','dick','lesbian','nude','nudity','porn','sexism','racism','religion','prayer','pornhub',
               'xxx','xxxxx','xvideos','xnxx','north america','naughty','america','prophet','die','hindu','muslim','hate','kill',
               'masjid','women','narrated','allah','muhammad','say','quran','bible','holy','islamophobia','islamophobic','imam',
               'children','jihaad','jihad','anti-Muslim','anti-Muslim hate','hate crime','social policy','equality','extremism', 
               'muslims in greece','hate muslims','end muslims','sex muslims','fuck muslims','muslims in united states','muslims in usa',
               'muslims in pakistan','muslims in india','india','indian','mosque attacks','attacks','attack','terror','terrorist','terrorism',
               'discrimination','black lifes','ali','namaz','pbuh','worship','god','gods','killer','enemy','palestine','rape']

# clean words in the sentences:
def search(array, element):
    if element in array:
        return True
    else:
        return False
    
def nameMatch(alumniNames, muslimNames):
    matches = False
    name = alumniNames.split(' ')
    lastName=''
    if len(name) == 2:
        lastName = name[1]
        firstName = name[0]
        if search(muslimNames, firstName) or search(muslimNames,lastName):
            matches = True
        else:
            matches=False
    else:
        firstName = name[0]
        if search(muslimNames, firstName):
            matches = True
        else:
            matches=False
    return matches
    
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
#         for word in arr:
#             result = patternSearching(pat, word)
#             if result == None:
#                 result_kmp = False
#             else:
#                 result_kmp = True
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

def test_re(s):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub(' ', s)
def sim_check(text):
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
    
    text = [text]
    for t in text:
        doc = nlp(t)
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
        doc = nlp(text)
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

def prep_data_model(text):
    tokens = tokenizer(text, max_length=150, truncation=True, 
                      padding='max_length', 
                      add_special_tokens=True, 
                      return_tensors='tf')
    tokens = {'input_ids': tf.cast(tokens['input_ids'], tf.float64), 'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}
    probs = model_load.predict(tokens)[0]
    pred = np.argmax(probs)
    pred = label_encoder.inverse_transform([pred])
    return np.round(probs,3),pred