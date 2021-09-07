import nltk
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english') # or any other list of your choice

def common_words(doc):
    try:
        doc = doc.split('.')
        cv=CountVectorizer()
        word_count_vector=cv.fit_transform(doc)

        tfidf_transformer=TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        result = list(zip(tfidf_transformer.idf_, cv.get_feature_names()))
        result.sort()
        #result.reverse()

        result = result[0:1000]

        #print(len(doc), [y + '=' + str(x) for x, y in result], '\n')

        result = [y for x, y in result]
        
        result = ' '.join(result)
    except:
        result = ''
    return result


def unique_words(doc):
    try:
        doc = doc.split('.')
        cv=CountVectorizer()
        word_count_vector=cv.fit_transform(doc)

        tfidf_transformer=TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        result = list(zip(tfidf_transformer.idf_, cv.get_feature_names()))
        result.sort()
        result.reverse()

        result = result[0:1000]

        #print(len(doc), [y + '=' + str(x) for x, y in result], '\n')

        result = [y for x, y in result]
        
        result = ' '.join(result)
    except:
        result = ''
    return result
    
    

def clean_text(text, ):

    def cleanhtml(raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext
  
    def remove_url(raw_url):
        url_out = re.sub(r"http\S+", '', raw_url)
        url_out = re.sub(r"www\S+", '', raw_url)
        return url_out

    def remove_special_characters(inp_txt, characters=string.punctuation.replace('.', '').replace('_', '').replace('-', '')):
        tokens = inp_txt.split(' ') # text tokenisation
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return (' ').join(filter(None, [pattern.sub('', t) for t in tokens]))

    def stem_text(inp_txt, stemmer=default_stemmer):
        tokens = inp_txt.split(' ') # text tokenisation
        stemmed_text = [stemmer.stem(t) if t!='.' else t for t in tokens]
        return (' ').join(stemmed_text)

    def remove_stopwords(inp_txt, stop_words=default_stopwords):
        tokens = inp_txt.split(' ') # text tokenisation
        removed_sw = [t for t in tokens if t not in stop_words]
        return (' ').join(removed_sw)
        
    def remove_singles(inp_txt):
        tokens = inp_txt.split(' ') # text tokenisation
        single_w = [t for t in tokens if t=='.' or len(t) >= 2]
        return (' ').join(single_w)


    text = str(text).strip(' ') # strip whitespaces
    text = str(text).replace('.', ' . ') # replace dots
    text = str(text).replace('\n', '')
    text = text.lower() # lowercase
    
    text = cleanhtml(text) # remove html
    text = remove_url(text)
    # text = stem_text(text) # stemming
    text = remove_special_characters(text) # remove punctuation and symbols
    text = remove_stopwords(text) # remove stopwords
    text = remove_singles(text)

    return text