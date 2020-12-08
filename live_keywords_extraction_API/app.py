# Sentiment polarity score libraries.
from textblob import TextBlob

# NLP Libraries
import spacy
import nltk
from nltk.corpus import wordnet
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize     
import en_core_web_sm  # SpaCy language model.
nlp = en_core_web_sm.load()
import pytextrank
import flask 
from flask import request,jsonify,render_template

# Preprocessing libraries
import re
from nlppreprocess import NLP # for removing stopwords except negation words.
#import contractions
obj=NLP(lemmatize=True,lemmatize_method='wordnet')

#text = 'This cozy restaurant has left the best impressions! Hospitable hosts, delicious dishes, beautiful presentation, wide wine list and wonderful dessert.# & % @: I recommend to everyone! I Can\'t  went wouldn\'t I would like to come back here again and again.'

tr = pytextrank.TextRank()
# add PyTextRank to the spaCy pipeline
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

# Main Keywords Function.
def main_keywords(text):
    '''Extract keywords from text using pytextrank.'''
    
    # Data Cleaning (Remove unwanted Characters).
    text = re.sub('[\n!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]','',text)
    
    # Replace can't to can not.
    #text = contractions.fix(text)
    
    # obj.process to remove stopwords and lemmatization
    text = obj.process(text)
    
    # Pass text in Spacy nlp object to get keywords.
    doc = nlp(text)
    
    # examine the top-ranked phrases in the document
    tags = [p.text for p in doc._.phrases if ' ' in str(p.text)]
    
    # Filter keywords with positive or neutral score.
    keywords = [each for each in tags if round(TextBlob(str(each)).sentiment.polarity, 3)>=0.0]
    # Unique Keywords
    keywords = list(set(keywords))
    # Keep maximum 5 keywords.
    #keywords = keywords[:5]
    
    # join Keywords in Single string with ,(comma) Seperator.
    keywords = ','.join(keywords)
    
    return keywords



app = flask.Flask(__name__,template_folder='templates')


@app.route('/',methods = ['GET'])
def home():
    return render_template('index.html')

@app.route('/get_keywords',methods=['POST'])
def get_keywords():
    '''
    For rendering results on HTML GUI
    '''
    r_text = request.form['ureview']
    print(r_text)
    output=main_keywords(str(r_text))
    
    return render_template('index.html',extracted_keywords='Extracted Keywords are : {}'.format(output ))

@app.route('/live_keywords/<string:text>', methods = ['GET'])
def live_keywords(text):
    '''API to get keywords from text using text.'''
    output = main_keywords(text)
    return output
    
if __name__=='__main__':
    app.run()
    
