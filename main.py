import numpy as np
import re
import json
import tensorflow as tf
from tensorflow import keras
from flask import Flask,render_template,request,abort,jsonify 
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import  Concatenate

with open('word_dict.json') as f:
    dict_words = json.load(f)
    tokenizer = Tokenizer(filters='')
    tokenizer.word_index = dict_words

bigru_model = keras.models.load_model("bigru_model.h5")

index_word={}
for key,value in tokenizer.word_index.items():
  index_word[value]=key 

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

def final1(input_string,beam_search=False,k=3,max_len=12,no_suggestions=3):

  """takes input string and gives sentence completion suggestions"""

  #Preprocessing steps
  textdata = input_string
  textdata = re.sub(r"[a-zA-Z0-9_+.-]+@+[a-zA-Z0-9_+.-]+"," ",textdata) #removes mail ids
  textdata = re.sub(r'http\S+|www\S+'," ", textdata) #removes links/URLs
  textdata = re.sub(r"<.*>"," ",textdata) #removes string enclosed within < >
  textdata = re.sub(r"\[.*\]"," ",textdata) #removes string enclosed within [ ]
  textdata = re.sub(r"\(.*\)"," ",textdata) #removes string enclosed within ( )
  textdata = re.sub(r"[\n]{2,}",".",textdata) #replaces double or more new line char with '.'
  textdata = re.sub(r"[\t\n]"," ",textdata) #removes tabs, single new line char
  textdata = re.sub(r"[\-\\\_\/]"," ",textdata) #removes '-' '\' '_' and '/'
  textdata = re.sub(r"[a-zA-Z]+:"," ",textdata) #removes words that end with ':'
  textdata = re.sub(r"\b_|_\b"," ",textdata) #removes "_" at start\end of words
  textdata = re.sub(r'AM|PM|A\.M|P\.M|a\.m|p\.m|pm'," ",textdata) #remove timestamp texts 
  textdata = re.sub(r"\d+:\d+\.",".",textdata) #removes time, if at end and retains sentence
  textdata = textdata.lower() #makes all text lowercase
  x = re.findall(r'\S+\'\S+',textdata)
  if x is not None:
    for word in x:
      if word in CONTRACTION_MAP: #decontracts words
        textdata = textdata.replace(word, CONTRACTION_MAP[word.lower()])
  textdata = re.sub(r'[^a-zA-z!?,. ]',' ', textdata) #remove everything except [^a-zA-z!?,. ]
  textdata = re.sub(r'[?!,]','.', textdata) #changes '?!,' to '.'
  textdata = re.sub(r' +', ' ', textdata) #removes extra spaces in between string
  textdata = re.sub(r' *\.','.', textdata) #removing blank sentences
  textdata = re.sub(r'\.+', '.', textdata) #convert multidots to single
  textdata = re.sub(r'\. ','.',textdata) #removing space at start of each sentence
  textdata = textdata.strip('. ') #removes '.' and spaces in start\end of strings

  textdata = textdata.split('.')
  textdata = textdata[-1]    #only last sentence that need suggestions is retained 
  if len(textdata.split()) >= 22:   #if last sentence has more than 21 tokens , last 21 are retained
    textdata = textdata.split()[-21:] 

 #tokenizing steps
  textdata = "<start> " + textdata + " <end>" #adding start and end tokens
  data_sequence = tokenizer.texts_to_sequences([textdata,textdata])   #tokenizer takes minimum two elements
  encoder_seq = pad_sequences(data_sequence, maxlen=23, padding='post')
  encoder_seq = tf.expand_dims(encoder_seq[0], axis=0)
  #encoding part
  encoder_out, fw_state_h, ba_state_h = bigru_model.layers[3](bigru_model.layers[1](bigru_model.layers[0](encoder_seq)))

  #greedy search steps
  if beam_search == False:
    state_h = Concatenate()([fw_state_h, ba_state_h])
    dec_input = np.zeros((1,1))
    dec_input[0,0] = tokenizer.word_index['<start>']
    stop_condition=False
    sent=''
    
    #decoding part
    while not stop_condition:
        predicted_out, state_h = bigru_model.layers[6](bigru_model.layers[4](bigru_model.layers[2](dec_input)),initial_state=state_h)
        dense_out = bigru_model.layers[7](predicted_out)
        output = np.argmax(dense_out)                    #argmax ~= greedy search
        dec_input = np.reshape(output, (1, 1))
        if index_word[output] == '<end>':
            stop_condition=True
        else:
            sent=sent + ' ' + index_word[output]

    return sent

  #beam_search steps
  else:
    states = Concatenate()([fw_state_h, ba_state_h])

    #variable declaration
    start_token = tokenizer.word_index['<start>']
    end_token   = tokenizer.word_index['<end>']
    eos_sent = []
    eosent_score = []
    top_ksent = [[start_token]]
    top_kscore = [0]
    counter = 0

    #beam_search loop
    while counter<max_len: 
      temp_sent  = []
      temp_score = []
      counter += 1
      for i,sent in enumerate(top_ksent):
        state_h = states
        for j in range(len(sent)):
          dec_input = np.reshape(sent[j],(1,1))
          predicted_out, state_h = bigru_model.layers[6](bigru_model.layers[4](bigru_model.layers[2](dec_input)),initial_state=state_h)      
        score_0 = bigru_model.layers[7](predicted_out)[0,0,:]
        top_k0 = (np.argsort(score_0)[::-1][:k]).tolist()
        k_scores = np.log((np.sort(score_0)[::-1][:k])).tolist()
        if end_token in top_k0:
          eos_sent.append([*sent,end_token])
          eosent_score.append((top_kscore[i]+k_scores[top_k0.index(end_token)])/2)
          del k_scores[top_k0.index(end_token)]
          del top_k0[top_k0.index(end_token)]
        temp_sent.extend([[*sent,m] for m in top_k0])
        temp_score.extend([(m+top_kscore[i])/2 for m in k_scores])
    
      top_ksent =  [temp_sent[l] for l in (np.argsort(temp_score)[::-1][:k]).tolist()]
      top_kscore = (np.sort(temp_score)[::-1][:k]).tolist()

    #tokens to sentence conversion
    eos_index = (np.argsort(eosent_score)[::-1][:no_suggestions]).tolist()
    eos_sent = [eos_sent[h] for h in eos_index]
    eosent_score = (np.sort(eosent_score)[::-1][:no_suggestions]).tolist()
    
    best_sentences = []
    for p,sentence in enumerate(eos_sent):
      best_sent = ''
      for word in sentence[1:-1]:
        best_sent += ' ' + index_word[word]
      best_sentences.append(best_sent)
    
    return best_sentences

app = Flask(__name__) 
   
@app.route("/")
def hello():
    return "Hello World!"

@app.route('/index')
def index():
    return render_template('index.html')

@app.errorhandler(406)
def not_found(e):
    return jsonify(error = str(e)), 404

def form_error(input_string):
    if input_string == "":
        abort(406,"Please give some input")

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        input_string = request.form['input_string']
        form_error(input_string)
        try:
            beam = request.form["beam_search"]
        except:
            return jsonify(final1(input_string))
        else:
            beam = True
            return jsonify(final1(input_string,beam))
            
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080,debug=True)