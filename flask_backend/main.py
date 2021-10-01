from flask import jsonify, request, Flask
from datetime import datetime, timezone
from collections import defaultdict
import json
import os
from flask_cors import CORS
import time
import scispacy
import spacy
from spacy import displacy
from datamap import disease, symptoms
import tensorflow as tf

app = Flask(__name__)

CORS(app=app)

@app.route('/')
def index():
    return "Hello!!"

@app.route('/get-symptoms',methods=['POST'])
def get_symptoms():
    text = request.get_json('text')
    # !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_bc5cdr_md-0.4.0.tar.gz


    nlp = spacy.load("en_ner_bc5cdr_md")
    
    doc = nlp(text)

    predictions = []
    
    for entity in doc.ents:
        if entity.label_=="DISEASE":
            predictions.append(entity.text)
    # displacy.render(doc, style='ent', jupyter=True)

    return jsonify(symptoms=predictions)


def get_disease(symptoms):
    model_1 = tf.keras.models.load_model('sym2dis.h5')
    output = model_1.predict(tf.expand_dims(symptoms, axis=0))
    predict_class = tf.argmax(tf.squeeze(output, axis=0))
    
    return disease[predict_class]

    

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
    

