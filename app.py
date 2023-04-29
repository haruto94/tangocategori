from flask import Flask, render_template, request
import torch
from word2vec_ import word2vec_BERT
from vec_categorize import vec_categorize
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    word = request.form['word']
    
    vector = word2vec_BERT(word) # word2vec.pyのword2vec関数を呼び出す
    vector = torch.tensor(vector, dtype=torch.float32)
    result = vec_categorize(vector)
    return render_template('result.html', word=word, categories=result)

if __name__ == '__main__':
    app.run(port=8080, debug=True)

