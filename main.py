from flask import Flask, render_template, request, url_for, Markup, jsonify
import pickle
from flask_cors import CORS

from newspaper import Article
import urllib

app = Flask(__name__)
CORS(app)

pickle_in = open('model_fakenews.pickle','rb')
pac = pickle.load(pickle_in)
tfid = open('tfid.pickle','rb')
tfidf_vectorizer = pickle.load(tfid)

with open('model.pickle', 'rb') as handle:
	model = pickle.load(handle)


@app.route('/')
def home():
 	return render_template("main.html")

@app.route('/text')
def home():
 	return render_template("index.html")


@app.route('/link')
def main():
    return render_template('index2.html')


@app.route('/newscheck')
def newscheck():	
	abc = request.args.get('news')	
	input_data = [abc.rstrip()]
	# transforming input
	tfidf_test = tfidf_vectorizer.transform(input_data)
	# predicting the input
	y_pred = pac.predict(tfidf_test)
	return jsonify(result = y_pred[0])

#Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict',methods=['GET','POST'])
def predict():
    url =request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    #Passing the news article to the model and returing whether it is Fake or Real
    pred = model.predict([news])
    return render_template('index2.html', prediction_text='The news is "{}"'.format(pred[0]))



if __name__=='__main__':
    app.run(debug=True)
