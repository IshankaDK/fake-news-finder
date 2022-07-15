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
def main():
 	return render_template("main.html")

@app.route('/text')
def findByText():
 	return render_template("index.html")


@app.route('/link')
def findByLink():
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
@app.route('/predict')
def predict():
    url = request.args.get('news')
    # url =request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    # article.nlp()
    # news = "Wall Street Journal editor and Assistant Secretary of the Treasury for Economic Policy, I am astonished at the corruption that rules in the financial sector, the Treasury, the financial regulatory agencies, and the Federal Reserve.  In my day, there would have been indictments and prison sentences of bankers and high government officials.In America today there are no free financial markets.  All the markets are rigged by the Federal Reserve and the Treasury. The regulatory agencies, controlled by those the agencies are supposed to regulate, turn a blind eye, and even if they did not, they are helpless to enforce any law, because private interests are more powerful than the law.Even the government s statistical agencies have been corrupted. Inflation measures have been concocted in order to understate inflation. This lie not only saves Washington from paying Social Security cost-of-living adjustments and frees the money for more wars, but also by understating inflation, the government can create real GDP growth by counting inflation as real growth, just as the government creates 5% unemployment by not counting any discouraged workers who have looked for jobs until they can no longer afford the cost of looking and give up.  The official unemployment rate is 5%, but no one can find a job.  How can the unemployment rate be 5% when half of 25-year olds are living with relatives because they cannot afford an independent existence?  As John Williams (shadowfacts) reports, the unemployment rate that includes those Americans who have given up looking for a job because there are no jobs to be found is 23%.The Federal Reserve, a tool of a small handful of banks, has succeeded in creating the illusion of an economic recovery since June, 2009, by printing trillions of dollars that found their way not into the economy but into the prices of financial assets.  Artificially booming stock and bond markets are the presstitute financial media s  proof  of a booming economy.The handful of learned people that America has left, and it is only a small handful, understand that there has been no recovery from the previous recession and that a new downturn is upon us.  John Williams has pointed out that US industrial production, when properly adjusted for inflation, has never recovered its 2008 level, much less its 2000 peak, and has again turned down.The American consumer is exhausted, overwhelmed by debt and lack of income growth. The entire economic policy of America is focused on saving a handful of NY banks, not on saving the American economy.Economists and other Wall Street shills will dismiss the decline in industrial production as America is now a service economy. Economists pretend that these are high-tech services of the New Economy, but in fact waitresses, bartenders, part time retail clerks, and ambulatory health care services have replaced manufacturing and engineering jobs at a fraction of the pay, thus collapsing effective aggregate demand in the US. On occasions when neoliberal economists recognize problems, they blame them on China.It is unclear that the US economy can be revived. To revive the US economy would require the re-regulation of the financial system and the recall of the jobs and US GDP that offshoring gave to foreign countries. It would require, as Michael Hudson demonstrates in his new book, Killing the Host, a revolution in tax policy that would prevent the financial sector from extracting economic surplus and capitalizing it in debt obligations paying interest to the financial sector.The US government, controlled as it is by corrupt economic interests, would never permit policies that impinged on executive bonuses and Wall Street profits.  Today US capitalism makes its money by selling out the American economy and the people dependent upon it.In  freedom and democracy  America, the government and the economy serve interests totally removed from the interests of the American people. The sellout of the American people is protected by a huge canopy of propaganda provided by free market economists and financial presstitutes paid to lie for their living.When America fails, so will Washington s vassal states in Europe, Canada, Australia, and Japan.  Unless Washington destroys the world in nuclear war, the world will be remade, and the corrupt and dissolute West will be an insignificant part of the new world.Dr. Paul Craig Roberts was Assistant Secretary of the Treasury for Economic Policy and associate editor of the Wall Street Journal. He was columnist for Business Week, Scripps Howard News Service, and Creators Syndicate. He has had many university appointments. His internet columns have attracted a worldwide following. Roberts  latest books are The Failure of Laissez Faire Capitalism and Economic Dissolution of the West, How America Was Lost, and The Neoconservative Threat to World Order. READ MORE NWO NEWS AT: 21st Century Wire NWO Files"
    news = article.summary
    print("news" , news)
    # #Passing the news article to the model and returing whether it is Fake or Real
    pred = model.predict([news])
    return jsonify(result = pred[0])



if __name__=='__main__':
    app.run(debug=True)
