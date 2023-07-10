from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pandas as pd
import joblib
import pickle


app = Flask(__name__)

model = joblib.load('regressor.pkl')
onehot = joblib.load('OneHotee.joblib')


@app.route('/')
@app.route('/main')
def main():
	return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features =[[x for x in request.form.values()]]
	print("##"*20)
	print(int_features)
	c = ["make","model","vehicle_class","transmission","fuel","engine","cylinder","co2","smokerating"]
	df = pd.DataFrame(int_features,columns=c)
	print("@"*30)
	print(df)
	l = onehot.transform(df.iloc[:,:5])
	c = onehot.get_feature_names_out()
	t = pd.DataFrame(l,columns=c)
	l2 = df.iloc[:,5:]
	final =pd.concat([l2,t],axis=1)
	result = model.predict(final)
	print("The Result is :",result)


	print(int_features)

	return render_template("main.html",prediction_text="Your Vehicle Fuel Consumption  is : {}".format(result))


if __name__ == "__main__":
	app.debug=True
	app.run(host = '0.0.0.0', port =5000)
