import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['mean_radius', 'mean_texture', 'mean_perimeter',
                     'mean_area', 'mean_smoothness']

    df = pd.DataFrame(features_value, columns=features_name)
    dataset = pd.read_csv('breast.csv')
    X = dataset.iloc[:, :-1].values
    X_train = sc.fit_transform(X)
    f = sc.transform(df)

    output = model.predict(f)

    if output == 1:
        res_val = "Breast cancer"
    else:
        res_val = "no Breast cancer"

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
    app.run()
