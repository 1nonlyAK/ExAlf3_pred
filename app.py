from flask import Flask, request, render_template
import h2o
import pandas as pd

h2o.init()

model = h2o.import_mojo("model/GBM_grid_1_AutoML_3_AutoML_3_20250623_62009_model_24.zip")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {k: float(v) for k, v in request.form.items()}
    df = pd.DataFrame([input_data])
    h2o_df = h2o.H2OFrame(df)
    prediction = model.predict(h2o_df).as_data_frame().values[0][0]
    return render_template("index.html", prediction=round(prediction, 4))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)