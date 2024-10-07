from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Tải các mô hình
models = {}
model_names = ['id3', 'pla_model', 'Neural_Network', 'stacking']
for model_name in model_names:
    with open(f'Model/{model_name}.pkl', 'rb') as file:
        models[model_name] = joblib.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        Airplane = float(request.form['Airplane'])
        Dep_time = float(request.form['Dep_time'])
        Duration = float(request.form['Duration'])
        Total_Stops = float(request.form['Total_Stops'])
        Price = float(request.form['Price'])
        selected_model = request.form['model']

        input_data = np.array([[Airplane, Dep_time, Duration, Total_Stops, Price]])

        # Dự đoán với mô hình đã chọn
        prediction = models[selected_model].predict(input_data)[0]

    return render_template('index.html', prediction=prediction, model_names=model_names)

if __name__ == '__main__':
    # Lắng nghe trên tất cả các IP và sử dụng biến môi trường PORT
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
