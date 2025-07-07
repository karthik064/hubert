from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from models import Hubert,Wav2Vec,BEATs
import torch

device = torch.device("cuda:7")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Dummy list of model names
# MODEL_NAMES = ['BEATs_SoftMax', 'BEATs_Sigmoid', 'Hubert', 'Wav2Vec']
MODEL_NAMES = {'BEATs_SoftMax':BEATs("/nfs_storage/2022/mohit.goyani/Karthik/evaluator/Final_model/BEATs_softmax/checkpoint-7888",device).to(device), 
               'BEATs_Sigmoid':BEATs("/nfs_storage/2022/mohit.goyani/Karthik/evaluator/Final_model/BEATs_sigmoid/checkpoint-7888",device).to(device), 
               'Hubert':Hubert("/nfs_storage/2022/mohit.goyani/Karthik/Hubert/fairseqmain/hubert_finetuned/hubert/checkpoint-10880",device).to(device),
                'Wav2Vec':Wav2Vec("/nfs_storage/2022/mohit.goyani/Karthik/Hubert/fairseqmain/wav2vec_finetuned/hubert/checkpoint-16320",device).to(device)
                }

    

# Dummy function for running inference
def run_model(model_name, audio_path):
    # Replace this with your actual logic
    with torch.no_grad(): 
        return MODEL_NAMES[model_name].forward(audio_path)
        
    
    return f"Predicted label by {model_name} for {os.path.basename(audio_path)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        selected_model = request.form['model']
        audio_file = request.files['audio']

        if audio_file and selected_model:
            print(audio_file)
            filename = audio_file.filename
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(filename)

            prediction = run_model(selected_model, filename)

    return render_template('index.html', model_names=MODEL_NAMES, prediction=prediction)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True,host="0.0.0.0")
