from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from models import Hubert, Wav2Vec, BEATs
import torch

device = torch.device("cuda:7")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your actual model instances
MODEL_NAMES = {
    'BEATs_SoftMax': BEATs("/nfs_storage/2022/mohit.goyani/Karthik/evaluator/Final_model/BEATs_softmax/checkpoint-7888", device).to(device),
    'BEATs_Sigmoid': BEATs("/nfs_storage/2022/mohit.goyani/Karthik/evaluator/Final_model/BEATs_sigmoid/checkpoint-7888", device).to(device),
    'Hubert': Hubert("/nfs_storage/2022/mohit.goyani/Karthik/Hubert/fairseqmain/hubert_finetuned/hubert/checkpoint-10880", device).to(device),
    'Wav2Vec': Wav2Vec("/nfs_storage/2022/mohit.goyani/Karthik/Hubert/fairseqmain/wav2vec_finetuned/hubert/checkpoint-16320", device).to(device)
}

# Dummy inference function (replace with your real model logic)
def run_model(model_name, audio_path):
    with torch.no_grad():
        return MODEL_NAMES[model_name].forward(audio_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_model = None
    selected_audio_choice = 'upload'

    if request.method == 'POST':
        selected_model = request.form.get('model')
        selected_audio_choice = request.form.get('audio_choice')

        audio_path = None

        if selected_audio_choice == 'upload':
            audio_file = request.files.get('audio')
            if audio_file and audio_file.filename != '':
                filename = secure_filename(audio_file.filename)
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio_file.save(audio_path)
        elif selected_audio_choice == 'sample1':
            audio_path = 'static/audio/sample1.wav'
        elif selected_audio_choice == 'sample2':
            audio_path = 'static/audio/sample2.wav'

        if audio_path and selected_model:
            prediction = run_model(selected_model, audio_path)

    return render_template(
        'index.html',
        model_names=MODEL_NAMES.keys(),
        prediction=prediction,
        selected_model=selected_model,
        selected_audio_choice=selected_audio_choice
    )

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host="0.0.0.0")
