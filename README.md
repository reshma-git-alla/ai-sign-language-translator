# AI Sign Language Translator

This project is a semester-ready NNDL mini system for recognizing sign-language gestures from webcam video. It uses MediaPipe to extract 3D hand landmarks and an LSTM-based deep learning model to classify temporal sign sequences.

## What this project includes

- `collect_data.py`: records labeled hand-sign sequences from your webcam.
- `train.py`: trains an LSTM classifier on the captured sequences.
- `realtime_translator.py`: runs live inference and builds a translated sentence.
- `labels.json`: editable list of target signs for your dataset.
- `src/sign_translator/`: reusable package for landmarks, model definition, dataset handling, and inference.

## Suggested project workflow

1. Choose 4 to 8 signs for your first milestone in `labels.json`.
2. Collect 20 to 30 samples per sign with `collect_data.py`.
3. Train the model with `train.py`.
4. Run `realtime_translator.py` to test live translation.
5. Improve accuracy by collecting more balanced samples, better lighting, and more signer variation.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Deployable web app

This project now includes a Streamlit deployment entrypoint at `app.py`.

Run it locally:

```powershell
streamlit run app.py
```

The web app accepts an uploaded video file, processes the frames with MediaPipe, and uses your trained model from `artifacts/models/sign_translator.keras` to predict the sign sequence.

### Deploy on Streamlit Cloud

1. Push this project to GitHub.
2. Open Streamlit Cloud and create a new app from your repo.
3. Set the main file path to `app.py`.
4. Deploy.

Notes:

- Make sure `artifacts/models/sign_translator.keras` is present in the repo or generated during deployment.
- The deployed app uses uploaded videos instead of desktop webcam capture because hosted servers cannot reliably open your local camera through OpenCV windows.

## Usage

### 1. Edit your labels

Update `labels.json` with the signs you want to recognize. Example:

```json
[
  "HELLO",
  "THANK_YOU",
  "YES",
  "NO",
  "PLEASE"
]
```

### 2. Collect data

```powershell
python collect_data.py --samples 25
```

Recording flow:

- The webcam opens one label at a time.
- Press `Space` to start recording a sample.
- Hold the sign steadily during the countdown and recording.
- Press `Q` at any time to quit.

Saved landmark sequences are stored in `data/raw/<LABEL>/`.

### 3. Train the deep learning model

```powershell
python train.py --epochs 40 --batch-size 16
```

Artifacts created:

- `artifacts/models/sign_translator.keras`
- `artifacts/metadata/model_info.json`
- `data/processed/training_history.json`

### 4. Run real-time translation

```powershell
python realtime_translator.py
```

Controls:

- `Q`: quit the app
- `C`: clear the current translated sentence

If no trained model exists yet, the app falls back to a simple heuristic demo mode so the pipeline still opens and can be demonstrated.

## If some signs get confused

Common confusing pairs are signs with similar hand shapes such as `NO` vs `HELLO` or `PLEASE` vs `THANK_YOU`.

To improve those cases:

- Record extra samples only for the confusing labels.
- Keep one consistent style for each label and do not mix variants.
- Make the starting hand position distinct for each sign.
- Use the same camera distance, lighting, and background.
- Retrain after adding the new samples.

The live app now also rejects uncertain predictions unless they stay stable for several frames, which helps reduce false words.

## Model architecture

The current neural network is:

- Input: `30 x 63` landmark sequence
- LSTM(64) -> Dropout
- LSTM(128) -> Dropout
- LSTM(64)
- Dense(64) -> Dense(32) -> Softmax

This is a good NNDL baseline because it learns from temporal motion patterns instead of single-frame classification.

## Recommended improvements for your report/demo

- Add more signs and more signers to improve generalization.
- Compare LSTM with GRU or 1D CNN + LSTM.
- Plot training and validation accuracy from `training_history.json`.
- Add confusion matrix evaluation after training.
- Extend to two-hand signs by increasing `max_num_hands`.
- Build a Streamlit or Flask frontend after the core model is stable.

## Suggested report structure

- Problem statement: real-time sign-to-text translation for human-computer interaction.
- Dataset: webcam-collected landmark sequences for selected signs.
- Preprocessing: hand detection, landmark normalization, temporal sequence construction.
- Model: LSTM-based sequence classifier.
- Loss/optimizer: sparse categorical cross-entropy with Adam.
- Evaluation: test accuracy, live demo observations, failure cases.
- Future work: sentence-level translation, full ISL/ASL vocabulary, transformer models.

## Notes

- This project currently recognizes isolated signs, not full grammar-aware sign language sentences.
- Accuracy depends heavily on clean data collection.
- For a better academic demo, keep your initial label set small and well-separated.
