# Face Mask Detection

A computer vision project that classifies an image as **WithMask** or **WithoutMask** using a PyTorch MobileNetV2 model. A FastAPI service exposes predictions through an image upload endpoint.

## Features

- MobileNetV2 image classification
- FastAPI prediction endpoint with confidence score
- Training and evaluation scripts
- Docker support
- Evaluation plots in `RESULTS/`

## Repository contents

- `app.py` — FastAPI inference service
- `train_model.py` — model training script
- `DataManagement.py` — dataset cleaning and splitting
- `evaluator.py` — evaluation and metrics
- `augmentation.py` — image augmentation utilities
- `EDAnotebook.py` — exploratory data analysis
- `mask_detector.pth` — trained model weights
- `RESULTS/` — saved evaluation plots
- `Dockerfile` and `requirements.txt` — deployment setup

## Dataset

The dataset is not included in this repository. Download it from [Kaggle: Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset).

## Run the API locally

Use Python 3.10, then install the dependencies:

```bash
pip install -r requirements.txt
```

Start the API from the repository root:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs), choose **POST /predict**, and upload an image.

Example response:

```json
{
  "class": "WithMask",
  "confidence": 0.98,
  "status": "mask_on",
  "action": "Allow entry"
}
```

## Run with Docker

```bash
docker build -t face-mask-detection .
docker run --rm -p 8000:8000 face-mask-detection
```

Then visit [http://localhost:8000/docs](http://localhost:8000/docs).

## License

The source code is distributed under the MIT License. The dataset is provided separately on Kaggle and may have its own terms.
