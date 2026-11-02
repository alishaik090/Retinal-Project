# Retina Inference Service

An API-only, deployable service for the existing retinal vessel-segmentation checkpoint. It contains no UI, HTML, Streamlit application, dashboard, static site, or client-side visualization.

> Research use only. Output is not a diagnosis or a clinical decision.

## What is preserved

- The trained checkpoint architecture and its inference preprocessing: RGB decoding, `cv2.INTER_AREA` resize, `/255` tensor scaling, ImageNet normalization inside the saved `ResNet18UNet`, sigmoid, and binary thresholding.
- The saved 256px DRIVE evaluation protocol and FOV-aware metrics in `evaluation/segmentation_metrics.json`.
- The recorded held-out result: Dice `0.7217`, IoU `0.5646`, precision `0.7541`, recall `0.6957`, specificity `0.9652`, and pixel accuracy `0.9277`.

The checkpoint is intentionally not committed. Put the existing `vessel_unet_best.pt` in `app/model/weights/`, or set an absolute `MODEL_PATH` in `.env`.
The detailed [model card](docs/MODEL_CARD.md) remains with the service.

## Layout

```text
app/
  api/           # /health and /predict routes plus schemas
  config/        # environment settings
  inference/     # unchanged preprocessing, thresholding, mask encoding
  model/         # checkpoint-compatible architectures and single-load loader
  utils/         # logging
evaluation/      # metrics and offline DRIVE verification
main.py          # ASGI application
```

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Place app\model\weights\vessel_unet_best.pt, or update MODEL_PATH in .env
uvicorn main:app --host 0.0.0.0 --port 8000
```

`GET /health` returns `{"status":"healthy","model":"loaded"}` only after a compatible checkpoint is loaded. With no checkpoint it returns a non-crashing `degraded` status and explanatory detail.

## API

### `POST /predict`

Send `multipart/form-data` with one `image` field (JPEG, PNG, TIFF, BMP, or WebP; default maximum 20 MB). The service never reloads weights per request.

```json
{
  "prediction": "retinal_vessel_segmentation",
  "confidence": 0.42,
  "segmentation_mask": "base64 PNG",
  "processing_time": "0.430 sec",
  "model_name": "ResNet18UNet",
  "threshold": 0.65,
  "vessel_pixel_fraction": 0.18
}
```

`confidence` is the mean vessel probability, not a calibrated disease confidence. `segmentation_mask` is a PNG encoded as base64.

## Verify saved metrics without retraining

Provide the original labelled DRIVE training directory and the existing checkpoint:

```powershell
python -m evaluation.verify_segmentation `
  --model-path C:\path\to\vessel_unet_best.pt `
  --drive-dir C:\path\to\DRIVE `
  --metrics-path evaluation\segmentation_metrics.json
```

This re-runs inference only on the preserved held-out IDs and prints Dice, IoU, and related metrics. It does not train or alter weights.

## Docker

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Mount or copy the checkpoint into `app/model/weights/` before deployment. For production, place the service behind HTTPS and add authentication, request-rate controls, audit logging, retention policies, and service-specific clinical governance.

## Integration

This service is intended to be called by the separate Multi-Organ Medical Platform via `POST /predict`. The platform should treat unavailable health as a non-routable dependency and display the service response without inventing a medical result.
