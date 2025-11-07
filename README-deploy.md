# ai-invoice-data-extractor - Final Deployable Monorepo

This monorepo contains:
- backend/extractor_agent : FastAPI service (interpret_invoice) with OCR + HS ranking + BigQuery writer
- frontend : Vite React app (production containerized with Nginx). The frontend **hardcodes** the backend URL; replace the placeholder in src/App.jsx
- cloudbuild.yaml : build both images with Cloud Build

## Frontend hardcoded backend URL
In `frontend/src/App.jsx` the BACKEND_URL is set to:
`https://extractor-REPLACE_WITH_YOUR_URL.a.run.app`
Replace this with your actual Cloud Run extractor URL before building the frontend image.

## Build & Deploy (recommended)
1. Ensure APIs enabled:
   ```bash
   gcloud services enable run.googleapis.com aiplatform.googleapis.com bigquery.googleapis.com cloudbuild.googleapis.com
   ```
2. Create BigQuery dataset (once):
   ```bash
   bq --location=us-central1 mk --dataset ai-invoice-data-extractor:crossbordersense
   ```
3. Build & push images via Cloud Build:
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   ```
4. Deploy to Cloud Run:
   ```bash
   gcloud run deploy extractor --image gcr.io/ai-invoice-data-extractor/extractor --region us-central1 --allow-unauthenticated
   gcloud run deploy cb-frontend --image gcr.io/ai-invoice-data-extractor/cb-frontend --region us-central1 --allow-unauthenticated
   ```

## Notes
- Frontend currently serves static site from Nginx and expects backend to be accessible over HTTPS.
- Ensure the Cloud Run extractor service account has `roles/bigquery.dataEditor` and `roles/aiplatform.user`.
