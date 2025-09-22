# AI-Driven Unified Data Platform for Oceanographic, Fisheries, and Molecular Biodiversity Insights

A robust, cloud-ready web platform prototype for unifying oceanographic, fisheries, and molecular biodiversity (eDNA) data. It provides modular ingestion pipelines, APIs for data access and management, and lightweight visualization scaffolding to explore trends.

## Key Capabilities
- Scalable backend architecture (FastAPI) with modular data ingestion pipelines
- Visualization scaffolding for oceanographic and biodiversity trends
- Integrated modules for taxonomy, otolith morphology, and eDNA data
- Well-documented APIs and user manuals for future adoption and scaling
- Containerized runtime with Docker and optional Postgres

## Repository Structure
- `backend/` – FastAPI app, modules, ingestion, tests, Dockerfile
- `client/` – Static client placeholder (can evolve into React/Next/Vite)
- `docs/` – Architecture, API, data models, and user manual
- `.github/workflows/` – CI stub

## Quickstart

### Option A: Run with Docker (recommended)
1. Copy environment template and set DB credentials:
   - Windows (PowerShell): `Copy-Item .env.example .env`
   - Linux/macOS: `cp .env.example .env`
2. Start Postgres and API:
   - `docker compose up --build`
3. Open API docs: http://localhost:8000/docs
4. Open client placeholder: open `client/public/index.html` in a browser (it calls the API health endpoint).

To stop: `docker compose down`

### Option B: Local development (no containers)
1. Python 3.11+
2. From `backend/`:
   - Create a virtual env: `python -m venv .venv`
   - Activate it:
     - PowerShell: `.venv\Scripts\Activate.ps1`
     - bash/zsh: `source .venv/bin/activate`
   - Install deps: `pip install -r requirements.txt`
   - Run: `uvicorn app.main:app --reload`
3. API runs at http://localhost:8000 and docs at http://localhost:8000/docs

## Next Steps
- Replace the static client with a full SPA (e.g., React + Vite/Next.js)
- Implement database models and migrations (SQLAlchemy + Alembic)
- Fill ingestion pipelines with actual connectors/parsers
- Add auth, RBAC, and multi-tenant data boundaries as required
