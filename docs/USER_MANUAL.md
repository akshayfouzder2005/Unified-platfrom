# User Manual (Prototype)

## Running the Platform
- Docker: `docker compose up --build`
- Local dev: run FastAPI via `uvicorn app.main:app --reload` from `backend/`

## Exploring the API
- Health: http://localhost:8000/api/health
- Docs: http://localhost:8000/docs

## Visualizations
- Open `client/public/index.html` in your browser to see a demo chart and API connectivity.

## Roadmap
- Replace static client with a full SPA
- Add authentication and role-based access control
- Implement real ingestion connectors and persistence
- Add tests, linters, and CI gates
