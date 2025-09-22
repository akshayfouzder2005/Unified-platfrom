# Architecture Overview

This prototype is structured as a service-oriented backend (FastAPI) with modular domain packages and ingestion pipelines. It includes a simple static client and containerized deployment via Docker.

## Components
- API (FastAPI): modular routers for taxonomy, otolith, eDNA, and visualization
- Ingestion Pipelines: per-domain ingestion scaffolding
- Database: Postgres (optional) for persistence and analytics
- Client: static placeholder page to validate API connectivity

## Data Domains
- Oceanographic (e.g., SST, salinity, currents)
- Fisheries (e.g., catch records, stock assessments)
- Molecular (eDNA; taxa detection, abundance proxies)

## Scaling and Cloud Readiness
- Stateless API containers behind a load balancer
- Asynchronous ingestion jobs (future: Celery/RQ/Arq/Kafka)
- Observability: logs/metrics/traces (future)
- Infra as code (future: Terraform/Azure/Bicep/CloudFormation)
