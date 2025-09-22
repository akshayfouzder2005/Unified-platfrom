# Data Models (Draft)

This document will consolidate key models spanning oceanographic, fisheries, and eDNA domains.

## Oceanographic
- Observation: time, location, variable, value, unit, source

## Fisheries
- CatchRecord: species, date, location, gear, weight, count, unit

## eDNA
- Sample: sample_id, location, date, protocol, taxa_detections
- TaxaDetection: taxon, score, confidence, read_count

## Cross-cutting
- Taxon: scientific_name, rank, parent, identifiers
