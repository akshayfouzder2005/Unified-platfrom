from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
import io
import csv
from ...core.database import get_db
from ...crud import taxonomy as taxonomy_crud
from ...schemas.taxonomy import TaxonomicUnitCreate, TaxonomicRankCreate

router = APIRouter()

@router.post("/taxonomy/upload-csv")
async def upload_taxonomy_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process a CSV file containing taxonomic data."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['scientific_name', 'rank', 'rank_level']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        results = {
            "processed_rows": 0,
            "created_ranks": 0,
            "created_species": 0,
            "errors": []
        }
        
        # Process each row
        for index, row in df.iterrows():
            try:
                # Create or get rank
                rank = taxonomy_crud.get_taxonomic_rank_by_name(db, row['rank'])
                if not rank:
                    rank_data = TaxonomicRankCreate(
                        name=row['rank'],
                        level=int(row['rank_level']),
                        description=row.get('rank_description', None)
                    )
                    rank = taxonomy_crud.create_taxonomic_rank(db, rank_data)
                    results["created_ranks"] += 1
                
                # Check if species already exists
                existing_species = taxonomy_crud.get_taxonomic_unit_by_name(db, row['scientific_name'])
                if not existing_species:
                    # Create species
                    species_data = TaxonomicUnitCreate(
                        scientific_name=row['scientific_name'],
                        common_name=row.get('common_name', None),
                        author=row.get('author', None),
                        year_described=int(row['year_described']) if pd.notna(row.get('year_described')) else None,
                        rank_id=rank.id,
                        parent_id=None,  # Could be enhanced to handle parent relationships
                        is_valid=bool(row.get('is_valid', True)),
                        is_marine=bool(row.get('is_marine', True)),
                        description=row.get('description', None)
                    )
                    taxonomy_crud.create_taxonomic_unit(db, species_data)
                    results["created_species"] += 1
                
                results["processed_rows"] += 1
                
            except Exception as e:
                results["errors"].append(f"Row {index + 1}: {str(e)}")
        
        return results
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/taxonomy/download-template")
async def download_taxonomy_template():
    """Download a CSV template for taxonomy data upload."""
    template_data = {
        'scientific_name': ['Gadus morhua', 'Salmo salar'],
        'common_name': ['Atlantic cod', 'Atlantic salmon'],
        'author': ['Linnaeus', 'Linnaeus'],
        'year_described': [1758, 1758],
        'rank': ['species', 'species'],
        'rank_level': [7, 7],
        'rank_description': ['Species level', 'Species level'],
        'is_valid': [True, True],
        'is_marine': [True, False],
        'description': ['Marine fish species', 'Anadromous fish species']
    }
    
    df = pd.DataFrame(template_data)
    
    # Convert to CSV string
    output = io.StringIO()
    df.to_csv(output, index=False)
    csv_content = output.getvalue()
    
    return {
        "filename": "taxonomy_template.csv",
        "content": csv_content,
        "instructions": {
            "scientific_name": "Required - Scientific name of the species",
            "common_name": "Optional - Common name",
            "author": "Optional - Taxonomic authority",
            "year_described": "Optional - Year species was described",
            "rank": "Required - Taxonomic rank (e.g., species, genus, family)",
            "rank_level": "Required - Numeric level (1=Kingdom, 2=Phylum, etc.)",
            "is_valid": "Optional - Whether this is a valid taxon (default: true)",
            "is_marine": "Optional - Whether this is a marine species (default: true)",
            "description": "Optional - Additional description"
        }
    }

@router.get("/taxonomy/validate-csv")
async def validate_taxonomy_csv(
    file: UploadFile = File(...)
):
    """Validate a taxonomy CSV file without importing the data."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        validation_results = {
            "total_rows": len(df),
            "valid_rows": 0,
            "errors": [],
            "warnings": [],
            "column_info": {}
        }
        
        # Check columns
        required_columns = ['scientific_name', 'rank', 'rank_level']
        optional_columns = ['common_name', 'author', 'year_described', 'is_valid', 'is_marine', 'description']
        
        for col in required_columns:
            if col not in df.columns:
                validation_results["errors"].append(f"Missing required column: {col}")
            else:
                validation_results["column_info"][col] = {
                    "present": True,
                    "non_null_count": df[col].notna().sum(),
                    "null_count": df[col].isna().sum()
                }
        
        for col in optional_columns:
            if col in df.columns:
                validation_results["column_info"][col] = {
                    "present": True,
                    "non_null_count": df[col].notna().sum(),
                    "null_count": df[col].isna().sum()
                }
        
        # Validate data types and values
        for index, row in df.iterrows():
            row_errors = []
            
            if pd.isna(row.get('scientific_name')) or row.get('scientific_name', '').strip() == '':
                row_errors.append("scientific_name is required")
            
            if pd.isna(row.get('rank')) or row.get('rank', '').strip() == '':
                row_errors.append("rank is required")
            
            if pd.isna(row.get('rank_level')):
                row_errors.append("rank_level is required")
            else:
                try:
                    level = int(row['rank_level'])
                    if level < 1:
                        row_errors.append("rank_level must be positive")
                except (ValueError, TypeError):
                    row_errors.append("rank_level must be a number")
            
            # Check year_described if present
            if pd.notna(row.get('year_described')):
                try:
                    year = int(row['year_described'])
                    if year < 1758 or year > 2100:
                        row_errors.append("year_described should be between 1758 and 2100")
                except (ValueError, TypeError):
                    row_errors.append("year_described must be a number")
            
            if row_errors:
                validation_results["errors"].append(f"Row {index + 1}: {', '.join(row_errors)}")
            else:
                validation_results["valid_rows"] += 1
        
        # Add warnings
        if validation_results["valid_rows"] < validation_results["total_rows"]:
            validation_results["warnings"].append(
                f"{validation_results['total_rows'] - validation_results['valid_rows']} rows have validation errors"
            )
        
        return validation_results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error validating file: {str(e)}")

@router.get("/ingestion/status")
def get_ingestion_status():
    """Get status of data ingestion operations."""
    return {
        "supported_formats": ["CSV"],
        "supported_data_types": ["taxonomy", "oceanographic", "edna", "otolith"],
        "max_file_size": "10MB",
        "active_jobs": [],
        "recent_imports": []
    }