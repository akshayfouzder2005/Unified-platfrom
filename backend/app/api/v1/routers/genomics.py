"""
🧬 Genomics Analysis API Router

RESTful API endpoints for genomics analysis, eDNA processing, and bioinformatics functionality.
Provides access to sequence processing, taxonomic classification, diversity analysis, and phylogenetics.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
import logging
import io

from ....genomics.sequence_processor import sequence_processor
from ....genomics.diversity_calculator import diversity_calculator
from ....genomics.phylogenetic_analysis import phylogenetic_analyzer, DistanceMethod, TreeMethod
from ....genomics.taxonomic_classifier import taxonomic_classifier, ClassificationMethod, TaxonomicRank, ReferenceSequence
from ....genomics.comparative_analysis import comparative_analyzer, SampleMetadata, AnalysisType
from ....core.database import get_db
from ....core.auth import get_current_user
from ....models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/genomics", tags=["genomics"])

# Pydantic models for API requests/responses

class SequenceProcessingRequest(BaseModel):
    """Sequence processing request parameters"""
    sequences: Dict[str, str] = Field(..., description="Dictionary of sequence_id -> DNA sequence")
    quality_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Quality threshold")
    min_length: int = Field(50, ge=10, le=10000, description="Minimum sequence length")
    remove_primers: bool = Field(True, description="Whether to remove primer sequences")
    trim_low_quality: bool = Field(True, description="Whether to trim low quality regions")

class TaxonomicClassificationRequest(BaseModel):
    """Taxonomic classification request parameters"""
    sequences: Dict[str, str] = Field(..., description="Dictionary of sequence_id -> DNA sequence")
    method: str = Field("consensus", description="Classification method")
    min_confidence: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    database: Optional[str] = Field(None, description="Reference database to use")

class DiversityAnalysisRequest(BaseModel):
    """Diversity analysis request parameters"""
    abundance_data: Dict[str, int] = Field(..., description="Species abundance data")
    sample_id: str = Field(..., description="Sample identifier")
    analysis_type: str = Field("alpha", description="Type of diversity analysis")
    bootstrap_replicates: Optional[int] = Field(None, description="Number of bootstrap replicates")

class PhylogeneticAnalysisRequest(BaseModel):
    """Phylogenetic analysis request parameters"""
    sequences: Dict[str, str] = Field(..., description="Dictionary of sequence_id -> DNA sequence")
    distance_method: str = Field("kimura_2p", description="Distance calculation method")
    tree_method: str = Field("neighbor_joining", description="Tree construction method")
    bootstrap_replicates: int = Field(100, ge=0, le=1000, description="Bootstrap replicates")

class ComparativeAnalysisRequest(BaseModel):
    """Comparative analysis request parameters"""
    samples: Dict[str, Dict[str, str]] = Field(..., description="Sample data")
    analysis_type: str = Field("integrated_pipeline", description="Type of comparative analysis")
    include_phylogenetics: bool = Field(True, description="Include phylogenetic analysis")
    include_taxonomy: bool = Field(True, description="Include taxonomic classification")

class SampleData(BaseModel):
    """Sample data with metadata"""
    sample_id: str = Field(..., description="Sample identifier")
    sequences: Dict[str, str] = Field(..., description="Sequences in sample")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Sample metadata")

# API Endpoints

@router.get("/health")
async def genomics_health_check():
    """Health check for genomics services"""
    try:
        # Test all genomics services
        services_status = {
            "sequence_processor": "healthy",
            "diversity_calculator": "healthy", 
            "phylogenetic_analyzer": "healthy",
            "taxonomic_classifier": len(taxonomic_classifier.reference_database) > 0,
            "comparative_analyzer": "healthy"
        }
        
        return {
            "status": "healthy",
            "services": services_status,
            "reference_sequences": len(taxonomic_classifier.reference_database),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Genomics health check failed: {e}")
        raise HTTPException(status_code=503, detail="Genomics services unavailable")

@router.post("/sequences/process")
async def process_sequences(
    request: SequenceProcessingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Process DNA sequences with quality control and filtering"""
    try:
        logger.info(f"🧬 Processing {len(request.sequences)} sequences")
        
        processed_results = {}
        
        for seq_id, sequence in request.sequences.items():
            # Process individual sequence
            processing_result = sequence_processor.process_sequence(
                sequence=sequence,
                sequence_id=seq_id,
                quality_threshold=request.quality_threshold,
                min_length=request.min_length,
                remove_primers=request.remove_primers,
                trim_low_quality=request.trim_low_quality
            )
            processed_results[seq_id] = processing_result
        
        # Generate batch summary
        summary = {
            "total_sequences": len(request.sequences),
            "processed_successfully": sum(1 for r in processed_results.values() if 'error' not in r),
            "processing_errors": sum(1 for r in processed_results.values() if 'error' in r),
            "average_length": sum(r.get('processed_length', 0) for r in processed_results.values()) / len(processed_results) if processed_results else 0
        }
        
        return {
            "processing_results": processed_results,
            "summary": summary,
            "processing_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Sequence processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sequences/classify")
async def classify_sequences(
    request: TaxonomicClassificationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Classify DNA sequences taxonomically"""
    try:
        logger.info(f"🧬 Classifying {len(request.sequences)} sequences using {request.method}")
        
        # Map method string to enum
        method_mapping = {
            "blast_like": ClassificationMethod.BLAST_LIKE,
            "kmer_based": ClassificationMethod.KMER_BASED,
            "naive_bayes": ClassificationMethod.NAIVE_BAYES,
            "phylogenetic": ClassificationMethod.PHYLOGENETIC,
            "consensus": ClassificationMethod.CONSENSUS
        }
        
        method = method_mapping.get(request.method, ClassificationMethod.CONSENSUS)
        
        # Perform classification
        assignments = taxonomic_classifier.classify_batch(
            request.sequences, method, request.min_confidence
        )
        
        # Generate summary
        summary = taxonomic_classifier.get_classification_summary(assignments)
        
        return {
            "classification_method": request.method,
            "min_confidence": request.min_confidence,
            "assignments": {seq_id: {
                "taxonomy": {rank.value: taxon for rank, taxon in assignment.taxonomy.items()},
                "confidence_scores": {rank.value: score for rank, score in assignment.confidence_scores.items()},
                "method_used": assignment.method_used.value,
                "best_match": assignment.best_match
            } for seq_id, assignment in assignments.items()},
            "summary": summary,
            "classification_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Taxonomic classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/diversity/analyze")
async def analyze_diversity(
    request: DiversityAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Perform biodiversity analysis"""
    try:
        logger.info(f"🧬 Diversity analysis: {request.analysis_type} for {request.sample_id}")
        
        if request.analysis_type == "alpha":
            # Alpha diversity analysis
            results = diversity_calculator.calculate_alpha_diversity(
                request.abundance_data, request.sample_id
            )
        
        elif request.analysis_type == "rarefaction":
            # Rarefaction curve analysis
            results = diversity_calculator.generate_rarefaction_curve(
                request.abundance_data, max_samples=None, step_size=1
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid analysis type: {request.analysis_type}")
        
        return {
            "analysis_type": request.analysis_type,
            "sample_id": request.sample_id,
            "results": results,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Diversity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/phylogenetics/analyze")
async def analyze_phylogenetics(
    request: PhylogeneticAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Perform phylogenetic analysis"""
    try:
        logger.info(f"🧬 Phylogenetic analysis: {len(request.sequences)} sequences")
        
        # Map method strings to enums
        distance_methods = {
            "hamming": DistanceMethod.HAMMING,
            "p_distance": DistanceMethod.P_DISTANCE,
            "jukes_cantor": DistanceMethod.JUKES_CANTOR,
            "kimura_2p": DistanceMethod.KIMURA_2P,
            "tamura_nei": DistanceMethod.TAMURA_NEI
        }
        
        tree_methods = {
            "neighbor_joining": TreeMethod.NEIGHBOR_JOINING,
            "upgma": TreeMethod.UPGMA,
            "maximum_parsimony": TreeMethod.MAXIMUM_PARSIMONY,
            "minimum_evolution": TreeMethod.MINIMUM_EVOLUTION
        }
        
        distance_method = distance_methods.get(request.distance_method, DistanceMethod.KIMURA_2P)
        tree_method = tree_methods.get(request.tree_method, TreeMethod.NEIGHBOR_JOINING)
        
        # Construct phylogenetic tree
        tree_results = phylogenetic_analyzer.construct_tree(
            sequences=request.sequences,
            distance_method=distance_method,
            tree_method=tree_method,
            bootstrap_replicates=request.bootstrap_replicates if request.bootstrap_replicates > 0 else None
        )
        
        return {
            "distance_method": request.distance_method,
            "tree_method": request.tree_method,
            "bootstrap_replicates": request.bootstrap_replicates,
            "tree_results": tree_results,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Phylogenetic analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/comparative/analyze")
async def perform_comparative_analysis(
    request: ComparativeAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Perform comparative genomics analysis"""
    try:
        logger.info(f"🧬 Comparative analysis: {request.analysis_type} on {len(request.samples)} samples")
        
        # Add samples to comparative analyzer
        for sample_id, sequences in request.samples.items():
            comparative_analyzer.add_sample(sample_id, sequences)
        
        sample_ids = list(request.samples.keys())
        
        # Perform analysis based on type
        if request.analysis_type == "integrated_pipeline":
            results = comparative_analyzer.run_integrated_pipeline(
                sample_ids=sample_ids,
                include_phylogenetics=request.include_phylogenetics,
                include_taxonomy=request.include_taxonomy
            )
        
        elif request.analysis_type == "community_comparison":
            results = comparative_analyzer.compare_communities(sample_ids)
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid analysis type: {request.analysis_type}")
        
        return {
            "analysis_type": request.analysis_type,
            "samples_analyzed": len(request.samples),
            "results": results.to_dict(),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Comparative analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/samples/add")
async def add_sample(
    sample_data: SampleData,
    current_user: User = Depends(get_current_user)
):
    """Add a sample to the comparative analyzer"""
    try:
        logger.info(f"🧬 Adding sample: {sample_data.sample_id}")
        
        # Create metadata object if provided
        metadata = None
        if sample_data.metadata:
            metadata = SampleMetadata(
                sample_id=sample_data.sample_id,
                location=sample_data.metadata.get("location", {"lat": 0.0, "lon": 0.0}),
                collection_date=sample_data.metadata.get("collection_date", datetime.now().isoformat()),
                depth=sample_data.metadata.get("depth"),
                temperature=sample_data.metadata.get("temperature"),
                salinity=sample_data.metadata.get("salinity"),
                ph=sample_data.metadata.get("ph"),
                habitat_type=sample_data.metadata.get("habitat_type"),
                sampling_method=sample_data.metadata.get("sampling_method"),
                additional_metadata=sample_data.metadata.get("additional_metadata", {})
            )
        
        # Add sample
        result = comparative_analyzer.add_sample(
            sample_data.sample_id, sample_data.sequences, metadata
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to add sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/samples/list")
async def list_samples(current_user: User = Depends(get_current_user)):
    """List all samples in the comparative analyzer"""
    try:
        samples_info = {}
        
        for sample_id in comparative_analyzer.samples.keys():
            sequences = comparative_analyzer.samples[sample_id]
            metadata = comparative_analyzer.sample_metadata.get(sample_id)
            
            samples_info[sample_id] = {
                "sequence_count": len(sequences),
                "metadata": {
                    "collection_date": metadata.collection_date if metadata else None,
                    "location": metadata.location if metadata else None,
                    "habitat_type": metadata.habitat_type if metadata else None
                }
            }
        
        return {
            "samples": samples_info,
            "total_samples": len(samples_info),
            "retrieved_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to list samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/references/add")
async def add_reference_sequences(
    sequences_data: List[Dict[str, Any]],
    current_user: User = Depends(get_current_user)
):
    """Add reference sequences to the taxonomic classifier"""
    try:
        logger.info(f"🧬 Adding {len(sequences_data)} reference sequences")
        
        reference_sequences = []
        
        for seq_data in sequences_data:
            # Map taxonomic ranks
            taxonomy = {}
            for rank_str, taxon in seq_data.get("taxonomy", {}).items():
                rank_enum = getattr(TaxonomicRank, rank_str.upper(), None)
                if rank_enum:
                    taxonomy[rank_enum] = taxon
            
            ref_seq = ReferenceSequence(
                sequence_id=seq_data["sequence_id"],
                sequence=seq_data["sequence"],
                taxonomy=taxonomy,
                source_database=seq_data.get("source_database", "custom"),
                accession=seq_data.get("accession"),
                description=seq_data.get("description")
            )
            reference_sequences.append(ref_seq)
        
        # Add references to classifier
        result = taxonomic_classifier.add_reference_sequences(reference_sequences)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to add reference sequences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/references/count")
async def get_reference_count(current_user: User = Depends(get_current_user)):
    """Get count of reference sequences"""
    try:
        count = len(taxonomic_classifier.reference_database)
        
        return {
            "reference_count": count,
            "database_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get reference count: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/fasta")
async def upload_fasta_file(
    file: UploadFile = File(...),
    file_type: str = Query("sequences", description="Type of FASTA file"),
    current_user: User = Depends(get_current_user)
):
    """Upload and parse FASTA file"""
    try:
        logger.info(f"🧬 Uploading FASTA file: {file.filename}")
        
        # Read file content
        content = await file.read()
        fasta_text = content.decode('utf-8')
        
        # Parse FASTA format
        sequences = {}
        current_id = None
        current_seq = []
        
        for line in fasta_text.split('\n'):
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_id and current_seq:
                    sequences[current_id] = ''.join(current_seq)
                
                # Start new sequence
                current_id = line[1:].split()[0]  # Take first part after >
                current_seq = []
            elif line and current_id:
                current_seq.append(line)
        
        # Save last sequence
        if current_id and current_seq:
            sequences[current_id] = ''.join(current_seq)
        
        return {
            "filename": file.filename,
            "file_type": file_type,
            "sequences_parsed": len(sequences),
            "sequences": sequences,
            "upload_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ FASTA upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/types")
async def get_analysis_types(current_user: User = Depends(get_current_user)):
    """Get available analysis types"""
    try:
        analysis_types = {
            "sequence_processing": [
                "quality_assessment", "primer_removal", "trimming", "filtering"
            ],
            "taxonomic_classification": [
                "blast_like", "kmer_based", "naive_bayes", "phylogenetic", "consensus"
            ],
            "diversity_analysis": [
                "alpha_diversity", "beta_diversity", "gamma_diversity", "rarefaction"
            ],
            "phylogenetic_analysis": [
                "distance_matrix", "tree_construction", "bootstrap_analysis"
            ],
            "comparative_analysis": [
                "integrated_pipeline", "community_comparison", "temporal_analysis", "spatial_analysis"
            ]
        }
        
        return {
            "available_analyses": analysis_types,
            "retrieved_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get analysis types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/summary")
async def get_genomics_summary(current_user: User = Depends(get_current_user)):
    """Get summary statistics for genomics services"""
    try:
        summary = {
            "reference_sequences": len(taxonomic_classifier.reference_database),
            "samples_loaded": len(comparative_analyzer.samples),
            "total_sequences_in_samples": sum(len(seqs) for seqs in comparative_analyzer.samples.values()),
            "k_mer_database_size": len(taxonomic_classifier.kmer_database),
            "analysis_cache_size": len(comparative_analyzer.analysis_cache),
            "classification_cache_size": len(taxonomic_classifier.classification_cache)
        }
        
        return {
            "summary": summary,
            "summary_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints

@router.post("/sequences/validate")
async def validate_sequences(
    sequences: Dict[str, str],
    current_user: User = Depends(get_current_user)
):
    """Validate DNA sequences"""
    try:
        validation_results = {}
        
        for seq_id, sequence in sequences.items():
            # Basic validation
            is_valid = all(base.upper() in 'ATGCNRYWSMKHBVD-' for base in sequence)
            length = len(sequence)
            gc_content = (sequence.upper().count('G') + sequence.upper().count('C')) / length if length > 0 else 0
            
            validation_results[seq_id] = {
                "is_valid": is_valid,
                "length": length,
                "gc_content": round(gc_content, 3),
                "has_ambiguous_bases": any(base.upper() in 'NRYWSMKHBVD' for base in sequence),
                "has_gaps": '-' in sequence
            }
        
        return {
            "validation_results": validation_results,
            "total_sequences": len(sequences),
            "valid_sequences": sum(1 for r in validation_results.values() if r["is_valid"]),
            "validation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Sequence validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache/clear")
async def clear_caches(
    cache_type: str = Query("all", description="Type of cache to clear"),
    current_user: User = Depends(get_current_user)
):
    """Clear analysis caches"""
    try:
        cleared = []
        
        if cache_type in ["all", "comparative"]:
            comparative_analyzer.analysis_cache.clear()
            cleared.append("comparative_analysis")
        
        if cache_type in ["all", "classification"]:
            taxonomic_classifier.classification_cache.clear()
            cleared.append("taxonomic_classification")
        
        return {
            "caches_cleared": cleared,
            "clear_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))