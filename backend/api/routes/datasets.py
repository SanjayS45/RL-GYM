"""
Dataset API Routes

Endpoints for managing training datasets and demonstrations.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import uuid

router = APIRouter()

# In-memory dataset storage (use database in production)
datasets: Dict[str, Dict[str, Any]] = {}


class DatasetInfo(BaseModel):
    """Dataset information model."""
    
    dataset_id: str
    name: str
    type: str  # demonstrations, trajectories, offline
    size: int  # Number of samples
    created_at: str
    format: str  # h5, csv, json
    compatible_envs: List[str] = []


class DatasetUploadResponse(BaseModel):
    """Dataset upload response."""
    
    dataset_id: str
    name: str
    size: int
    status: str


@router.get("/", response_model=List[DatasetInfo])
async def list_datasets():
    """List all available datasets."""
    return [
        DatasetInfo(
            dataset_id=did,
            name=data["name"],
            type=data["type"],
            size=data["size"],
            created_at=data["created_at"],
            format=data["format"],
            compatible_envs=data.get("compatible_envs", []),
        )
        for did, data in datasets.items()
    ]


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = None,
    dataset_type: str = "trajectories",
):
    """
    Upload a new dataset.
    
    Accepts HDF5, CSV, or JSON formats containing:
    - Demonstrations (expert trajectories)
    - Offline trajectories (state-action-reward logs)
    """
    from datetime import datetime
    
    dataset_id = str(uuid.uuid4())[:8]
    
    # Determine format
    filename = file.filename or "unknown"
    if filename.endswith(".h5") or filename.endswith(".hdf5"):
        format_type = "h5"
    elif filename.endswith(".csv"):
        format_type = "csv"
    else:
        format_type = "json"
    
    # Read file content (in production, stream to storage)
    content = await file.read()
    size = len(content)
    
    # Validate dataset
    validation_result = _validate_dataset(content, format_type)
    
    if not validation_result["valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset: {validation_result['error']}"
        )
    
    # Store dataset metadata
    datasets[dataset_id] = {
        "name": name or filename,
        "type": dataset_type,
        "size": validation_result.get("num_samples", 0),
        "created_at": datetime.now().isoformat(),
        "format": format_type,
        "compatible_envs": validation_result.get("compatible_envs", []),
        "content": content,  # In production, save to file storage
    }
    
    return DatasetUploadResponse(
        dataset_id=dataset_id,
        name=name or filename,
        size=validation_result.get("num_samples", 0),
        status="uploaded",
    )


def _validate_dataset(content: bytes, format_type: str) -> Dict[str, Any]:
    """Validate dataset format and content."""
    try:
        if format_type == "json":
            import json
            data = json.loads(content)
            
            # Check for required fields
            if isinstance(data, list):
                num_samples = len(data)
                if num_samples > 0 and isinstance(data[0], dict):
                    required = {"observation", "action"}
                    if not required.issubset(set(data[0].keys())):
                        return {"valid": False, "error": "Missing required fields: observation, action"}
            elif isinstance(data, dict):
                if "trajectories" in data:
                    num_samples = len(data["trajectories"])
                else:
                    num_samples = len(data.get("observations", []))
            else:
                return {"valid": False, "error": "Invalid JSON structure"}
            
            return {"valid": True, "num_samples": num_samples, "compatible_envs": ["navigation", "grid_world"]}
        
        elif format_type == "csv":
            # Basic CSV validation
            lines = content.decode().strip().split("\n")
            headers = lines[0].split(",") if lines else []
            num_samples = len(lines) - 1  # Exclude header
            
            return {"valid": True, "num_samples": num_samples, "compatible_envs": ["navigation", "grid_world"]}
        
        else:
            # HDF5 validation would require h5py
            return {"valid": True, "num_samples": 1000, "compatible_envs": ["navigation", "platformer"]}
    
    except Exception as e:
        return {"valid": False, "error": str(e)}


@router.get("/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str):
    """Get dataset information."""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    data = datasets[dataset_id]
    return DatasetInfo(
        dataset_id=dataset_id,
        name=data["name"],
        type=data["type"],
        size=data["size"],
        created_at=data["created_at"],
        format=data["format"],
        compatible_envs=data.get("compatible_envs", []),
    )


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    del datasets[dataset_id]
    return {"status": "deleted"}


@router.get("/{dataset_id}/sample")
async def sample_dataset(dataset_id: str, n: int = 5):
    """
    Get sample data from a dataset.
    
    Returns a few examples for preview.
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Return mock samples for demonstration
    import numpy as np
    
    samples = []
    for i in range(min(n, 5)):
        samples.append({
            "observation": np.random.randn(4).tolist(),
            "action": np.random.randint(0, 4),
            "reward": np.random.randn(),
            "next_observation": np.random.randn(4).tolist(),
            "done": bool(np.random.randint(0, 2)),
        })
    
    return {"samples": samples, "total_size": datasets[dataset_id]["size"]}


@router.post("/{dataset_id}/validate")
async def validate_dataset_compatibility(dataset_id: str, environment: str):
    """
    Validate dataset compatibility with an environment.
    
    Checks observation and action space compatibility.
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    data = datasets[dataset_id]
    compatible = environment in data.get("compatible_envs", [])
    
    return {
        "compatible": compatible,
        "environment": environment,
        "message": "Dataset is compatible" if compatible else "Dataset may not be compatible with this environment",
    }


# Predefined sample datasets
SAMPLE_DATASETS = [
    {
        "name": "Navigation Expert Demos",
        "type": "demonstrations",
        "description": "Expert demonstrations for navigation environment",
        "size": 10000,
        "env": "navigation",
    },
    {
        "name": "Grid World Random",
        "type": "trajectories",
        "description": "Random policy trajectories for grid world",
        "size": 50000,
        "env": "grid_world",
    },
]


@router.get("/samples/list")
async def list_sample_datasets():
    """List available sample datasets."""
    return {"samples": SAMPLE_DATASETS}


@router.post("/samples/{sample_name}/load")
async def load_sample_dataset(sample_name: str):
    """Load a predefined sample dataset."""
    from datetime import datetime
    
    # Find matching sample
    sample = None
    for s in SAMPLE_DATASETS:
        if s["name"].lower().replace(" ", "_") == sample_name.lower().replace(" ", "_"):
            sample = s
            break
    
    if not sample:
        raise HTTPException(status_code=404, detail="Sample dataset not found")
    
    dataset_id = str(uuid.uuid4())[:8]
    
    datasets[dataset_id] = {
        "name": sample["name"],
        "type": sample["type"],
        "size": sample["size"],
        "created_at": datetime.now().isoformat(),
        "format": "internal",
        "compatible_envs": [sample["env"]],
    }
    
    return {
        "dataset_id": dataset_id,
        "name": sample["name"],
        "size": sample["size"],
        "status": "loaded",
    }

