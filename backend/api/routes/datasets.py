"""Dataset routes for RL-GYM API."""
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import os
import json
import uuid
from datetime import datetime

from datasets import DatasetLoader, DatasetValidator, DatasetPreprocessor
from config import Config

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Initialize components
config = Config()
dataset_loader = DatasetLoader(config.data_dir)
dataset_validator = DatasetValidator()
dataset_preprocessor = DatasetPreprocessor()

# Store dataset metadata
_datasets: Dict[str, Dict[str, Any]] = {}


class DatasetMetadata(BaseModel):
    """Dataset metadata model."""
    id: str
    name: str
    type: str
    size: int
    num_trajectories: Optional[int] = None
    num_transitions: Optional[int] = None
    created_at: str
    environment: Optional[str] = None
    description: Optional[str] = None


class DatasetUploadResponse(BaseModel):
    """Dataset upload response model."""
    id: str
    status: str
    message: str
    validation_results: Optional[Dict[str, Any]] = None


@router.get("/list")
async def list_datasets():
    """List all available datasets."""
    datasets = []
    
    # Add stored datasets
    for dataset_id, metadata in _datasets.items():
        datasets.append(metadata)
    
    # Also check the data directory for any existing files
    data_dir = config.data_dir
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath) and filename.endswith(('.json', '.npz', '.pkl')):
                if filename not in [d.get('filename') for d in _datasets.values()]:
                    datasets.append({
                        "id": filename.rsplit('.', 1)[0],
                        "name": filename,
                        "type": "file",
                        "size": os.path.getsize(filepath),
                        "created_at": datetime.fromtimestamp(
                            os.path.getctime(filepath)
                        ).isoformat()
                    })
    
    return {"datasets": datasets}


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    dataset_type: str = Form("trajectories"),
    environment: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """Upload a new dataset."""
    try:
        # Generate dataset ID
        dataset_id = str(uuid.uuid4())[:8]
        
        # Read file contents
        contents = await file.read()
        
        # Determine file format
        filename = file.filename or "dataset"
        file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'json'
        
        # Save to data directory
        data_dir = config.data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        save_filename = f"{dataset_id}.{file_ext}"
        save_path = os.path.join(data_dir, save_filename)
        
        with open(save_path, 'wb') as f:
            f.write(contents)
        
        # Validate dataset
        validation_results = None
        try:
            # Load and validate
            if file_ext == 'json':
                data = json.loads(contents.decode('utf-8'))
                validation_results = dataset_validator.validate_trajectories(data)
            else:
                validation_results = {"status": "loaded", "format": file_ext}
        except Exception as e:
            validation_results = {"status": "warning", "message": str(e)}
        
        # Calculate metadata
        num_trajectories = None
        num_transitions = None
        
        if file_ext == 'json':
            try:
                data = json.loads(contents.decode('utf-8'))
                if isinstance(data, list):
                    num_trajectories = len(data)
                    num_transitions = sum(
                        len(traj.get('transitions', [])) 
                        for traj in data if isinstance(traj, dict)
                    )
            except:
                pass
        
        # Store metadata
        metadata = {
            "id": dataset_id,
            "name": name or filename,
            "filename": save_filename,
            "type": dataset_type,
            "size": len(contents),
            "num_trajectories": num_trajectories,
            "num_transitions": num_transitions,
            "environment": environment,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        _datasets[dataset_id] = metadata
        
        return DatasetUploadResponse(
            id=dataset_id,
            status="success",
            message=f"Dataset '{name or filename}' uploaded successfully",
            validation_results=validation_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset metadata."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return _datasets[dataset_id]


@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, limit: int = 10):
    """Preview dataset contents."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    metadata = _datasets[dataset_id]
    filepath = os.path.join(config.data_dir, metadata.get('filename', f"{dataset_id}.json"))
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    try:
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Return preview
            if isinstance(data, list):
                return {
                    "preview": data[:limit],
                    "total_items": len(data)
                }
            else:
                return {"preview": data}
        else:
            return {"message": f"Preview not supported for {filepath.rsplit('.', 1)[-1]} files"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/preprocess")
async def preprocess_dataset(
    dataset_id: str,
    normalize: bool = True,
    standardize: bool = False
):
    """Preprocess a dataset."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    metadata = _datasets[dataset_id]
    filepath = os.path.join(config.data_dir, metadata.get('filename', f"{dataset_id}.json"))
    
    try:
        # Load dataset
        dataset = dataset_loader.load(filepath)
        
        # Preprocess
        processed = dataset_preprocessor.preprocess(
            dataset,
            normalize=normalize,
            standardize=standardize
        )
        
        # Save processed version
        processed_filename = f"{dataset_id}_processed.json"
        processed_path = os.path.join(config.data_dir, processed_filename)
        
        with open(processed_path, 'w') as f:
            json.dump(processed.to_dict(), f)
        
        # Create new metadata entry for processed dataset
        processed_id = f"{dataset_id}_processed"
        _datasets[processed_id] = {
            **metadata,
            "id": processed_id,
            "name": f"{metadata['name']} (processed)",
            "filename": processed_filename,
            "created_at": datetime.now().isoformat(),
            "preprocessing": {
                "normalize": normalize,
                "standardize": standardize
            }
        }
        
        return {
            "status": "success",
            "original_id": dataset_id,
            "processed_id": processed_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    metadata = _datasets[dataset_id]
    filepath = os.path.join(config.data_dir, metadata.get('filename', f"{dataset_id}.json"))
    
    # Delete file
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # Remove metadata
    del _datasets[dataset_id]
    
    return {"status": "deleted", "dataset_id": dataset_id}


@router.get("/types/supported")
async def get_supported_types():
    """Get supported dataset types and formats."""
    return {
        "types": [
            {
                "id": "demonstrations",
                "name": "Expert Demonstrations",
                "description": "State-action pairs from expert policies",
                "format": {
                    "trajectories": [
                        {
                            "states": "list of states",
                            "actions": "list of actions"
                        }
                    ]
                }
            },
            {
                "id": "trajectories",
                "name": "Full Trajectories",
                "description": "Complete state-action-reward-next_state tuples",
                "format": {
                    "trajectories": [
                        {
                            "transitions": [
                                {
                                    "state": "state",
                                    "action": "action",
                                    "reward": "float",
                                    "next_state": "next_state",
                                    "done": "bool"
                                }
                            ]
                        }
                    ]
                }
            },
            {
                "id": "replay_buffer",
                "name": "Replay Buffer",
                "description": "Raw experience tuples for offline RL",
                "format": {
                    "states": "array",
                    "actions": "array",
                    "rewards": "array",
                    "next_states": "array",
                    "dones": "array"
                }
            }
        ],
        "formats": ["json", "npz", "pkl", "h5"]
    }


@router.post("/validate")
async def validate_dataset_format(file: UploadFile = File(...)):
    """Validate a dataset without uploading."""
    try:
        contents = await file.read()
        filename = file.filename or "dataset"
        
        if filename.endswith('.json'):
            data = json.loads(contents.decode('utf-8'))
            results = dataset_validator.validate_trajectories(data)
        else:
            results = {
                "status": "unknown",
                "message": f"Cannot validate {filename.rsplit('.', 1)[-1]} format automatically"
            }
        
        return {
            "filename": filename,
            "size": len(contents),
            "validation": results
        }
        
    except json.JSONDecodeError as e:
        return {
            "filename": file.filename,
            "validation": {
                "status": "error",
                "message": f"Invalid JSON: {str(e)}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
