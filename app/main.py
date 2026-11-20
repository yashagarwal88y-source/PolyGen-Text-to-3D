import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.prompt_enhancer import PromptEnhancer
from app.generator import TextTo3DGenerator
from app.mesh_processor import MeshProcessor

app = FastAPI(
    title="PolyGen: Generative Text-to-3D Asset Pipeline",
    description="End-to-end Text-to-3D synthesis with LLM prompt enhancement and trimesh validation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path("static")
METADATA_LOG = OUTPUT_DIR / "history.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

enhancer = PromptEnhancer()
generator = TextTo3DGenerator()
processor = MeshProcessor()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=2, max_length=500, description="Raw user description of the 3D asset")
    enhance_prompt: bool = Field(default=True, description="Whether to expand prompt via LLM layer")
    normalize_mesh: bool = Field(default=True, description="Center and scale mesh to unit bounding box")
    decimate_mesh: bool = Field(default=True, description="Reduce face count if above target threshold")


@app.get("/")
def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return {"message": "PolyGen API is running. UI will be served from static/index.html."}
    return FileResponse(index_file)


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "backend": generator.backend,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/generate")
def generate_asset(req: GenerateRequest):
    try:
        # 1. Prompt Enhancement Layer
        if req.enhance_prompt:
            prompt_data = enhancer.enhance(req.prompt)
        else:
            prompt_data = {
                "raw_prompt": req.prompt.strip(),
                "enhanced_prompt": req.prompt.strip(),
                "provider": "bypassed",
            }

        # 2. Text-to-3D Synthesis Layer
        gen_data = generator.generate(
            enhanced_prompt=prompt_data["enhanced_prompt"],
            raw_prompt=prompt_data["raw_prompt"],
        )

        # 3. Mesh Post-Processing & Validation Layer
        mesh_data = processor.process(
            input_path=gen_data["file_path"],
            normalize=req.normalize_mesh,
            decimate=req.decimate_mesh,
        )

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset_id": gen_data["asset_id"],
            "model_url": f"/outputs/{gen_data['asset_id']}.glb",
            "raw_prompt": prompt_data["raw_prompt"],
            "enhanced_prompt": prompt_data["enhanced_prompt"],
            "prompt_provider": prompt_data["provider"],
            "generation_backend": gen_data["backend"],
            "generation_time_sec": gen_data["generation_time_sec"],
            "mesh_metrics": mesh_data,
        }

        with METADATA_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}")


@app.get("/api/history")
def list_history(limit: int = 20):
    if not METADATA_LOG.exists():
        return {"items": []}
    lines = METADATA_LOG.read_text(encoding="utf-8").strip().splitlines()
    items = [json.loads(line) for line in lines[-limit:]]
    items.reverse()
    return {"items": items}