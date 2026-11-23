import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.prompt_enhancer import PromptEnhancer
from app.generator import TextTo3DGenerator
from app.mesh_processor import MeshProcessor
from app.evaluator import PipelineEvaluator
from app.main import app


def test_prompt_enhancer_expands_input():
    enhancer = PromptEnhancer()
    result = enhancer.enhance("a medieval iron shield")
    assert result["raw_prompt"] == "a medieval iron shield"
    assert len(result["enhanced_prompt"]) > len(result["raw_prompt"])


def test_prompt_enhancer_rejects_empty_string():
    enhancer = PromptEnhancer()
    with pytest.raises(ValueError):
        enhancer.enhance("   ")


def test_generator_and_mesh_processor_normalization(tmp_path: Path):
    gen = TextTo3DGenerator(backend="mock")
    gen_res = gen.generate("A futuristic reconnaissance drone", raw_prompt="drone")
    glb_path = Path(gen_res["file_path"])
    assert glb_path.exists()

    processor = MeshProcessor(target_faces=5000)
    metrics = processor.process(glb_path, normalize=True, decimate=True)

    assert metrics["final_vertices"] > 0
    assert metrics["final_faces"] > 0
    assert max(metrics["bounding_box"]) == pytest.approx(1.0, rel=1e-2)

    # Clean up temporary test artifact
    glb_path.unlink(missing_ok=True)


def test_evaluator_alignment_score():
    evaluator = PipelineEvaluator()
    score = evaluator._compute_prompt_alignment_score(
        "medieval iron shield",
        "A standalone 3D asset of medieval iron shield with crisp beveled geometry and PBR metallic surface lighting.",
    )
    assert 0.8 <= score <= 1.0


def test_fastapi_health_and_generate_endpoint():
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    resp = client.post(
        "/api/generate",
        json={
            "prompt": "A stone watchtower",
            "enhance_prompt": True,
            "normalize_mesh": True,
            "decimate_mesh": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "asset_id" in body
    assert body["mesh_metrics"]["is_watertight"] is True

    # Clean up generated test .glb
    Path("outputs", f"{body['asset_id']}.glb").unlink(missing_ok=True)