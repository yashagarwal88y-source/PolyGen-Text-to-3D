import os
import time
import uuid
import requests
import trimesh
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TextTo3DGenerator:
    """
    Asynchronous/synchronous wrapper for Text-to-3D generation backends
    with a procedural trimesh fallback for offline testing and demos.
    """

    def __init__(self, backend: str | None = None):
        self.backend = (backend or os.getenv("GENERATION_BACKEND", "mock")).lower()
        self.hf_token = os.getenv("HF_TOKEN")
        self.tripo_key = os.getenv("TRIPO_API_KEY")

    def generate(self, enhanced_prompt: str, raw_prompt: str = "") -> dict:
        start_time = time.perf_counter()
        asset_id = f"polygen_{uuid.uuid4().hex[:8]}"
        output_path = OUTPUT_DIR / f"{asset_id}.glb"

        used_backend = self.backend

        try:
            if self.backend == "tripo" and self.tripo_key and "your_" not in self.tripo_key:
                self._generate_via_tripo(enhanced_prompt, output_path)
            elif self.backend == "huggingface" and self.hf_token and "your_" not in self.hf_token:
                self._generate_via_hf(enhanced_prompt, output_path)
            else:
                used_backend = "procedural_fallback"
                self._generate_procedural_asset(raw_prompt or enhanced_prompt, output_path)
        except Exception as exc:
            # Graceful fallback so demo/evaluation pipelines never crash on rate limits
            used_backend = f"procedural_fallback (recovered from: {type(exc).__name__})"
            self._generate_procedural_asset(raw_prompt or enhanced_prompt, output_path)

        latency_sec = round(time.perf_counter() - start_time, 3)

        return {
            "asset_id": asset_id,
            "file_path": str(output_path),
            "backend": used_backend,
            "generation_time_sec": latency_sec,
        }

    def _generate_via_tripo(self, prompt: str, output_path: Path) -> None:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.tripo_key}",
        }
        create_resp = requests.post(
            "https://api.tripo3d.ai/v2/openapi/task",
            headers=headers,
            json={"type": "text_to_model", "prompt": prompt[:1000]},
            timeout=30,
        )
        create_resp.raise_for_status()
        task_id = create_resp.json()["data"]["task_id"]

        for _ in range(45):
            time.sleep(2)
            status_resp = requests.get(
                f"https://api.tripo3d.ai/v2/openapi/task/{task_id}",
                headers=headers,
                timeout=20,
            )
            status_resp.raise_for_status()
            data = status_resp.json()["data"]
            if data["status"] == "success":
                model_url = data["output"]["model"]
                glb_bytes = requests.get(model_url, timeout=60).content
                output_path.write_bytes(glb_bytes)
                return
            if data["status"] in {"failed", "cancelled"}:
                raise RuntimeError(f"Tripo3D task ended with status: {data['status']}")

        raise TimeoutError("Tripo3D generation timed out.")

    def _generate_via_hf(self, prompt: str, output_path: Path) -> None:
        # Uses Hugging Face Inference API for text-to-3D (e.g., Shap-E)
        api_url = "https://api-inference.huggingface.co/models/openai/shap-e"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        resp = requests.post(api_url, headers=headers, json={"inputs": prompt}, timeout=90)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)

    def _generate_procedural_asset(self, prompt: str, output_path: Path) -> None:
        """Builds a clean, watertight 3D mesh based on prompt keywords."""
        text = prompt.lower()

        if "shield" in text:
            mesh = trimesh.creation.cylinder(radius=1.0, height=0.12, sections=32)
            boss = trimesh.creation.icosphere(subdivisions=2, radius=0.28)
            boss.apply_translation([0, 0, 0.08])
            mesh = trimesh.util.concatenate([mesh, boss])
        elif "drone" in text or "spaceship" in text:
            body = trimesh.creation.icosphere(subdivisions=2, radius=0.45)
            body.apply_scale([1.4, 0.8, 0.4])
            ring = trimesh.creation.annulus(r_min=0.75, r_max=0.9, height=0.06, sections=32)
            mesh = trimesh.util.concatenate([body, ring])
        elif "tower" in text or "pillar" in text or "sword" in text:
            base = trimesh.creation.box(extents=[0.8, 0.8, 0.2])
            shaft = trimesh.creation.cylinder(radius=0.25, height=1.8, sections=24)
            shaft.apply_translation([0, 0, 0.9])
            mesh = trimesh.util.concatenate([base, shaft])
        else:
            # Default sculpted torus-knot style geometric artifact
            mesh = trimesh.creation.torus(major_radius=0.7, minor_radius=0.22, major_sections=36, minor_sections=18)

        # Apply subtle metallic-blue PBR vertex color so it renders cleanly in Three.js
        mesh.visual.vertex_colors = np.array([70, 140, 230, 255], dtype=np.uint8)
        mesh.export(output_path)


if __name__ == "__main__":
    gen = TextTo3DGenerator()
    result = gen.generate("A futuristic reconnaissance drone with brushed titanium armor")
    print("Generated asset:", result)