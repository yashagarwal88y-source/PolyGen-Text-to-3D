import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from app.prompt_enhancer import PromptEnhancer
from app.generator import TextTo3DGenerator
from app.mesh_processor import MeshProcessor

BENCHMARK_PROMPTS = [
    "A medieval iron shield with a raised central boss",
    "A futuristic reconnaissance drone with brushed titanium armor",
    "An ancient stone watchtower with a square base",
    "A sleek sci-fi energy core ring",
]


class PipelineEvaluator:
    """
    Runs automated quantitative evaluation across a benchmark prompt suite,
    tracking latency, polygon topology, watertightness, and prompt expansion ratio.
    """

    def __init__(self, target_faces: int = 15000):
        self.enhancer = PromptEnhancer()
        self.generator = TextTo3DGenerator()
        self.processor = MeshProcessor(target_faces=target_faces)

    def _compute_prompt_alignment_score(self, raw_prompt: str, enhanced_prompt: str) -> float:
        """
        Computes a normalized semantic keyword retention & 3D descriptor richness score [0.0, 1.0]
        as a lightweight deterministic proxy when heavy vision-language checkpoints are offline.
        """
        raw_tokens = {w.lower().strip(".,!?") for w in raw_prompt.split() if len(w) > 2}
        enhanced_lower = enhanced_prompt.lower()
        if not raw_tokens:
            return 1.0

        retained = sum(1 for tok in raw_tokens if tok in enhanced_lower)
        retention_ratio = retained / len(raw_tokens)

        # Bonus for explicit 3D geometry / PBR material descriptors
        pbr_keywords = ["geometry", "pbr", "surface", "lighting", "beveled", "metallic", "proportions"]
        descriptor_hits = sum(1 for kw in pbr_keywords if kw in enhanced_lower)
        richness_score = min(descriptor_hits / 4.0, 1.0)

        return round(0.6 * retention_ratio + 0.4 * richness_score, 4)

    def run_benchmark(self, prompts: list[str] | None = None) -> dict:
        test_prompts = prompts or BENCHMARK_PROMPTS
        results = []

        for raw_prompt in test_prompts:
            t0 = time.perf_counter()

            # 1. Prompt Enhancement
            prompt_res = self.enhancer.enhance(raw_prompt)

            # 2. 3D Asset Generation
            gen_res = self.generator.generate(
                enhanced_prompt=prompt_res["enhanced_prompt"],
                raw_prompt=raw_prompt,
            )

            # 3. Mesh Post-Processing & Validation
            mesh_res = self.processor.process(gen_res["file_path"])

            total_latency = round(time.perf_counter() - t0, 4)
            alignment_score = self._compute_prompt_alignment_score(
                raw_prompt, prompt_res["enhanced_prompt"]
            )

            results.append({
                "raw_prompt": raw_prompt,
                "asset_id": gen_res["asset_id"],
                "prompt_provider": prompt_res["provider"],
                "generation_backend": gen_res["backend"],
                "total_latency_sec": total_latency,
                "final_vertices": mesh_res["final_vertices"],
                "final_faces": mesh_res["final_faces"],
                "is_watertight": mesh_res["is_watertight"],
                "alignment_score": alignment_score,
            })

        latencies = [r["total_latency_sec"] for r in results]
        faces = [r["final_faces"] for r in results]
        scores = [r["alignment_score"] for r in results]
        watertight_rate = sum(1 for r in results if r["is_watertight"]) / len(results)

        report = {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "num_samples": len(results),
            "summary": {
                "mean_latency_sec": round(float(np.mean(latencies)), 4),
                "mean_face_count": round(float(np.mean(faces)), 1),
                "watertight_pass_rate": round(float(watertight_rate), 4),
                "mean_alignment_score": round(float(np.mean(scores)), 4),
            },
            "samples": results,
        }

        report_path = Path("outputs") / "benchmark_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return report


if __name__ == "__main__":
    evaluator = PipelineEvaluator()
    summary_report = evaluator.run_benchmark()
    print(json.dumps(summary_report["summary"], indent=2))