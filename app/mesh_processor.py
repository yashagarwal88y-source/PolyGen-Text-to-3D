from pathlib import Path
import numpy as np
import trimesh


class MeshProcessor:
    """
    Validates, normalizes, and optimizes generated 3D meshes (.glb/.obj)
    for consistent Three.js rendering and quantitative evaluation.
    """

    def __init__(self, target_faces: int = 15000):
        self.target_faces = target_faces

    def process(self, input_path: str | Path, normalize: bool = True, decimate: bool = True) -> dict:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Mesh file not found: {path}")

        loaded = trimesh.load(path, force="mesh")
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded

        initial_faces = int(len(mesh.faces))
        initial_vertices = int(len(mesh.vertices))

        # 1. Repair normals & remove degenerate/unreferenced geometry (trimesh 4.x API)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()

        # 2. Optional polygon decimation for web performance
        decimated = False
        if decimate and len(mesh.faces) > self.target_faces:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=self.target_faces)
                decimated = True
            except Exception:
                pass

        # 3. Center at origin (0, 0, 0) and scale largest axis to 1.0 unit
        if normalize and len(mesh.vertices) > 0:
            mesh.apply_translation(-mesh.centroid)
            max_extent = float(np.max(mesh.extents))
            if max_extent > 0:
                mesh.apply_scale(1.0 / max_extent)

        # Overwrite with normalized, web-ready .glb
        mesh.export(path)

        bounds = mesh.extents.tolist() if len(mesh.vertices) > 0 else [0.0, 0.0, 0.0]

        return {
            "file_path": str(path),
            "initial_vertices": initial_vertices,
            "initial_faces": initial_faces,
            "final_vertices": int(len(mesh.vertices)),
            "final_faces": int(len(mesh.faces)),
            "is_watertight": bool(mesh.is_watertight),
            "bounding_box": [round(float(x), 4) for x in bounds],
            "decimated": decimated,
        }


if __name__ == "__main__":
    glb_files = sorted(Path("outputs").glob("*.glb"))
    if not glb_files:
        print("No .glb files found in outputs/. Run `python -m app.generator` first.")
    else:
        processor = MeshProcessor()
        metrics = processor.process(glb_files[-1])
        print("Processed mesh metrics:", metrics)