import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const container = document.getElementById("canvas-container");
const statusText = document.getElementById("status-text");
const backendBadge = document.getElementById("backend-badge");
const generateBtn = document.getElementById("generate-btn");
const promptInput = document.getElementById("prompt-input");
const enhancedDisplay = document.getElementById("enhanced-prompt-display");
const downloadBtn = document.getElementById("download-btn");

const optEnhance = document.getElementById("opt-enhance");
const optNormalize = document.getElementById("opt-normalize");
const optDecimate = document.getElementById("opt-decimate");

const metricVerts = document.getElementById("metric-verts");
const metricFaces = document.getElementById("metric-faces");
const metricWatertight = document.getElementById("metric-watertight");
const metricLatency = document.getElementById("metric-latency");

const wireframeBtn = document.getElementById("toggle-wireframe");
const spinBtn = document.getElementById("toggle-spin");
const resetBtn = document.getElementById("reset-camera");

// --- 1. Three.js Scene, Camera & Renderer Setup ---
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  45,
  container.clientWidth / container.clientHeight,
  0.05,
  100
);
const defaultCamPos = new THREE.Vector3(1.8, 1.3, 2.2);
camera.position.copy(defaultCamPos);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

// Studio Lighting & Reference Grid
const ambientLight = new THREE.AmbientLight(0xffffff, 1.1);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
keyLight.position.set(3, 5, 4);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xdbeafe, 0.9);
fillLight.position.set(-4, 2, -3);
scene.add(fillLight);

const gridHelper = new THREE.GridHelper(3, 12, 0x94a3b8, 0xcbd5e1);
gridHelper.position.y = -0.55;
scene.add(gridHelper);

// OrbitControls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.8;

let currentModel = null;
let wireframeEnabled = false;
const loader = new GLTFLoader();

function setWireframe(enabled) {
  wireframeEnabled = enabled;
  wireframeBtn.textContent = `Wireframe: ${enabled ? "ON" : "OFF"}`;
  wireframeBtn.classList.toggle("active", enabled);

  if (!currentModel) return;
  currentModel.traverse((child) => {
    if (child.isMesh && child.material) {
      child.material.wireframe = enabled;
    }
  });
}

function loadGlbModel(url) {
  statusText.textContent = "Loading 3D mesh into Three.js viewport...";
  loader.load(
    url,
    (gltf) => {
      if (currentModel) {
        scene.remove(currentModel);
      }
      currentModel = gltf.scene;
      scene.add(currentModel);
      setWireframe(wireframeEnabled);

      downloadBtn.href = url;
      downloadBtn.style.display = "inline-block";
      statusText.textContent = `Loaded ${url.split("/").pop()} — Drag to orbit, scroll to zoom.`;
    },
    undefined,
    (err) => {
      console.error("GLTF load error:", err);
      statusText.textContent = "Failed to load .GLB asset into viewport.";
    }
  );
}

// --- 2. UI Event Handlers ---
wireframeBtn.addEventListener("click", () => setWireframe(!wireframeEnabled));

spinBtn.addEventListener("click", () => {
  controls.autoRotate = !controls.autoRotate;
  spinBtn.textContent = `Auto-Rotate: ${controls.autoRotate ? "ON" : "OFF"}`;
  spinBtn.classList.toggle("active", controls.autoRotate);
});

resetBtn.addEventListener("click", () => {
  camera.position.copy(defaultCamPos);
  controls.target.set(0, 0, 0);
  controls.update();
});

window.addEventListener("resize", () => {
  if (!container.clientWidth || !container.clientHeight) return;
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

// --- 3. API Integration ---
async function checkBackendHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    backendBadge.textContent = `Backend: ${data.backend}`;
  } catch {
    backendBadge.textContent = "Backend: Offline";
  }
}

generateBtn.addEventListener("click", async () => {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    statusText.textContent = "Please enter a 3D asset description first.";
    promptInput.focus();
    return;
  }

  generateBtn.disabled = true;
  generateBtn.textContent = "Synthesizing 3D Mesh...";
  statusText.textContent = "Enhancing prompt and generating 3D geometry...";

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        enhance_prompt: optEnhance.checked,
        normalize_mesh: optNormalize.checked,
        decimate_mesh: optDecimate.checked,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Generation error");
    }

    enhancedDisplay.textContent = `[${data.prompt_provider}] ${data.enhanced_prompt}`;
    metricVerts.textContent = data.mesh_metrics.final_vertices.toLocaleString();
    metricFaces.textContent = data.mesh_metrics.final_faces.toLocaleString();
    metricWatertight.textContent = data.mesh_metrics.is_watertight ? "Yes" : "No";
    metricLatency.textContent = `${data.generation_time_sec}s`;
    backendBadge.textContent = `Backend: ${data.generation_backend}`;

    loadGlbModel(data.model_url);
  } catch (err) {
    statusText.textContent = `Error: ${err.message}`;
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Generate 3D Asset (.GLB)";
  }
});

// Render Loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

spinBtn.classList.add("active");
checkBackendHealth();
animate();