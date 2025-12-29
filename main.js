import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

// --- 1. SETUP SCENE ---
const scene = new THREE.Scene();
scene.add(new THREE.GridHelper(30, 30, 0x333333, 0x111111)); 

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 10, 35);

const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('canvas'), antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// --- 2. OBJECTS ---
const handMarker = new THREE.Mesh(
    new THREE.SphereGeometry(1.5, 32, 32),
    new THREE.MeshBasicMaterial({ color: 0xff3333 })
);
scene.add(handMarker);

const particleCount = 5000;
const pGeometry = new THREE.BufferGeometry();
const pPositions = new Float32Array(particleCount * 3);
const pVelocities = new Float32Array(particleCount * 3);

for (let i = 0; i < particleCount * 3; i++) {
    pPositions[i] = (Math.random() - 0.5) * 20;
    pVelocities[i] = 0;
}
pGeometry.setAttribute('position', new THREE.BufferAttribute(pPositions, 3));
const particles = new THREE.Points(pGeometry, new THREE.PointsMaterial({ color: 0x00ffff, size: 0.4 }));
scene.add(particles);

// --- 3. START EVERYTHING ---
const videoElement = document.getElementById('webcam');
let handLandmarker;

// STARTUP FUNCTION: Camera First, Then AI
async function startApp() {
    
    // STEP 1: Ask for Camera immediately
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
        videoElement.play();
        console.log("Camera started!");
    } catch (err) {
        alert("Camera failed to start: " + err.message);
        return; // Stop if no camera
    }

    // STEP 2: Load AI in the background
    console.log("Loading AI Model...");
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });
    console.log("AI Loaded!");
}

startApp(); // Run startup

// --- 4. ANIMATION LOOP ---
let lastVideoTime = -1;

function animate() {
    requestAnimationFrame(animate);
    controls.update();

    // Only calculate physics if AI is ready AND video is playing
    if (handLandmarker && videoElement.readyState >= 2) {
        
        // A. FIND HAND
        if (videoElement.currentTime !== lastVideoTime) {
            lastVideoTime = videoElement.currentTime;
            const results = handLandmarker.detectForVideo(videoElement, performance.now());
            
            if (results.landmarks.length > 0) {
                const tip = results.landmarks[0][8];
                const targetX = (0.5 - tip.x) * 30;
                const targetY = (0.5 - tip.y) * 20;
                
                // Move Red Ball
                handMarker.position.x += (targetX - handMarker.position.x) * 0.2;
                handMarker.position.y += (targetY - handMarker.position.y) * 0.2;
                handMarker.position.z = 0;
            }
        }

        // B. MOVE PARTICLES
        const pos = particles.geometry.attributes.position.array;
        for (let i = 0; i < particleCount; i++) {
            const ix = i * 3; const iy = i * 3 + 1; const iz = i * 3 + 2;

            const dx = pos[ix] - handMarker.position.x;
            const dy = pos[iy] - handMarker.position.y;
            const dz = pos[iz] - handMarker.position.z;
            const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

            if (dist < 6) {
                const force = (6 - dist) / 6;
                pVelocities[ix] += (dx / dist) * force * 0.5;
                pVelocities[iy] += (dy / dist) * force * 0.5;
                pVelocities[iz] += (dz / dist) * force * 0.5;
            }

            pVelocities[ix] = (pVelocities[ix] * 0.95) - (pos[ix] * 0.005);
            pVelocities[iy] = (pVelocities[iy] * 0.95) - (pos[iy] * 0.005);
            pVelocities[iz] = (pVelocities[iz] * 0.95) - (pos[iz] * 0.005);

            pos[ix] += pVelocities[ix];
            pos[iy] += pVelocities[iy];
            pos[iz] += pVelocities[iz];
        }
        particles.geometry.attributes.position.needsUpdate = true;
    }

    renderer.render(scene, camera);
}

animate();

// Handle Resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});