"use client";

import { useEffect, useRef, useState } from "react";

/**
 * The 3D element on the training screen.
 *
 * **Where your GLB goes:** drop the file at `public/models/training-rig.glb`.
 * Nothing else needs to change. On mount this component checks whether that
 * file exists and loads it if so, otherwise it renders the procedural GPU fan
 * below. That keeps the screen working today and makes swapping in a real model
 * a file copy rather than a code change.
 *
 * **What the model should contain.** One or more baked animation clips, each
 * looping cleanly from first to last frame, authored around the origin and
 * roughly one unit tall. Every clip found is played together, so a robot arm
 * with separate clips per joint works without configuration. Playback rate is
 * driven by `activity`, so the rig visibly runs faster when the mesh is busy
 * and idles when it is not. If the model has no clips it is simply rotated.
 *
 * The whole thing is deliberately one file with no scene graph framework. It is
 * meant to be replaced and improved, not built on.
 */

const MODEL_URL = "/models/training-rig.glb";

export type RigState = "idle" | "training" | "aggregating" | "done" | "failed";

const STATE_COLOUR: Record<RigState, number> = {
  idle: 0x626c80,
  training: 0x5b7cfa,
  aggregating: 0x22d3ee,
  done: 0x34d399,
  failed: 0xf76b6b,
};

export default function TrainingRig({
  state = "idle",
  activity = 0,
  height = 260,
}: {
  /** Drives colour and whether the rig runs at all. */
  state?: RigState;
  /** 0 to 1. Scales playback speed, so a busy mesh visibly spins faster. */
  activity?: number;
  height?: number;
}) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [source, setSource] = useState<"model" | "fallback" | "loading">("loading");

  // Live values the render loop reads, so changing props never rebuilds the scene.
  const stateRef = useRef(state);
  const activityRef = useRef(activity);
  stateRef.current = state;
  activityRef.current = activity;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let disposed = false;
    let cleanup = () => {};

    (async () => {
      const THREE = await import("three");
      if (disposed) return;

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
      camera.position.set(0, 0.9, 3.4);
      camera.lookAt(0, 0.1, 0);

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(mount.clientWidth, mount.clientHeight, false);
      mount.appendChild(renderer.domElement);

      scene.add(new THREE.AmbientLight(0xffffff, 0.55));
      const key = new THREE.DirectionalLight(0xffffff, 1.6);
      key.position.set(3, 4, 2);
      scene.add(key);
      const rim = new THREE.PointLight(STATE_COLOUR.training, 18, 12);
      rim.position.set(-2, 1.4, -1.6);
      scene.add(rim);

      const root = new THREE.Group();
      scene.add(root);

      let mixer: import("three").AnimationMixer | null = null;
      let spinner: import("three").Object3D | null = null;
      const tinted: import("three").MeshStandardMaterial[] = [];

      /** The stand-in: a GPU fan that actually spins. */
      function buildFallback() {
        const body = new THREE.Mesh(
          new THREE.CylinderGeometry(1.05, 1.05, 0.34, 48, 1, true),
          new THREE.MeshStandardMaterial({
            color: 0x1a1f2b,
            metalness: 0.6,
            roughness: 0.45,
            side: THREE.DoubleSide,
          })
        );
        body.rotation.x = Math.PI / 2;
        root.add(body);

        const ring = new THREE.Mesh(
          new THREE.TorusGeometry(1.05, 0.045, 16, 64),
          new THREE.MeshStandardMaterial({
            color: STATE_COLOUR.training,
            emissive: STATE_COLOUR.training,
            emissiveIntensity: 0.75,
            metalness: 0.3,
            roughness: 0.3,
          })
        );
        root.add(ring);
        tinted.push(ring.material as import("three").MeshStandardMaterial);

        const fan = new THREE.Group();
        const bladeMaterial = new THREE.MeshStandardMaterial({
          color: 0x8e9bb5,
          metalness: 0.75,
          roughness: 0.35,
        });
        for (let index = 0; index < 9; index += 1) {
          const blade = new THREE.Mesh(new THREE.BoxGeometry(0.78, 0.02, 0.3), bladeMaterial);
          blade.position.x = 0.42;
          blade.rotation.z = 0.42;
          const arm = new THREE.Group();
          arm.add(blade);
          arm.rotation.z = (index / 9) * Math.PI * 2;
          fan.add(arm);
        }
        const hub = new THREE.Mesh(
          new THREE.CylinderGeometry(0.24, 0.24, 0.16, 24),
          new THREE.MeshStandardMaterial({ color: 0x2b3345, metalness: 0.8, roughness: 0.3 })
        );
        hub.rotation.x = Math.PI / 2;
        fan.add(hub);
        root.add(fan);
        spinner = fan;

        setSource("fallback");
      }

      // Probe before loading: a 404 through GLTFLoader logs a console error that
      // looks like a bug, and a missing model is the expected state until one
      // is supplied.
      let hasModel = false;
      try {
        const probe = await fetch(MODEL_URL, { method: "HEAD" });
        hasModel = probe.ok && (probe.headers.get("content-type") ?? "").indexOf("html") === -1;
      } catch {
        hasModel = false;
      }
      if (disposed) return;

      if (hasModel) {
        try {
          const { GLTFLoader } = await import("three/examples/jsm/loaders/GLTFLoader.js");
          const gltf = await new GLTFLoader().loadAsync(MODEL_URL);
          if (disposed) return;

          const model = gltf.scene;
          // Normalise whatever arrives to a predictable size and centre, so the
          // camera never has to be retuned for a new model.
          const box = new THREE.Box3().setFromObject(model);
          const size = box.getSize(new THREE.Vector3());
          const centre = box.getCenter(new THREE.Vector3());
          const scale = 1.9 / Math.max(size.x, size.y, size.z || 1);
          model.scale.setScalar(scale);
          model.position.sub(centre.multiplyScalar(scale));
          root.add(model);

          if (gltf.animations.length > 0) {
            mixer = new THREE.AnimationMixer(model);
            for (const clip of gltf.animations) mixer.clipAction(clip).play();
          } else {
            spinner = model;
          }
          setSource("model");
        } catch {
          buildFallback();
        }
      } else {
        buildFallback();
      }

      const clock = new THREE.Clock();
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      let raf = 0;

      function frame() {
        if (disposed) return;
        const delta = clock.getDelta();
        const current = stateRef.current;
        const running = current === "training" || current === "aggregating";

        // Idle still turns, slowly, so the rig never looks broken.
        const speed = reduceMotion ? 0 : running ? 0.35 + activityRef.current * 2.4 : 0.12;

        if (mixer) mixer.update(delta * speed);
        if (spinner) spinner.rotation.z -= delta * speed * 1.6;

        const colour = STATE_COLOUR[current] ?? STATE_COLOUR.idle;
        rim.color.setHex(colour);
        for (const material of tinted) {
          material.color.setHex(colour);
          material.emissive.setHex(colour);
        }

        root.rotation.y = Math.sin(clock.elapsedTime * 0.25) * 0.16;

        renderer.render(scene, camera);
        raf = requestAnimationFrame(frame);
      }
      frame();

      function resize() {
        const width = mount!.clientWidth;
        const height = mount!.clientHeight;
        if (!width) return;
        renderer.setSize(width, height, false);
        camera.aspect = width / Math.max(1, height);
        camera.updateProjectionMatrix();
      }
      const observer = new ResizeObserver(resize);
      observer.observe(mount);
      resize();

      cleanup = () => {
        cancelAnimationFrame(raf);
        observer.disconnect();
        renderer.dispose();
        scene.traverse((object) => {
          const mesh = object as import("three").Mesh;
          mesh.geometry?.dispose?.();
          const material = mesh.material as import("three").Material | import("three").Material[];
          if (Array.isArray(material)) material.forEach((item) => item.dispose());
          else material?.dispose?.();
        });
        renderer.domElement.remove();
      };
    })();

    return () => {
      disposed = true;
      cleanup();
    };
  }, []);

  return (
    <div className="rig" style={{ height }}>
      <div ref={mountRef} className="rig-canvas" />
      {source === "fallback" ? (
        <span className="rig-note" title={`Drop a .glb at public${MODEL_URL} to replace this`}>
          placeholder rig
        </span>
      ) : null}
    </div>
  );
}
