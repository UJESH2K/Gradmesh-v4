"use client";

import { useEffect, useRef } from "react";

type Node = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  power: number;
};

type Pulse = {
  from: number;
  to: number;
  progress: number;
  speed: number;
  outbound: boolean;
};

/**
 * The hero graphic: a coordinator at the centre, contributor GPUs orbiting it,
 * and shards travelling out and gradients coming back.
 *
 * It is a literal picture of the product rather than decoration, so the sizes
 * mean something: a node's radius tracks its compute, and shard pulses leave
 * for strong nodes more often than weak ones.
 */
export default function MeshCanvas({ density = 16 }: { density?: number }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width = 0;
    let height = 0;
    let ratio = 1;
    let frame = 0;
    let running = true;

    const nodes: Node[] = [];
    const pulses: Pulse[] = [];

    function resize() {
      const rect = canvas!.getBoundingClientRect();
      ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = rect.width;
      height = rect.height;
      canvas!.width = Math.max(1, Math.floor(width * ratio));
      canvas!.height = Math.max(1, Math.floor(height * ratio));
      context!.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    function seed() {
      nodes.length = 0;
      // Index 0 is the coordinator, pinned to the centre.
      nodes.push({ x: width / 2, y: height / 2, vx: 0, vy: 0, radius: 7, power: 1 });

      for (let index = 0; index < density; index += 1) {
        const angle = (index / density) * Math.PI * 2 + Math.random() * 0.5;
        const spread = Math.min(width, height) * (0.22 + Math.random() * 0.26);
        const power = 0.25 + Math.random() ** 1.6 * 0.75;
        nodes.push({
          x: width / 2 + Math.cos(angle) * spread * (width / Math.min(width, height)),
          y: height / 2 + Math.sin(angle) * spread,
          vx: (Math.random() - 0.5) * 0.16,
          vy: (Math.random() - 0.5) * 0.16,
          radius: 1.8 + power * 3.4,
          power,
        });
      }
    }

    function emitPulse() {
      if (nodes.length < 2) return;
      // Weight selection by compute, which is exactly how the real scheduler
      // hands out shards.
      const total = nodes.slice(1).reduce((sum, node) => sum + node.power, 0);
      let target = Math.random() * total;
      let chosen = 1;
      for (let index = 1; index < nodes.length; index += 1) {
        target -= nodes[index].power;
        if (target <= 0) {
          chosen = index;
          break;
        }
      }
      const outbound = Math.random() > 0.42;
      pulses.push({
        from: outbound ? 0 : chosen,
        to: outbound ? chosen : 0,
        progress: 0,
        speed: 0.006 + Math.random() * 0.008,
        outbound,
      });
    }

    function draw() {
      if (!running) return;
      frame += 1;
      context!.clearRect(0, 0, width, height);

      const centre = nodes[0];

      // Spokes from the coordinator to every contributor.
      for (let index = 1; index < nodes.length; index += 1) {
        const node = nodes[index];
        const distance = Math.hypot(node.x - centre.x, node.y - centre.y);
        const fade = Math.max(0, 1 - distance / (Math.max(width, height) * 0.62));
        context!.strokeStyle = `rgba(91, 124, 250, ${0.05 + fade * 0.09})`;
        context!.lineWidth = 0.6 + node.power * 0.5;
        context!.beginPath();
        context!.moveTo(centre.x, centre.y);
        context!.lineTo(node.x, node.y);
        context!.stroke();
      }

      // Peer links, drawn only when two contributors are close, so the graph
      // breathes instead of turning into a solid mat.
      for (let a = 1; a < nodes.length; a += 1) {
        for (let b = a + 1; b < nodes.length; b += 1) {
          const distance = Math.hypot(nodes[a].x - nodes[b].x, nodes[a].y - nodes[b].y);
          const threshold = Math.min(width, height) * 0.24;
          if (distance > threshold) continue;
          context!.strokeStyle = `rgba(255, 255, 255, ${0.05 * (1 - distance / threshold)})`;
          context!.lineWidth = 0.5;
          context!.beginPath();
          context!.moveTo(nodes[a].x, nodes[a].y);
          context!.lineTo(nodes[b].x, nodes[b].y);
          context!.stroke();
        }
      }

      // Shards travelling out, gradients coming back.
      for (let index = pulses.length - 1; index >= 0; index -= 1) {
        const pulse = pulses[index];
        pulse.progress += pulse.speed;
        if (pulse.progress >= 1) {
          pulses.splice(index, 1);
          continue;
        }
        const from = nodes[pulse.from];
        const to = nodes[pulse.to];
        if (!from || !to) {
          pulses.splice(index, 1);
          continue;
        }
        const eased = pulse.progress * pulse.progress * (3 - 2 * pulse.progress);
        const x = from.x + (to.x - from.x) * eased;
        const y = from.y + (to.y - from.y) * eased;
        const alpha = Math.sin(pulse.progress * Math.PI);
        context!.fillStyle = pulse.outbound
          ? `rgba(91, 124, 250, ${alpha * 0.9})`
          : `rgba(34, 211, 238, ${alpha * 0.85})`;
        context!.beginPath();
        context!.arc(x, y, pulse.outbound ? 2.4 : 2.0, 0, Math.PI * 2);
        context!.fill();
      }

      // Contributor nodes.
      for (let index = 1; index < nodes.length; index += 1) {
        const node = nodes[index];
        if (!reduceMotion) {
          node.x += node.vx;
          node.y += node.vy;
          // Gentle spring back toward the coordinator keeps the cluster framed.
          node.vx += (centre.x - node.x) * 0.000045;
          node.vy += (centre.y - node.y) * 0.000045;
          node.vx *= 0.999;
          node.vy *= 0.999;
        }

        const glow = context!.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.radius * 5);
        glow.addColorStop(0, `rgba(91, 124, 250, ${0.16 * node.power})`);
        glow.addColorStop(1, "rgba(91, 124, 250, 0)");
        context!.fillStyle = glow;
        context!.beginPath();
        context!.arc(node.x, node.y, node.radius * 5, 0, Math.PI * 2);
        context!.fill();

        context!.fillStyle = `rgba(233, 237, 241, ${0.35 + node.power * 0.5})`;
        context!.beginPath();
        context!.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        context!.fill();
      }

      // The coordinator, with a slow breathing ring.
      const breath = reduceMotion ? 0 : Math.sin(frame / 46) * 3;
      const halo = context!.createRadialGradient(
        centre.x,
        centre.y,
        0,
        centre.x,
        centre.y,
        46 + breath
      );
      halo.addColorStop(0, "rgba(91, 124, 250, 0.32)");
      halo.addColorStop(1, "rgba(91, 124, 250, 0)");
      context!.fillStyle = halo;
      context!.beginPath();
      context!.arc(centre.x, centre.y, 46 + breath, 0, Math.PI * 2);
      context!.fill();

      context!.strokeStyle = "rgba(91, 124, 250, 0.5)";
      context!.lineWidth = 1;
      context!.beginPath();
      context!.arc(centre.x, centre.y, 15 + breath * 0.4, 0, Math.PI * 2);
      context!.stroke();

      context!.fillStyle = "#5b7cfa";
      context!.beginPath();
      context!.arc(centre.x, centre.y, 5.5, 0, Math.PI * 2);
      context!.fill();

      if (!reduceMotion && frame % 22 === 0) emitPulse();
      requestAnimationFrame(draw);
    }

    resize();
    seed();
    draw();

    const observer = new ResizeObserver(() => {
      resize();
      seed();
    });
    observer.observe(canvas);

    return () => {
      running = false;
      observer.disconnect();
    };
  }, [density]);

  return <canvas ref={canvasRef} className="mesh-canvas" aria-hidden="true" />;
}
