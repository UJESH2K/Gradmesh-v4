"use client";

import { useEffect, useRef } from "react";

/**
 * The moving field behind the landing page.
 *
 * Three slow-drifting radial blobs composited on a dark ground, with a static
 * grain layer over the top. The grain is the part that makes it read as
 * infrastructure rather than as a generic gradient: it hides the banding that
 * large soft gradients produce on 8-bit displays, which is what usually makes
 * this kind of background look cheap.
 *
 * Runs at half resolution because the content is entirely low-frequency, so the
 * upscale is invisible and the fill cost drops by four. Pauses when the tab is
 * hidden and renders one static frame when the visitor prefers reduced motion.
 */

type Blob = {
  x: number;
  y: number;
  radius: number;
  colour: [number, number, number];
  driftX: number;
  driftY: number;
  phase: number;
};

const PALETTE: [number, number, number][] = [
  [91, 124, 250], // azure, the product accent
  [34, 211, 238], // cyan
  [124, 92, 255], // violet, for depth at the edges
];

export default function LiveBackground({
  intensity = 1,
  className,
}: {
  intensity?: number;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const SCALE = 0.5;

    let width = 0;
    let height = 0;
    let frame = 0;
    let raf = 0;
    let running = true;
    let grain: HTMLCanvasElement | null = null;
    const blobs: Blob[] = [];

    function buildGrain() {
      // One noise tile, generated once and tiled. Regenerating per frame would
      // cost more than everything else on this canvas combined.
      const tile = document.createElement("canvas");
      tile.width = 160;
      tile.height = 160;
      const tileContext = tile.getContext("2d");
      if (!tileContext) return null;
      const image = tileContext.createImageData(tile.width, tile.height);
      for (let index = 0; index < image.data.length; index += 4) {
        const value = 128 + (Math.random() - 0.5) * 46;
        image.data[index] = value;
        image.data[index + 1] = value;
        image.data[index + 2] = value;
        image.data[index + 3] = 15;
      }
      tileContext.putImageData(image, 0, 0);
      return tile;
    }

    function resize() {
      const rect = canvas!.getBoundingClientRect();
      width = Math.max(1, Math.floor(rect.width * SCALE));
      height = Math.max(1, Math.floor(rect.height * SCALE));
      canvas!.width = width;
      canvas!.height = height;
      seed();
    }

    function seed() {
      blobs.length = 0;
      const span = Math.max(width, height);
      for (let index = 0; index < PALETTE.length; index += 1) {
        blobs.push({
          x: width * (0.25 + 0.25 * index),
          y: height * (index % 2 === 0 ? 0.35 : 0.68),
          radius: span * (0.55 + index * 0.12),
          colour: PALETTE[index],
          driftX: span * (0.16 + index * 0.05),
          driftY: span * (0.1 + index * 0.04),
          phase: index * 2.1,
        });
      }
    }

    function draw() {
      if (!running) return;
      frame += 1;

      context!.fillStyle = "#07080b";
      context!.fillRect(0, 0, width, height);
      context!.globalCompositeOperation = "lighter";

      const time = reduceMotion ? 0 : frame / 900;
      for (const blob of blobs) {
        const x = blob.x + Math.cos(time + blob.phase) * blob.driftX;
        const y = blob.y + Math.sin(time * 0.8 + blob.phase) * blob.driftY;
        const [r, g, b] = blob.colour;

        const gradient = context!.createRadialGradient(x, y, 0, x, y, blob.radius);
        gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${0.3 * intensity})`);
        gradient.addColorStop(0.45, `rgba(${r}, ${g}, ${b}, ${0.1 * intensity})`);
        gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
        context!.fillStyle = gradient;
        context!.beginPath();
        context!.arc(x, y, blob.radius, 0, Math.PI * 2);
        context!.fill();
      }

      context!.globalCompositeOperation = "source-over";

      if (grain) {
        const pattern = context!.createPattern(grain, "repeat");
        if (pattern) {
          context!.fillStyle = pattern;
          context!.fillRect(0, 0, width, height);
        }
      }

      if (reduceMotion) return;
      raf = requestAnimationFrame(draw);
    }

    grain = buildGrain();
    resize();
    draw();

    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    const onVisibility = () => {
      if (document.hidden) {
        running = false;
        cancelAnimationFrame(raf);
      } else if (!reduceMotion) {
        running = true;
        draw();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      observer.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [intensity]);

  return <canvas ref={canvasRef} className={className || "live-bg"} aria-hidden="true" />;
}
