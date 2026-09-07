"use client";

import { useState } from "react";

/**
 * A command the user is meant to paste somewhere else. Copying is the whole
 * point of the element, so the button is part of the field rather than a
 * separate affordance they have to find.
 */
export default function CopyLine({
  value,
  label,
  tone = "default",
}: {
  value: string;
  label?: string;
  tone?: "default" | "muted";
}) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
    } catch {
      // Clipboard access is blocked over plain HTTP in some browsers. Fall back
      // to selecting the text so the user can still copy it by hand.
      const range = document.createRange();
      const node = document.getElementById(`copy-${hash(value)}`);
      if (node) {
        range.selectNodeContents(node);
        const selection = window.getSelection();
        selection?.removeAllRanges();
        selection?.addRange(range);
      }
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }

  return (
    <div className="stack-sm">
      {label ? <span className="label">{label}</span> : null}
      <div className="copy-line" style={tone === "muted" ? { background: "rgba(0,0,0,0.25)" } : undefined}>
        <code id={`copy-${hash(value)}`}>{value}</code>
        <button type="button" className="btn btn-sm" onClick={copy} aria-label="Copy to clipboard">
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
    </div>
  );
}

function hash(value: string): string {
  let result = 0;
  for (let index = 0; index < value.length; index += 1) {
    result = (result * 31 + value.charCodeAt(index)) | 0;
  }
  return Math.abs(result).toString(36);
}
