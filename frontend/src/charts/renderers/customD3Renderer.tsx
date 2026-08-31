// charts/renderers/customD3Renderer.tsx
import * as d3 from "d3";
import { useEffect, useRef, useState } from "react";

// Runs an author-supplied D3/JS function body against `data`/`params`.
//
// Known, deliberately-accepted limitations for this first slice:
// - No sandboxing (iframe/worker/CSP): the code runs directly in the main
//   frame with full DOM/page access. Fine for a single trusted team
//   authoring its own charts; must be revisited before wider/shared use.
// - No execution timeout: a synchronous infinite loop in the author's code
//   will hang the tab. There is no worker/iframe boundary here to interrupt it.
export function CustomD3Renderer({
  code,
  data,
  params,
  onResult,
}: {
  code: string;
  data: Record<string, Record<string, number>>;
  params?: Record<string, any>;
  // Reports whether the last render attempt compiled and ran without
  // throwing - used by ChartStudio to gate Publish on a successful Preview.
  onResult?: (ok: boolean) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Tracked the same way GenericChartRenderer tracks it for published charts,
  // so Preview behaves identically to real usage once published.
  const [width, setWidth] = useState(400);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setWidth(entry.contentRect.width);
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    el.innerHTML = "";
    setError(null);

    try {
      // "use strict" so a successful Preview also rules out sloppy-mode-only
      // syntax (e.g. `with`) that a real .ts module could never parse once
      // this code is published as a generated source file.
      const fn = new Function(
        "container",
        "data",
        "params",
        "d3",
        "containerWidth",
        `"use strict";\n${code}`,
      );
      fn(el, data, params ?? {}, d3, width);
      onResult?.(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to render custom chart");
      onResult?.(false);
    }
  }, [code, data, params, onResult, width]);

  return (
    <div style={{ width: "100%" }}>
      {error && (
        <div
          style={{
            color: "#b91c1c",
            fontSize: 12,
            fontFamily: "monospace",
            whiteSpace: "pre-wrap",
            marginBottom: 8,
          }}
        >
          {error}
        </div>
      )}
      <div ref={containerRef} style={{ width: "100%", overflow: "auto" }} />
    </div>
  );
}
