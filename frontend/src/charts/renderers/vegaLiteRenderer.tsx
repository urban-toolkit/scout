// charts/renderers/vegaLiteRenderer.tsx
import { useEffect, useRef, useState } from "react";
import { renderVegaLiteSpec } from "./vegaLiteRuntime";

// Runs an author-supplied Vega-Lite spec against `data` (auto-injected as
// data.values = [{key, field, value}, ...] - see vegaLiteRuntime.ts) for
// Chart Studio's live Preview. Mirrors CustomD3Renderer's shape (ref +
// ResizeObserver-tracked width + reset-before-render + onResult callback for
// the Publish gate), but properly awaits/catches renderVegaLiteSpec's
// Promise instead of eval'ing a code string.
export function VegaLiteRenderer({
  spec,
  data,
  onResult,
}: {
  spec: unknown;
  data: Record<string, Record<string, number>>;
  // Reports whether the last render attempt compiled and ran without
  // throwing - used by ChartStudio to gate Publish on a successful Preview.
  onResult?: (ok: boolean) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [error, setError] = useState<string | null>(null);
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

    let cancelled = false;

    el.innerHTML = "";
    setError(null);

    (async () => {
      try {
        await renderVegaLiteSpec(el, spec as Record<string, any>, data, width);
        if (!cancelled) onResult?.(true);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to render Vega-Lite spec");
          onResult?.(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [spec, data, onResult, width]);

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
