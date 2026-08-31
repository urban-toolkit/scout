// charts/renderers/genericChartRenderer.tsx
import * as d3 from "d3";
import { useEffect, useRef, useState } from "react";

export type ChartRenderFn = (
  container: HTMLDivElement,
  data: Record<string, Record<string, number>>,
  params: Record<string, any>,
  d3lib: typeof d3,
  containerWidth: number,
) => void | Promise<void>;

// Runs a published (compiled, statically-imported) chart's render function -
// the same container-reset / error-isolation contract as CustomD3Renderer,
// but calling a real function instead of eval'ing a code string. Tracks
// container width via ResizeObserver, so a chart's render function gets
// called again on resize - a chart only becomes responsive by reading
// containerWidth, but every chart gets re-invoked automatically, with no
// extra code required to opt in.
export function GenericChartRenderer({
  renderFn,
  data,
  props,
}: {
  renderFn: ChartRenderFn;
  data: Record<string, Record<string, number>>;
  props?: Record<string, any>;
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

    // renderFn may be synchronous (D3 charts) or return a Promise (e.g.
    // Vega-Lite charts, whose rendering is inherently async) - await
    // either way so a rejected/thrown render is always caught here, not
    // just synchronous throws. `cancelled` guards against setting state
    // for a render an effect re-run has already superseded.
    (async () => {
      try {
        await renderFn(el, data, props ?? {}, d3, width);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to render chart");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [renderFn, data, props, width]);

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
