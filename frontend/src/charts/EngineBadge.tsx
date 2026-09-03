// charts/EngineBadge.tsx
import type { ChartEngine } from "./chartTypes";
import "./EngineBadge.css";

const ENGINE_LABEL: Record<ChartEngine, string> = {
  d3: "D3",
  "vega-lite": "Vega-Lite",
};

// Small colored pill identifying which engine renders a chart - the one
// place engine identity is called out at all, so the rest of the gallery's
// titles/descriptions can stay implementation-agnostic (see
// charts/galleryExamples.ts) while still letting engine be told apart at a
// glance, by color, across the gallery grid and a chart's own page.
export function EngineBadge({ engine }: { engine: ChartEngine }) {
  return (
    <span className={`engine-badge engine-badge--${engine}`}>
      {ENGINE_LABEL[engine]}
    </span>
  );
}
