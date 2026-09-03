// charts/chartTypes.ts
import { appUrl } from "../utils/runtimePaths";

export type ChartEngine = "d3" | "vega-lite";

export type ChartTypeSummary = {
  name: string;
  // Human-readable name shown in Chart Studio's inputs/dropdown and Chart
  // Gallery - "Bar chart" rather than the real technical `name` ("bar").
  // Falls back to `name` server-side for charts published before this
  // field existed.
  displayName: string;
  description: string;
  createdAt: string | null;
  engine: ChartEngine;
};

export type ChartTypeRecord = ChartTypeSummary & {
  code: string;
  // The "Example usage"/"Sample data" JSON this chart was last published
  // with (see pages/ChartStudioPage.tsx) - absent for a chart published
  // before these were persisted.
  exampleUsage?: unknown;
  sampleData?: unknown;
};

export async function listChartTypes(
  signal?: AbortSignal,
): Promise<ChartTypeSummary[]> {
  const res = await fetch(appUrl("/api/chart-types"), { signal });
  if (!res.ok) {
    throw new Error(`Failed to list chart types: ${res.status}`);
  }
  const data = await res.json();
  return data.chartTypes ?? [];
}

export async function getChartType(
  name: string,
  signal?: AbortSignal,
): Promise<ChartTypeRecord> {
  const res = await fetch(appUrl(`/api/chart-types/${encodeURIComponent(name)}`), {
    signal,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `Unknown chart type: ${name}`);
  }
  return res.json();
}

export async function publishChartType(
  record: {
    // Set only when overwriting a chart Chart Studio already loaded - its
    // existing real identifier. Omitted for a brand-new chart, whose
    // identifier the backend mints from displayName + engine (see
    // server.py's publish_chart_type/_engine_tag).
    name?: string;
    displayName: string;
    description: string;
    code: string;
    engine: ChartEngine;
    exampleUsage: unknown;
    sampleData: unknown;
  },
): Promise<ChartTypeRecord> {
  const res = await fetch(appUrl("/api/chart-types"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(record),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.error ?? `Failed to publish chart type: ${res.status}`);
  }
  return body;
}

// Writes Studio's Sample data as real backend/data/served/metric/<key>.csv
// files - the same shape /api/comparison-view reads - so a chart can be
// tested against real backend data immediately. Overwrites existing CSVs
// for the keys supplied.
export async function seedMetricData(
  data: Record<string, Record<string, number>>,
): Promise<{ written: string[] }> {
  const res = await fetch(appUrl("/api/seed-metric-data"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.error ?? `Failed to push sample data: ${res.status}`);
  }
  return body;
}
