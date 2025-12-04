// utils/renderComparison.ts
import type { ComparisonDef } from "./types";
import type { ReactNode } from "react";
import { ComparisonBarChart } from "./comparisonBarChart";
import { ComparisonPieChart } from "./comparisonPieChart";
import { ComparisonTable } from "./comparisonTable";

type ComparisonAPIResponse = {
  status: string;
  metric: string;
  encoding: string;
  unit: string;
  values: Record<string, number>;
};

export async function renderComparisonFromDef(
  def: ComparisonDef,
  signal?: AbortSignal
): Promise<ReactNode> {
  console.log("Rendering comparison from definition:", def);

  const res = await fetch("http://localhost:5000/api/comparison-view", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(def),
    signal,
  });

  if (!res.ok) {
    throw new Error(`Comparison fetch failed: ${res.status}`);
  }

  const data: ComparisonAPIResponse = await res.json();

  if (data.encoding === "bar") {
    return (
      <ComparisonBarChart
        values={data.values}
        metric={data.metric}
        unit={data.unit}
      />
    );
  }

  if (data.encoding === "pie") {
    return (
      <ComparisonPieChart
        values={data.values}
        metric={data.metric}
        unit={data.unit}
      />
    );
  }

  if (data.encoding === "table") {
    return (
      <ComparisonTable
        values={data.values}
        metric={data.metric}
        unit={data.unit}
      />
    );
  }

  return (
    <pre style={{ fontSize: "10px" }}>{JSON.stringify(data, null, 2)}</pre>
  );
}
