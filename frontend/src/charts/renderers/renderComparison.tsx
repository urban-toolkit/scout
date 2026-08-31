// charts/renderers/renderComparison.tsx
import type { ComparisonDef } from "../../utils/types";
import type { ReactNode } from "react";
import { GenericChartRenderer } from "./genericChartRenderer";
import { customChartRenderers } from "../generated/customChartRegistry.generated";
import { appUrl } from "../../utils/runtimePaths";

type ComparisonAPIResponse = {
  status: string;
  axis: "x" | "y" | null;
  axisLabel: string | null;
  chart: string;
  props: Record<string, any>;
  data: Record<string, Record<string, number>>;
};

export async function renderComparisonFromDef(
  def: ComparisonDef,
  signal?: AbortSignal,
): Promise<ReactNode> {
  console.log("Rendering comparison from definition:", def);

  const res = await fetch(appUrl("/api/comparison-view"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(def),
    signal,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `Comparison fetch failed: ${res.status}`);
  }

  const data: ComparisonAPIResponse = await res.json();

  // bar/pie/table are ordinary published chart types too - every chart,
  // "default" or user-created, is authored and published through Chart
  // Studio and resolved from the same registry.
  const renderFn = customChartRenderers[data.chart];
  if (!renderFn) {
    throw new Error(`Unknown chart type: ${data.chart}`);
  }

  return (
    <GenericChartRenderer
      renderFn={renderFn}
      data={data.data}
      props={{ ...data.props, axis: data.axis, axisLabel: data.axisLabel }}
    />
  );
}
