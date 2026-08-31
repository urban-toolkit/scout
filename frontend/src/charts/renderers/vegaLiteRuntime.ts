// charts/renderers/vegaLiteRuntime.ts
import embed, { type Result } from "vega-embed";

// SCOUT's data contract is Record<scenarioKey, Record<metricField, number>>,
// but Vega-Lite specs expect a flat array of tabular records. This is the
// one place that conversion happens, shared by Studio's live preview and
// every published Vega-Lite chart's generated file.
export function toLongFormatRecords(
  data: Record<string, Record<string, number>>,
): Array<{ key: string; field: string; value: number }> {
  return Object.entries(data).flatMap(([key, row]) =>
    Object.entries(row).map(([field, value]) => ({ key, field, value })),
  );
}

type ContainerState = { generation: number; result: Result | null };
const containerState = new WeakMap<HTMLDivElement, ContainerState>();

// Renders a Vega-Lite spec into `container`, injecting `data` as
// data.values (see toLongFormatRecords). Tracks the previous embed Result
// per container and calls its finalize() before re-embedding - plain
// container.innerHTML="" does NOT do this: it leaves a document-level click
// listener and the Vega View's internal timers/listeners registered, which
// leaks on every re-render (Studio's Preview re-runs on every keystroke).
//
// embed() is async, so a generation counter per container guards against a
// stale, superseded render resolving after a newer one has already started
// (e.g. fast typing in Studio) and finalizing/discarding it immediately
// instead of leaving it live.
export async function renderVegaLiteSpec(
  container: HTMLDivElement,
  spec: Record<string, any>,
  data: Record<string, Record<string, number>>,
  containerWidth: number,
): Promise<void> {
  const state = containerState.get(container) ?? { generation: 0, result: null };
  const myGeneration = state.generation + 1;
  state.generation = myGeneration;
  containerState.set(container, state);

  if (state.result) {
    try {
      state.result.finalize();
    } catch {
      /* ignore */
    }
    state.result = null;
  }

  const values = toLongFormatRecords(data);
  const fullSpec = {
    ...spec,
    data: { values },
    width: spec.width ?? Math.max(containerWidth - 20, 100),
  };

  const result = await embed(container, fullSpec as any, { actions: false });

  const current = containerState.get(container);
  if (!current || current.generation !== myGeneration) {
    // A newer render request superseded this one while embed() was resolving.
    try {
      result.finalize();
    } catch {
      /* ignore */
    }
    return;
  }
  current.result = result;
  result.view.addEventListener("error", (_evt: unknown, info: unknown) => {
    console.error("Vega-Lite runtime error:", info);
  });
}
