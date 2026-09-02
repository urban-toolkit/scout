// charts/deriveRenderParams.ts
// Shared by Chart Studio's live Preview and Chart Gallery's example pages -
// both need to turn a comparison-node-shaped "example usage" JSON (key,
// x/y, props) into the render params a chart's actual render function
// receives (axis/axisLabel/props).

// Mirrors backend server.py's _as_field_list: a bare string is shorthand
// for a 1-item set.
export function asFieldList(value: unknown): string[] {
  if (value == null) return [];
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value as string[];
  return [];
}

// Mirrors backend server.py's comparison_view() derivation of axis/axisLabel
// from a ComparisonDef's x/y fields, and renderComparison.tsx's merge of
// props with axis/axisLabel - so a rendered preview matches what a real
// Comparison node would actually pass to a chart's render function.
export function deriveRenderParams(example: unknown): {
  axis: "x" | "y";
  axisLabel: string;
  props: Record<string, any>;
} {
  const def = (example ?? {}) as {
    key?: unknown;
    x?: string | string[];
    y?: string | string[];
    props?: Record<string, any>;
  };

  if (!Array.isArray(def.key) || def.key.length === 0) {
    throw new Error('Example usage must include a non-empty "key" array.');
  }

  const xFields = asFieldList(def.x);
  const yFields = asFieldList(def.y);
  const fields = [...xFields, ...yFields];
  if (fields.length === 0) {
    throw new Error('Example usage must declare at least one field via "x" or "y".');
  }

  return {
    axis: xFields.length > 0 ? "x" : "y",
    axisLabel: fields[0],
    props: def.props ?? {},
  };
}

// Filters a data object down to just the rows named in an example usage's
// "key" array - mirrors the real /api/comparison-view endpoint, which only
// ever hands a chart the scenarios a Comparison node actually asked for, not
// every scenario that happens to exist. Chart Gallery's live-editable
// example otherwise had no consequence for adding/removing a "key" entry
// (every render function iterates whatever `data` it receives in full) and
// silently ignored a typo'd key instead of surfacing it.
export function filterDataByExampleKeys(
  data: Record<string, Record<string, number>>,
  example: unknown,
): Record<string, Record<string, number>> {
  const def = (example ?? {}) as { key?: unknown };

  if (!Array.isArray(def.key) || def.key.length === 0) {
    throw new Error('Example usage must include a non-empty "key" array.');
  }

  const missing = def.key.filter((k) => !(typeof k === "string" && k in data));
  if (missing.length > 0) {
    throw new Error(
      `Unknown scenario key${missing.length > 1 ? "s" : ""} in "key": ${missing
        .map((k) => JSON.stringify(k))
        .join(", ")} - not present in the sample data.`,
    );
  }

  return Object.fromEntries((def.key as string[]).map((k) => [k, data[k]]));
}

// Confirms a "x"/"y" field name (as resolved by deriveRenderParams into
// axisLabel) actually exists on the scenario rows it'll be read from -
// otherwise every chart just plots `undefined` for a typo'd field name
// instead of surfacing the mistake.
export function assertFieldExistsInData(
  data: Record<string, Record<string, number>>,
  field: string,
): void {
  const missingIn = Object.entries(data)
    .filter(([, row]) => !(field in row))
    .map(([key]) => key);

  if (missingIn.length > 0) {
    throw new Error(
      `Unknown field "${field}" in "x"/"y" - not present in the sample data for: ${missingIn.join(", ")}.`,
    );
  }
}
