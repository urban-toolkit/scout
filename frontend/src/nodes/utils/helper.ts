import * as d3 from "d3";
import * as d3Chromatic from "d3-scale-chromatic";

// ---- tiny helpers ----
export function pickInterpolator(name?: string) {
  const safe = (name || "Greys").trim();
  if (!safe) {
    return d3.interpolateGreys;
  }

  // "viridis" -> "Viridis", "Reds" -> "Reds"
  const cap = safe.charAt(0).toUpperCase() + safe.slice(1).toLowerCase();
  const key = `interpolate${cap}`; // e.g., interpolateViridis, interpolateReds

  const fromCore = (d3 as any)[key];
  const fromChromatic = (d3Chromatic as any)[key];

  const interpolator = fromCore ?? fromChromatic;

  if (typeof interpolator === "function") {
    return interpolator;
  }

  console.warn(
    `[pickInterpolator] Unknown colormap "${name}", key="${key}". Falling back to Greys.`
  );
  return d3.interpolateGreys;
}

export function getPropertyRangeFromGeoJSON(
  fc: any,
  attr?: string
): [number, number] | null {
  if (!attr) return null;
  const vals: number[] = [];
  for (const f of fc?.features ?? []) {
    const v = Number(f?.properties?.[attr]);
    if (!Number.isNaN(v)) vals.push(v);
  }
  if (!vals.length) return null;
  const [min, max] = d3.extent(vals) as [number, number];
  return min == null || max == null ? null : [min, max];
}
