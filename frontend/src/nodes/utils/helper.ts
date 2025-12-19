import * as d3 from "d3";
import * as d3Chromatic from "d3-scale-chromatic";
import L from "leaflet";

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

export function tileBoundsFromXYZ(x: number, y: number, z: number, map: L.Map) {
  const tileSize = 256;

  const nwPoint = L.point(x * tileSize, y * tileSize);
  const sePoint = L.point((x + 1) * tileSize, (y + 1) * tileSize);

  const nwLatLng = map.unproject(nwPoint, z);
  const seLatLng = map.unproject(sePoint, z);

  return L.latLngBounds(nwLatLng, seLatLng);
}

export function colorToRgb(str: string): [number, number, number] {
  if (!str) return [0, 0, 0];

  const s = str.trim();

  // Handle hex: #rgb or #rrggbb
  if (s[0] === "#") {
    let hex = s.slice(1);
    if (hex.length === 3) {
      // e.g. "f0a" -> "ff00aa"
      hex = hex
        .split("")
        .map((c) => c + c)
        .join("");
    }
    if (hex.length === 6) {
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      return [
        Number.isFinite(r) ? r : 0,
        Number.isFinite(g) ? g : 0,
        Number.isFinite(b) ? b : 0,
      ];
    }
  }

  // Handle "rgb(r,g,b)" / "rgba(r,g,b,a)"
  if (s.startsWith("rgb")) {
    const nums = s
      .replace(/[^\d,]/g, "")
      .split(",")
      .map((x) => Number(x));
    return [nums[0] ?? 0, nums[1] ?? 0, nums[2] ?? 0];
  }

  // Fallback: try to extract any numbers in order
  const nums = s
    .replace(/[^\d,]/g, "")
    .split(",")
    .map((x) => Number(x));
  return [nums[0] ?? 0, nums[1] ?? 0, nums[2] ?? 0];
}
