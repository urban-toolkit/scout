// Shared between DataCatalogPanel and NodeRail so a dataset's icon color is
// the same wherever it's shown. Reuses the same accent palette NodeRail
// assigns to node categories, so it reads as part of the same system rather
// than a new one-off palette. Colored by file format (matching backend/
// server.py's _catalog_format_label) so every Feather file, every GeoTIFF,
// etc. reads the same regardless of which folder it lives in.
const FORMAT_COLORS: Record<string, string> = {
  Feather: "#238b45",
  Parquet: "#6a51a3",
  CSV: "#1f78b4",
  GeoJSON: "#e6550d",
  JSON: "#ca8a04",
  GeoTIFF: "#cb181d",
  "OSM PBF": "#02818a",
  Pickle: "#a16207",
  "Pickle (gzip)": "#a16207",
};
const FALLBACK_FORMAT_COLORS = [
  "#cb181d",
  "#238b45",
  "#1f78b4",
  "#6a51a3",
  "#e6550d",
  "#02818a",
  "#ca8a04",
];

export function formatColor(format: string): string {
  if (FORMAT_COLORS[format]) return FORMAT_COLORS[format];
  let hash = 0;
  for (let i = 0; i < format.length; i++) hash = (hash * 31 + format.charCodeAt(i)) >>> 0;
  return FALLBACK_FORMAT_COLORS[hash % FALLBACK_FORMAT_COLORS.length];
}
