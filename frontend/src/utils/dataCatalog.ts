import { appUrl } from "./runtimePaths";
import { getCurrentDataflowId } from "./dataflows";

export type CatalogDataset = {
  id: string;
  name: string;
  group: string;
  format: string;
  size: string;
  sizeBytes: number;
  modifiedAt: string;
  // True for a computed entry that's actually a folder of files (e.g. a
  // raster tile-set produced by a compute-catalog script) - absent/false
  // for everything else, including every ordinary data-catalog file (those
  // are never folders). See DataCatalogPanel's Computed tab.
  isDir?: boolean;
};

export async function listDataCatalog(
  signal?: AbortSignal,
): Promise<CatalogDataset[]> {
  const res = await fetch(appUrl("/api/data-catalog"), { signal });
  if (!res.ok) {
    throw new Error(`Failed to list data catalog: ${res.status}`);
  }
  const data = await res.json();
  return data.datasets ?? [];
}

// Lists the tile filenames inside a computed raster folder (e.g.
// "computed/A_raster") - same endpoint the map view itself uses to render
// tiles (see utils/renderViewLayers.ts), reused here purely for display so
// a folder-shaped computed entry can be expanded to show what's inside it.
export async function listComputedRasterTiles(
  ref: string,
  signal?: AbortSignal,
): Promise<string[]> {
  const dataflowId = getCurrentDataflowId();
  const url = new URL(appUrl(`/api/list-rasters/${ref}`), window.location.origin);
  if (dataflowId) url.searchParams.set("dataflow_id", dataflowId);
  const res = await fetch(url.toString(), { signal });
  if (!res.ok) {
    throw new Error(`Failed to list raster tiles: ${res.status}`);
  }
  return res.json();
}
