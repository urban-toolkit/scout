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

// URL for downloading one catalog entry (file or, zipped on the fly by the
// backend, a raster folder) - a plain <a href> away, no fetch needed. Mirrors
// listComputedRasterTiles' dataflow_id handling since a "computed/..." id
// needs it to resolve on the backend the same way.
export function dataCatalogDownloadUrl(id: string): string {
  const dataflowId = getCurrentDataflowId();
  const url = new URL(appUrl("/api/data-catalog/download"), window.location.origin);
  url.searchParams.set("id", id);
  if (dataflowId) url.searchParams.set("dataflow_id", dataflowId);
  return url.pathname + url.search;
}

// A single File, a .zip archive File, or a raw folder (from
// <input webkitdirectory>) - same three shapes as the Compute Catalog's own
// upload (see ComputeUploadSource), just for arbitrary data files instead
// of Python source.
export type DataUploadSource =
  | { mode: "file"; file: File }
  | { mode: "zip"; file: File }
  | { mode: "folder"; files: File[]; relpaths: string[] };

// Thrown when the backend reports the upload would silently overwrite an
// existing catalog file/folder - callers catch this, ask the user, and
// retry the same upload with confirmOverwrite: true.
export class DataCatalogOverwriteError extends Error {
  existing: string[];
  constructor(existing: string[]) {
    super(`Would overwrite: ${existing.join(", ")}`);
    this.existing = existing;
  }
}

export async function uploadDataCatalogItem(
  destination: string,
  source: DataUploadSource,
  confirmOverwrite = false,
): Promise<CatalogDataset[]> {
  const form = new FormData();
  form.set("mode", source.mode);
  form.set("destination", destination);
  form.set("confirmOverwrite", String(confirmOverwrite));

  if (source.mode === "file" || source.mode === "zip") {
    form.set(source.mode === "file" ? "file" : "archive", source.file);
  } else {
    for (const f of source.files) form.append("files", f);
    form.set("relpaths", JSON.stringify(source.relpaths));
  }

  const res = await fetch(appUrl("/api/data-catalog/upload"), { method: "POST", body: form });
  const body = await res.json().catch(() => ({}));

  if (body?.status === "confirm_overwrite") {
    throw new DataCatalogOverwriteError(body.existing ?? []);
  }
  if (!res.ok) {
    throw new Error(body.error ?? `Failed to upload dataset: ${res.status}`);
  }
  return body.datasets ?? [];
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
