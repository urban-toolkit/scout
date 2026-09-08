import { appUrl } from "./runtimePaths";

export type CatalogDataset = {
  id: string;
  name: string;
  group: string;
  format: string;
  size: string;
  sizeBytes: number;
  modifiedAt: string;
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
