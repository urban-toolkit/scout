import type { Edge, Node } from "@xyflow/react";
import { appUrl } from "./runtimePaths";
import type { ProjectDataset } from "./projectDatasets";

export type DataflowPreviewNode = {
  id: string;
  type?: string;
  position: { x: number; y: number };
  width?: number | null;
  height?: number | null;
};

export type DataflowPreviewEdge = {
  source?: string;
  target?: string;
};

export type DataflowSummary = {
  id: string;
  name: string;
  createdAt: string | null;
  updatedAt: string | null;
  nodeCount: number;
  nodesPreview: DataflowPreviewNode[];
  edgesPreview: DataflowPreviewEdge[];
};

export type DataflowRecord = {
  id: string;
  name: string;
  nodes: Node[];
  edges: Edge[];
  // Which catalog datasets this dataflow's data_layer nodes are allowed to
  // fetch - a permission list, not a copy of the underlying files.
  projectDatasets: ProjectDataset[];
  createdAt: string | null;
  updatedAt: string | null;
};

// Nodes (e.g. DataLayerNode) that need to know which dataflow they're
// running inside - to send along with a fetch, say - read it straight from
// the URL rather than having it threaded through node data/props.
export function getCurrentDataflowId(): string | null {
  const match = window.location.pathname.match(/\/dataflow\/([^/]+)/);
  return match ? decodeURIComponent(match[1]) : null;
}

export async function listDataflows(signal?: AbortSignal): Promise<DataflowSummary[]> {
  const res = await fetch(appUrl("/api/dataflows"), { signal });
  if (!res.ok) throw new Error(`Failed to list dataflows: ${res.status}`);
  const data = await res.json();
  return data.dataflows ?? [];
}

export async function createDataflow(name?: string): Promise<DataflowRecord> {
  const res = await fetch(appUrl("/api/dataflows"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(`Failed to create dataflow: ${res.status}`);
  return res.json();
}

export async function getDataflow(id: string, signal?: AbortSignal): Promise<DataflowRecord> {
  const res = await fetch(appUrl(`/api/dataflows/${encodeURIComponent(id)}`), { signal });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `Failed to load dataflow: ${res.status}`);
  }
  return res.json();
}

// Fire-and-forget autosave target - failures are swallowed by the caller
// (see App.tsx), which just retries on the next debounced change.
export async function saveDataflow(
  id: string,
  data: { nodes: Node[]; edges: Edge[]; name?: string; projectDatasets?: ProjectDataset[] },
): Promise<void> {
  const res = await fetch(appUrl(`/api/dataflows/${encodeURIComponent(id)}`), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Failed to save dataflow: ${res.status}`);
}

export async function renameDataflow(id: string, name: string): Promise<void> {
  const res = await fetch(appUrl(`/api/dataflows/${encodeURIComponent(id)}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `Failed to rename dataflow: ${res.status}`);
  }
}

export async function deleteDataflow(id: string): Promise<void> {
  const res = await fetch(appUrl(`/api/dataflows/${encodeURIComponent(id)}`), {
    method: "DELETE",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `Failed to delete dataflow: ${res.status}`);
  }
}

// Deletes one of this dataflow's own computed outputs (a projectDataset with
// group "computed", id shaped "computed/<name>") from
// data/dataflows/{dataflowId}_computed/ - unlike a catalog dataset (shared,
// read-only source data), a computed dataset belongs to this dataflow, so
// "remove from project" on one should actually delete its file, not just
// revoke a permission. Any View/Interaction node still referencing it via
// "computed/<name>" will then fail to resolve on next render.
export async function deleteComputedDataset(dataflowId: string, datasetId: string): Promise<void> {
  const prefix = "computed/";
  if (!datasetId.startsWith(prefix)) {
    throw new Error(`Not a computed dataset id: ${datasetId}`);
  }
  const name = datasetId.slice(prefix.length);
  const res = await fetch(
    appUrl(
      `/api/dataflows/${encodeURIComponent(dataflowId)}/computed/${name
        .split("/")
        .map(encodeURIComponent)
        .join("/")}`,
    ),
    { method: "DELETE" },
  );
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `Failed to delete computed dataset: ${res.status}`);
  }
}
