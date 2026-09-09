import { appUrl } from "./runtimePaths";

export type ComputeParam = {
  name: string;
  hasDefault: boolean;
  defaultRepr: string | null;
  // Source-text repr of the param's type annotation (e.g. "int", "str",
  // "Optional[float]"), or null if it has none - most uploaded functions
  // won't. See utils/computeParamTypes.ts for how this gets used.
  type: string | null;
};

export type ComputeCallable =
  | {
      kind: "function";
      name: string;
      importPath: string;
      sourceFile: string;
      docstring: string;
      params: ComputeParam[];
    }
  | {
      kind: "class";
      name: string;
      importPath: string;
      sourceFile: string;
      docstring: string;
      ctorParams: ComputeParam[];
      methods: { name: string; docstring: string; params: ComputeParam[] }[];
    };

export type ComputeCatalogEntry = {
  id: string;
  displayName: string;
  description: string;
  kind: "file" | "package";
  callables: ComputeCallable[];
  createdAt: string;
};

export async function listComputeCatalog(signal?: AbortSignal): Promise<ComputeCatalogEntry[]> {
  const res = await fetch(appUrl("/api/compute-catalog"), { signal });
  if (!res.ok) {
    throw new Error(`Failed to list compute catalog: ${res.status}`);
  }
  const data = await res.json();
  return data.items ?? [];
}

export type ComputeUploadMode = "file" | "zip" | "folder";

// A single .py File, or a .zip archive File - both go straight into
// FormData as one part. A raw folder (from <input webkitdirectory>) is a
// list of Files whose relative paths (File.webkitRelativePath) must be sent
// alongside them, since FormData itself doesn't carry directory structure.
export type ComputeUploadSource =
  | { mode: "file"; file: File }
  | { mode: "zip"; file: File }
  | { mode: "folder"; files: File[]; relpaths: string[] };

export async function uploadComputeItem(
  displayName: string,
  description: string,
  source: ComputeUploadSource,
): Promise<ComputeCatalogEntry> {
  const form = new FormData();
  form.set("mode", source.mode);
  form.set("displayName", displayName);
  form.set("description", description);

  if (source.mode === "file" || source.mode === "zip") {
    form.set(source.mode === "file" ? "file" : "archive", source.file);
  } else {
    for (const f of source.files) form.append("files", f);
    form.set("relpaths", JSON.stringify(source.relpaths));
  }

  const res = await fetch(appUrl("/api/compute-catalog"), { method: "POST", body: form });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.error ?? `Failed to upload model: ${res.status}`);
  }
  return body;
}

// Renames the entry's display name (and optionally its description) -
// doesn't touch its id/slug, so every callable's importPath (e.g.
// "compute.raster_conversion...") stays exactly as it was.
export async function renameComputeItem(
  id: string,
  displayName: string,
  description?: string,
): Promise<ComputeCatalogEntry> {
  const res = await fetch(appUrl(`/api/compute-catalog/${encodeURIComponent(id)}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ displayName, ...(description !== undefined ? { description } : {}) }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.error ?? `Failed to rename '${id}': ${res.status}`);
  }
  return body;
}

// Re-runs introspection against whatever is currently on disk for this
// entry - lets a hand-edit made directly to an already-uploaded file (e.g.
// removing a duplicate function) show up without re-uploading from scratch.
export async function refreshComputeItem(id: string): Promise<ComputeCatalogEntry> {
  const res = await fetch(appUrl(`/api/compute-catalog/${encodeURIComponent(id)}/refresh`), {
    method: "POST",
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.error ?? `Failed to refresh '${id}': ${res.status}`);
  }
  return body;
}
