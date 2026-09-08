import type { ProjectDataset } from "./projectDatasets";

// Dragging a dataset (or a whole folder of them) from NodeRail's "Datasets
// in project" list onto the canvas reuses the same dataTransfer key as
// template/code drags (NODE_DRAG_MIME) - this prefix on the *value* is what
// tells App.tsx's drop handler it's a dataset drag rather than a
// TemplateKey or the code-editor sentinel.
const PREFIX = "scout-dataset:";

export type DatasetDragPayload =
  | { kind: "file"; dataset: ProjectDataset }
  | { kind: "folder"; name: string; files: ProjectDataset[] };

export function encodeDatasetDrag(payload: DatasetDragPayload): string {
  return PREFIX + JSON.stringify(payload);
}

export function decodeDatasetDrag(value: string): DatasetDragPayload | null {
  if (!value.startsWith(PREFIX)) return null;
  try {
    return JSON.parse(value.slice(PREFIX.length));
  } catch {
    return null;
  }
}
