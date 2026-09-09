import type { ProjectComputeItem } from "./dataflows";

// Dragging a Compute Catalog entry onto the canvas reuses the same
// dataTransfer key as dataset/template/code drags (NODE_DRAG_MIME) - this
// prefix on the *value* is what tells App.tsx's drop handler it's a compute
// drag rather than a dataset payload, a TemplateKey, or the code-editor
// sentinel. See datasetDrag.ts for the sibling pattern this mirrors.
const PREFIX = "scout-compute:";

// Always a ProjectComputeItem (not the plain catalog entry shape) - only
// items already added to the project are ever draggable (see
// ComputeCatalogPanel/NodeRail), and that's what carries lastSelection.
export type ComputeItemDragPayload = { entry: ProjectComputeItem };

export function encodeComputeItemDrag(payload: ComputeItemDragPayload): string {
  return PREFIX + JSON.stringify(payload);
}

export function decodeComputeItemDrag(value: string): ComputeItemDragPayload | null {
  if (!value.startsWith(PREFIX)) return null;
  try {
    return JSON.parse(value.slice(PREFIX.length));
  } catch {
    return null;
  }
}
