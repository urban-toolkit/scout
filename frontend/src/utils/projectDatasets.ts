import type { CatalogDataset } from "./dataCatalog";

// A dataset that's been added to the current dataflow's project.
export type ProjectDataset = CatalogDataset;

// A folder as it appears within one section of the project list, rebuilt
// from each file's own catalog id - so a file added individually (not via
// its folder's own "Add to project" button) still nests under the same
// folder structure as everything else, exactly like the Browse-all tree.
export interface ProjectFolderNode {
  name: string;
  path: string;
  folders: Map<string, ProjectFolderNode>;
  files: ProjectDataset[];
}

function insertIntoProjectTree(root: ProjectFolderNode, parts: string[], dataset: ProjectDataset) {
  if (parts.length === 0) {
    root.files.push(dataset);
    return;
  }
  const [head, ...rest] = parts;
  if (!root.folders.has(head)) {
    root.folders.set(head, {
      name: head,
      path: root.path ? `${root.path}/${head}` : head,
      folders: new Map(),
      files: [],
    });
  }
  insertIntoProjectTree(root.folders.get(head)!, rest, dataset);
}

// `stripPrefix` drops a leading path segment shared by every dataset in
// the section (e.g. "osm/") before building the tree, so the section's own
// header ("OSM Data") isn't immediately followed by a redundant "osm"
// folder row repeating the same thing.
function buildProjectTree(datasets: ProjectDataset[], stripPrefix?: string): ProjectFolderNode {
  const root: ProjectFolderNode = { name: "", path: "", folders: new Map(), files: [] };
  for (const d of datasets) {
    const relativeId =
      stripPrefix && d.id.startsWith(stripPrefix) ? d.id.slice(stripPrefix.length) : d.id;
    const parts = relativeId.split("/");
    parts.pop();
    insertIntoProjectTree(root, parts, d);
  }
  return root;
}

export function collectProjectTreeFiles(node: ProjectFolderNode): ProjectDataset[] {
  const files = [...node.files];
  for (const child of node.folders.values()) files.push(...collectProjectTreeFiles(child));
  return files;
}

export type ProjectSection =
  | { kind: "tree"; key: "osm" | "other"; label: string; root: ProjectFolderNode }
  | { kind: "flat"; key: "computed"; label: string; files: ProjectDataset[] };

// Splits the project list into exactly three cards: OSM data (its own
// folder structure, "osm/" stripped since the section header already says
// so), Computed (flat - a data_layer fetch's output has no meaningful
// folder to nest under), and Other (everything else, its own folder
// structure). A section is omitted entirely if it would be empty.
export function groupProjectDatasetsForDisplay(items: ProjectDataset[]): ProjectSection[] {
  const osmItems = items.filter((d) => d.group === "osm");
  const computedItems = items.filter((d) => d.group === "computed");
  const otherItems = items.filter((d) => d.group !== "osm" && d.group !== "computed");

  const sections: ProjectSection[] = [];
  if (osmItems.length > 0) {
    sections.push({ kind: "tree", key: "osm", label: "OSM Data", root: buildProjectTree(osmItems, "osm/") });
  }
  if (computedItems.length > 0) {
    sections.push({ kind: "flat", key: "computed", label: "Computed", files: computedItems });
  }
  if (otherItems.length > 0) {
    sections.push({ kind: "tree", key: "other", label: "Other", root: buildProjectTree(otherItems) });
  }
  return sections;
}
