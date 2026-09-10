// Resolves whatever was dropped onto (or picked via) a single import
// dropzone into the same {mode, file(s)} shape both DataUploadSource and
// ComputeUploadSource already use - one .zip becomes a "zip" upload, a
// lone non-zip file becomes a "file" upload, and anything else (multiple
// loose files, or a real folder with subdirectories) becomes a "folder"
// upload, since that mode already accepts an arbitrary files+relpaths
// batch. This is the one place that has to know how to tell a dropped
// folder apart from a dropped file - every caller just gets a mode back.
export type ResolvedDrop =
  | { mode: "file"; file: File }
  | { mode: "zip"; file: File }
  | {
      mode: "folder";
      files: File[];
      relpaths: string[];
      // The folder's own name, when a single real folder (not just several
      // loose files) was the thing dropped/picked - callers use this to
      // pre-fill a destination field, purely a suggestion.
      suggestedDestination?: string;
    };

type FlatFile = { file: File; relativePath: string };

function readAllDirectoryEntries(reader: any): Promise<any[]> {
  return new Promise((resolve, reject) => {
    const all: any[] = [];
    const readBatch = () => {
      reader.readEntries((entries: any[]) => {
        if (entries.length === 0) {
          resolve(all);
          return;
        }
        // The directory-reading API caps entries per readEntries() call
        // (historically ~100) - keep calling until it comes back empty.
        all.push(...entries);
        readBatch();
      }, reject);
    };
    readBatch();
  });
}

function readEntry(entry: any, prefix: string): Promise<FlatFile[]> {
  return new Promise((resolve, reject) => {
    if (entry.isFile) {
      entry.file(
        (file: File) => resolve([{ file, relativePath: `${prefix}${file.name}` }]),
        reject,
      );
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      readAllDirectoryEntries(reader)
        .then((entries) => Promise.all(entries.map((e) => readEntry(e, `${prefix}${entry.name}/`))))
        .then((nested) => resolve(nested.flat()))
        .catch(reject);
    } else {
      resolve([]);
    }
  });
}

function classify(flat: FlatFile[]): ResolvedDrop | null {
  if (flat.length === 0) return null;

  const hasNesting = flat.some((f) => f.relativePath.includes("/"));
  if (!hasNesting && flat.length === 1) {
    const file = flat[0].file;
    return file.name.toLowerCase().endsWith(".zip") ? { mode: "zip", file } : { mode: "file", file };
  }
  if (!hasNesting) {
    // Multiple loose files with no shared folder - nothing to suggest as a
    // destination, they just land wherever the user types (or the root).
    return { mode: "folder", files: flat.map((f) => f.file), relpaths: flat.map((f) => f.relativePath) };
  }

  // A single shared top-level segment across every path means one real
  // folder was dropped/picked as a whole - strip it (same convention the
  // <input webkitdirectory> picker already used) so its files land
  // directly under the destination, not nested one level deeper again.
  const firstSegment = flat[0].relativePath.split("/")[0];
  const allShareIt = flat.every((f) => f.relativePath.startsWith(`${firstSegment}/`));
  const relpaths = allShareIt
    ? flat.map((f) => f.relativePath.slice(firstSegment.length + 1))
    : flat.map((f) => f.relativePath);
  return {
    mode: "folder",
    files: flat.map((f) => f.file),
    relpaths,
    suggestedDestination: allShareIt ? firstSegment : undefined,
  };
}

// Handles a real drop event - walks any dropped directories recursively via
// the (non-standard but universally supported) webkitGetAsEntry API, with a
// plain-FileList fallback for a browser/context that lacks it (drag-and-drop
// of files still works there, just not of folders).
export async function resolveDroppedItems(dataTransfer: DataTransfer): Promise<ResolvedDrop | null> {
  const items = dataTransfer.items;
  const canReadEntries = items && items.length > 0 && typeof items[0].webkitGetAsEntry === "function";
  if (!canReadEntries) {
    const files = Array.from(dataTransfer.files);
    return classify(files.map((file) => ({ file, relativePath: file.name })));
  }
  const entries = Array.from(items)
    .map((it) => it.webkitGetAsEntry())
    .filter((e): e is any => e != null);
  const nested = await Promise.all(entries.map((e) => readEntry(e, "")));
  return classify(nested.flat());
}

// Handles a plain <input type="file" multiple> selection (the "Browse
// files" fallback) - no directory entries possible here, just a flat
// FileList, so this never returns "folder" with a suggestedDestination.
export function resolvePickedFiles(fileList: FileList): ResolvedDrop | null {
  const files = Array.from(fileList);
  return classify(files.map((file) => ({ file, relativePath: file.name })));
}

// Handles a <input type="file" webkitdirectory multiple> selection (the
// "Browse a folder" fallback) - webkitRelativePath already carries the
// picked folder's own name as the first segment.
export function resolvePickedFolder(fileList: FileList): ResolvedDrop | null {
  const files = Array.from(fileList);
  return classify(
    files.map((file) => ({
      file,
      relativePath: (file as any).webkitRelativePath || file.name,
    })),
  );
}
