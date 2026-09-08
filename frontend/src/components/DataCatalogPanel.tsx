import { useEffect, useMemo, useState } from "react";
import Paper from "@mui/material/Paper";
import Slide from "@mui/material/Slide";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import TextField from "@mui/material/TextField";
import InputAdornment from "@mui/material/InputAdornment";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import StorageOutlinedIcon from "@mui/icons-material/StorageOutlined";
import FolderOutlinedIcon from "@mui/icons-material/FolderOutlined";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import CloseIcon from "@mui/icons-material/Close";
import SearchIcon from "@mui/icons-material/Search";
import UploadFileOutlinedIcon from "@mui/icons-material/UploadFileOutlined";

import { listDataCatalog, type CatalogDataset } from "../utils/dataCatalog";
import { formatColor } from "../utils/formatColors";
import {
  collectProjectTreeFiles,
  groupProjectDatasetsForDisplay,
  type ProjectDataset,
  type ProjectFolderNode,
} from "../utils/projectDatasets";

const PANEL_WIDTH = 420;

type CatalogTab = "all" | "project" | "computed";

interface FolderNode {
  name: string;
  path: string;
  folders: Map<string, FolderNode>;
  files: CatalogDataset[];
}

function insertIntoTree(root: FolderNode, parts: string[], dataset: CatalogDataset) {
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
  insertIntoTree(root.folders.get(head)!, rest, dataset);
}

function buildTree(datasets: CatalogDataset[]): FolderNode {
  const root: FolderNode = { name: "", path: "", folders: new Map(), files: [] };
  for (const d of datasets) {
    const parts = d.id.split("/");
    parts.pop();
    insertIntoTree(root, parts, d);
  }
  return root;
}

// A container folder like "osm" (which only holds a "chicago" subfolder, no
// files of its own) can still be added as a whole - this is what "all the
// files under here" means for its own Add/Remove button.
function collectAllFiles(folder: FolderNode): CatalogDataset[] {
  const files = [...folder.files];
  for (const child of folder.folders.values()) files.push(...collectAllFiles(child));
  return files;
}


// DatasetFileRow always needs an onAdd handler, but a row rendered with
// inProject=true (e.g. inside an expanded ProjectFolderRow) never actually
// calls it - there's nothing to add, it's already in the project.
function noop() {}

// A line-label-line divider ("---- OSM Data ----") separating one source's
// datasets from another's - shared by the "Browse all" and "In project"
// tabs so a given source is grouped the same way in both.
function SectionDivider({ label }: { label: string }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
      <Box sx={{ flex: 1, height: "1px", bgcolor: "#e5e7eb" }} />
      <Typography
        sx={{
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color: "#94a3b8",
          whiteSpace: "nowrap",
        }}
      >
        {label}
      </Typography>
      <Box sx={{ flex: 1, height: "1px", bgcolor: "#e5e7eb" }} />
    </Box>
  );
}

// Same fixed width/alignment for every add/remove button, folder or file -
// text-only so it reads as a minimal affordance, not a CTA. Both states
// share one width (sized for the longer "Remove from project" label) so
// toggling a row between them never shifts its position.
const TOGGLE_BUTTON_SX_BASE = {
  textTransform: "none" as const,
  fontWeight: 600,
  fontSize: 11,
  lineHeight: 1,
  minWidth: 0,
  width: 136,
  whiteSpace: "nowrap" as const,
  px: 0.5,
  py: 0.5,
  flexShrink: 0,
  justifyContent: "flex-end",
};

// Same styling for both states - only the label changes ("Add to project" /
// "Remove from project"), so this isn't styled as a destructive action.
const ADD_BUTTON_SX = {
  ...TOGGLE_BUTTON_SX_BASE,
  color: "#94a3b8",
  "&:hover": { color: "#334155", bgcolor: "transparent" },
};

// Shared right padding so a folder row's button and a file row's button
// land on the exact same right edge regardless of nesting depth.
const ROW_PR = 1;
// Horizontal space added per nesting level - kept small so nested rows
// stay close to their parent instead of drifting far right.
const INDENT_UNIT = 1.5;
// Constant nudge (not scaled by depth) so a folder's files line up under
// its own folder icon - i.e. depth*INDENT_UNIT + this equals the pixel
// offset of the folder row's icon (chevron + gap + folder icon width),
// converted to spacing units and minus the file card's own inner padding.
const FILE_INDENT_NUDGE = 2.25;

// Renders one file's card. `indent` is the margin-left (in spacing units)
// used to nest it under a folder - 0 for a file that sits at the catalog
// root, alongside top-level folders rather than inside one.
function DatasetFileRow({
  dataset: d,
  indent,
  inProject,
  onAdd,
  onRemove,
}: {
  dataset: CatalogDataset;
  indent: number;
  inProject: boolean;
  onAdd: (datasets: CatalogDataset[]) => void;
  onRemove: (ids: string[]) => void;
}) {
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 1,
        py: 0.75,
        px: ROW_PR,
        // Indenting a bordered box needs margin, not padding - padding only
        // pushes the icon/text inside, leaving the border itself flush
        // against the panel edge.
        ml: indent,
        border: "1px solid #e5e7eb",
        borderRadius: 1.5,
      }}
    >
      <Box
        sx={{
          width: 32,
          height: 32,
          flexShrink: 0,
          borderRadius: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: `${formatColor(d.format)}1a`,
        }}
      >
        <StorageOutlinedIcon sx={{ fontSize: 18, color: formatColor(d.format) }} />
      </Box>
      <Box sx={{ minWidth: 0, flex: 1 }}>
        <Typography
          sx={{
            fontWeight: 500,
            fontSize: 13,
            color: "#0f172a",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
        >
          {d.name}
        </Typography>
        <Typography
          variant="caption"
          sx={{
            color: "#64748b",
            display: "block",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
        >
          {d.format} · {d.size}
        </Typography>
      </Box>
      <Button
        variant="text"
        size="small"
        onClick={() => (inProject ? onRemove([d.id]) : onAdd([d]))}
        sx={ADD_BUTTON_SX}
      >
        {inProject ? "Remove from project" : "Add to project"}
      </Button>
    </Box>
  );
}

// Renders one folder within a tree-shaped project section ("OSM Data" or
// "Other" - see groupProjectDatasetsForDisplay), including any subfolders
// it actually has, rebuilt from each file's real catalog path regardless
// of whether it was added via its own folder button or as an individual
// file - "chicago" always nests the same way. depth 0 is the top-level
// entry (gets the bordered "card" look, like its file-row siblings);
// deeper levels read like the Browse-all tree.
function ProjectTreeFolderRow({
  name,
  node,
  depth,
  onRemove,
}: {
  name: string;
  node: ProjectFolderNode;
  depth: number;
  onRemove: (ids: string[]) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const childFolders = [...node.folders.values()].sort((a, b) => a.name.localeCompare(b.name));
  const files = [...node.files].sort((a, b) => a.name.localeCompare(b.name));
  const allFiles = collectProjectTreeFiles(node);

  return (
    <Box>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1,
          pl: depth * INDENT_UNIT,
          pr: ROW_PR,
          py: 0.5,
        }}
      >
        <Box
          component="button"
          onClick={() => setIsOpen((v) => !v)}
          aria-label={isOpen ? `Collapse ${name}` : `Expand ${name}`}
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 0.75,
            flex: 1,
            minWidth: 0,
            border: "none",
            background: "none",
            p: 0,
            cursor: "pointer",
            textAlign: "left",
          }}
        >
          <ChevronRightIcon
            sx={{
              fontSize: 20,
              color: "#94a3b8",
              transform: isOpen ? "rotate(90deg)" : "rotate(0deg)",
              transition: "transform 0.15s ease",
              flexShrink: 0,
            }}
          />
          <FolderOutlinedIcon sx={{ fontSize: 24, color: "#64748b", flexShrink: 0 }} />
          <Typography
            sx={{
              fontWeight: 500,
              fontSize: 13,
              color: "#0f172a",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {name}
          </Typography>
        </Box>
        <Button
          variant="text"
          size="small"
          onClick={() => onRemove(allFiles.map((f) => f.id))}
          sx={ADD_BUTTON_SX}
        >
          Remove from project
        </Button>
      </Box>

      {isOpen && (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75, mt: 0.5 }}>
          {childFolders.map((child) => (
            <ProjectTreeFolderRow
              key={child.path}
              name={child.name}
              node={child}
              depth={depth + 1}
              onRemove={onRemove}
            />
          ))}
          {files.map((d) => (
            <DatasetFileRow
              key={d.id}
              dataset={d}
              indent={depth * INDENT_UNIT + FILE_INDENT_NUDGE}
              inProject
              onAdd={noop}
              onRemove={onRemove}
            />
          ))}
        </Box>
      )}
    </Box>
  );
}

function FolderRow({
  folder,
  depth,
  expanded,
  onToggle,
  projectDatasetIds,
  onAddToProject,
  onRemoveFromProject,
}: {
  folder: FolderNode;
  depth: number;
  expanded: Set<string>;
  onToggle: (path: string) => void;
  projectDatasetIds: Set<string>;
  onAddToProject: (datasets: CatalogDataset[]) => void;
  onRemoveFromProject: (ids: string[]) => void;
}) {
  const isOpen = expanded.has(folder.path);
  const childFolders = [...folder.folders.values()].sort((a, b) =>
    a.name.localeCompare(b.name),
  );
  const files = [...folder.files].sort((a, b) => a.name.localeCompare(b.name));

  // A container folder like "osm" (which only holds a "chicago" subfolder,
  // no files of its own) can still be added as a whole - "all the files
  // under here" includes every descendant folder's files, not just this
  // folder's own.
  const allFiles = collectAllFiles(folder);
  const allFilesInProject =
    allFiles.length > 0 && allFiles.every((f) => projectDatasetIds.has(f.id));

  return (
    <Box>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1,
          pl: depth * INDENT_UNIT,
          pr: ROW_PR,
          py: 0.5,
        }}
      >
        <Box
          component="button"
          onClick={() => onToggle(folder.path)}
          aria-label={isOpen ? `Collapse ${folder.name}` : `Expand ${folder.name}`}
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 0.75,
            flex: 1,
            minWidth: 0,
            border: "none",
            background: "none",
            p: 0,
            cursor: "pointer",
            textAlign: "left",
          }}
        >
          <ChevronRightIcon
            sx={{
              fontSize: 20,
              color: "#94a3b8",
              transform: isOpen ? "rotate(90deg)" : "rotate(0deg)",
              transition: "transform 0.15s ease",
              flexShrink: 0,
            }}
          />
          <FolderOutlinedIcon sx={{ fontSize: 24, color: "#64748b", flexShrink: 0 }} />
          <Typography
            sx={{
              fontWeight: 500,
              fontSize: 13,
              color: "#0f172a",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {folder.name}
          </Typography>
        </Box>
        {allFiles.length > 0 && (
          <Button
            variant="text"
            size="small"
            onClick={() =>
              allFilesInProject
                ? onRemoveFromProject(allFiles.map((f) => f.id))
                : onAddToProject(allFiles)
            }
            sx={ADD_BUTTON_SX}
          >
            {allFilesInProject ? "Remove from project" : "Add to project"}
          </Button>
        )}
      </Box>

      {isOpen && (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75, mt: 0.5 }}>
          {childFolders.map((child) => (
            <FolderRow
              key={child.path}
              folder={child}
              depth={depth + 1}
              expanded={expanded}
              onToggle={onToggle}
              projectDatasetIds={projectDatasetIds}
              onAddToProject={onAddToProject}
              onRemoveFromProject={onRemoveFromProject}
            />
          ))}

          {files.map((d) => (
            <DatasetFileRow
              key={d.id}
              dataset={d}
              indent={depth * INDENT_UNIT + FILE_INDENT_NUDGE}
              inProject={projectDatasetIds.has(d.id)}
              onAdd={onAddToProject}
              onRemove={onRemoveFromProject}
            />
          ))}
        </Box>
      )}
    </Box>
  );
}

interface Props {
  open: boolean;
  onClose: () => void;
  projectDatasets: ProjectDataset[];
  onAddToProject: (datasets: CatalogDataset[]) => void;
  onRemoveFromProject: (ids: string[]) => void;
}

export default function DataCatalogPanel({
  open,
  onClose,
  projectDatasets,
  onAddToProject,
  onRemoveFromProject,
}: Props) {
  const [tab, setTab] = useState<CatalogTab>("all");
  const [search, setSearch] = useState("");
  const [datasets, setDatasets] = useState<CatalogDataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const projectDatasetIds = useMemo(
    () => new Set(projectDatasets.map((d) => d.id)),
    [projectDatasets],
  );
  const projectSections = useMemo(
    () => groupProjectDatasetsForDisplay(projectDatasets),
    [projectDatasets],
  );
  // A data_layer fetch's output (e.g. "A_buildings.geojson") is added to
  // the same projectDatasets list, tagged group "computed" - this tab is
  // just that slice of it.
  const computedDatasets = useMemo(
    () => projectDatasets.filter((d) => d.group === "computed"),
    [projectDatasets],
  );

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    listDataCatalog(controller.signal)
      .then(setDatasets)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message || "Failed to load datasets.");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [open]);

  const filteredDatasets = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return datasets;
    return datasets.filter(
      (d) =>
        d.name.toLowerCase().includes(query) ||
        d.id.toLowerCase().includes(query) ||
        d.format.toLowerCase().includes(query),
    );
  }, [datasets, search]);

  const tree = useMemo(() => buildTree(filteredDatasets), [filteredDatasets]);
  const rootFolders = useMemo(
    () => [...tree.folders.values()].sort((a, b) => a.name.localeCompare(b.name)),
    [tree],
  );
  // Files that live directly in the catalog root, with no enclosing folder.
  const rootFiles = useMemo(
    () => [...tree.files].sort((a, b) => a.name.localeCompare(b.name)),
    [tree],
  );
  // Same OSM-vs-everything-else split as the "In project" tab (see
  // groupProjectDatasetsForDisplay) - a root folder's own name is already
  // its top-level source, so no extra bookkeeping is needed here.
  const osmRootFolders = useMemo(
    () => rootFolders.filter((f) => f.name === "osm"),
    [rootFolders],
  );
  const otherRootFolders = useMemo(
    () => rootFolders.filter((f) => f.name !== "osm"),
    [rootFolders],
  );

  // While actively searching, show every folder that still has a match
  // expanded, rather than requiring clicks through a tree the user is
  // trying to search past.
  const effectiveExpanded = useMemo(() => {
    if (!search.trim()) return expanded;
    const all = new Set<string>();
    const collect = (folder: FolderNode) => {
      all.add(folder.path);
      folder.folders.forEach(collect);
    };
    tree.folders.forEach(collect);
    return all;
  }, [search, expanded, tree]);

  const toggleFolder = (path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const tabs: { key: CatalogTab; label: string }[] = [
    { key: "all", label: "Browse all" },
    { key: "project", label: "In project" },
    { key: "computed", label: "Computed" },
  ];

  return (
    <Slide direction="left" in={open} mountOnEnter unmountOnExit>
      <Paper
        elevation={8}
        square
        // Lets NodeRail's own "Data Catalog" flyout tell this panel apart
        // from a genuine outside click (see NodeRail.tsx) - clicking
        // anywhere in here shouldn't close that unrelated left-side flyout.
        className="data-catalog-panel"
        sx={{
          position: "fixed",
          top: 0,
          right: 0,
          bottom: 0,
          width: { xs: "100%", sm: PANEL_WIDTH },
          zIndex: 45,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          borderLeft: "1px solid #e5e7eb",
        }}
      >
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            px: 2,
            py: 1.5,
            borderBottom: "1px solid #e5e7eb",
            flexShrink: 0,
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <StorageOutlinedIcon fontSize="small" sx={{ color: "#334155" }} />
            <Typography sx={{ fontWeight: 600, fontSize: 15, color: "#0f172a" }}>
              Data Catalog
            </Typography>
          </Box>
          <IconButton size="small" onClick={onClose} aria-label="Close data catalog">
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>

        <Box sx={{ px: 2, pt: 1.5, flexShrink: 0 }}>
          <Typography variant="body2" sx={{ color: "#64748b" }}>
            Datasets available in this project.
          </Typography>
        </Box>

        <Box sx={{ px: 2, pt: 1.5, flexShrink: 0 }}>
          <TextField
            size="small"
            fullWidth
            placeholder="Search datasets, publishers, tags..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            slotProps={{
              input: {
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon fontSize="small" sx={{ color: "#94a3b8" }} />
                  </InputAdornment>
                ),
              },
            }}
          />
        </Box>

        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 3,
            px: 2,
            pt: 1.5,
            borderBottom: "1px solid #e5e7eb",
            flexShrink: 0,
          }}
        >
          {tabs.map((t) => (
            <Box
              key={t.key}
              component="button"
              onClick={() => setTab(t.key)}
              sx={{
                border: "none",
                background: "none",
                p: 0,
                pb: 1.25,
                fontSize: 14,
                fontFamily: "inherit",
                fontWeight: tab === t.key ? 600 : 500,
                color: tab === t.key ? "#0f172a" : "#94a3b8",
                borderBottom: tab === t.key ? "2px solid #0f172a" : "2px solid transparent",
                cursor: "pointer",
              }}
            >
              {t.label}
              {t.key === "all" && datasets.length > 0 ? ` (${datasets.length})` : ""}
              {t.key === "project" && projectDatasets.length > 0
                ? ` (${projectDatasets.length})`
                : ""}
              {t.key === "computed" && computedDatasets.length > 0
                ? ` (${computedDatasets.length})`
                : ""}
            </Box>
          ))}
        </Box>

        <Box sx={{ flex: 1, overflowY: "auto", p: 2 }}>
          {tab === "computed" ? (
            computedDatasets.length === 0 ? (
              <Typography variant="body2" sx={{ color: "#94a3b8", textAlign: "center", pt: 4 }}>
                Nothing computed yet - fetching a data layer will show its output here.
              </Typography>
            ) : (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                {[...computedDatasets]
                  .sort((a, b) => a.name.localeCompare(b.name))
                  .map((d) => (
                    <DatasetFileRow
                      key={d.id}
                      dataset={d}
                      indent={0}
                      inProject
                      onAdd={onAddToProject}
                      onRemove={onRemoveFromProject}
                    />
                  ))}
              </Box>
            )
          ) : tab === "project" ? (
            projectDatasets.length === 0 ? (
              <Typography variant="body2" sx={{ color: "#94a3b8", textAlign: "center", pt: 4 }}>
                No datasets added to this project yet.
              </Typography>
            ) : (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5 }}>
                {projectSections.map((section) => (
                  <Box key={section.key}>
                    <SectionDivider label={section.label} />
                    <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                      {section.kind === "flat"
                        ? [...section.files]
                            .sort((a, b) => a.name.localeCompare(b.name))
                            .map((d) => (
                              <DatasetFileRow
                                key={d.id}
                                dataset={d}
                                indent={0}
                                inProject
                                onAdd={onAddToProject}
                                onRemove={onRemoveFromProject}
                              />
                            ))
                        : (() => {
                            const folders = [...section.root.folders.values()].sort((a, b) =>
                              a.name.localeCompare(b.name),
                            );
                            const files = [...section.root.files].sort((a, b) =>
                              a.name.localeCompare(b.name),
                            );
                            return (
                              <>
                                {folders.map((folder) => (
                                  <ProjectTreeFolderRow
                                    key={folder.path}
                                    name={folder.name}
                                    node={folder}
                                    depth={0}
                                    onRemove={onRemoveFromProject}
                                  />
                                ))}
                                {files.map((d) => (
                                  <DatasetFileRow
                                    key={d.id}
                                    dataset={d}
                                    indent={0}
                                    inProject
                                    onAdd={onAddToProject}
                                    onRemove={onRemoveFromProject}
                                  />
                                ))}
                              </>
                            );
                          })()}
                    </Box>
                  </Box>
                ))}
              </Box>
            )
          ) : loading ? (
            <Box sx={{ display: "flex", justifyContent: "center", pt: 4 }}>
              <CircularProgress size={22} />
            </Box>
          ) : error ? (
            <Typography variant="body2" color="error" sx={{ textAlign: "center", pt: 4 }}>
              {error}
            </Typography>
          ) : rootFolders.length === 0 && rootFiles.length === 0 ? (
            <Typography variant="body2" sx={{ color: "#94a3b8", textAlign: "center", pt: 4 }}>
              {datasets.length === 0
                ? "No datasets found in the catalog."
                : "No datasets match your search."}
            </Typography>
          ) : (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5 }}>
              {osmRootFolders.length > 0 && (
                <Box>
                  <SectionDivider label="OSM Data" />
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
                    {osmRootFolders.map((folder) => (
                      <FolderRow
                        key={folder.path}
                        folder={folder}
                        depth={0}
                        expanded={effectiveExpanded}
                        onToggle={toggleFolder}
                        projectDatasetIds={projectDatasetIds}
                        onAddToProject={onAddToProject}
                        onRemoveFromProject={onRemoveFromProject}
                      />
                    ))}
                  </Box>
                </Box>
              )}
              {(otherRootFolders.length > 0 || rootFiles.length > 0) && (
                <Box>
                  <SectionDivider label="Other" />
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
                    {otherRootFolders.map((folder) => (
                      <FolderRow
                        key={folder.path}
                        folder={folder}
                        depth={0}
                        expanded={effectiveExpanded}
                        onToggle={toggleFolder}
                        projectDatasetIds={projectDatasetIds}
                        onAddToProject={onAddToProject}
                        onRemoveFromProject={onRemoveFromProject}
                      />
                    ))}
                    {rootFiles.map((d) => (
                      <DatasetFileRow
                        key={d.id}
                        dataset={d}
                        indent={0}
                        inProject={projectDatasetIds.has(d.id)}
                        onAdd={onAddToProject}
                        onRemove={onRemoveFromProject}
                      />
                    ))}
                  </Box>
                </Box>
              )}
            </Box>
          )}
        </Box>

        <Box sx={{ p: 2, borderTop: "1px solid #e5e7eb", flexShrink: 0 }}>
          <Button
            fullWidth
            variant="outlined"
            startIcon={<UploadFileOutlinedIcon />}
            sx={{
              bgcolor: "#fff",
              borderColor: "#cbd5e1",
              color: "#0f172a",
              textTransform: "none",
              fontWeight: 600,
              py: 1,
              "&:hover": { bgcolor: "#f8fafc", borderColor: "#94a3b8" },
            }}
          >
            Import dataset
          </Button>
        </Box>
      </Paper>
    </Slide>
  );
}
