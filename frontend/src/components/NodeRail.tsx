import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode, CSSProperties, DragEvent } from "react";
import { useReactFlow } from "@xyflow/react";
import Tooltip from "@mui/material/Tooltip";
import LayersOutlinedIcon from "@mui/icons-material/LayersOutlined";
import MergeTypeOutlinedIcon from "@mui/icons-material/MergeTypeOutlined";
import CodeOutlinedIcon from "@mui/icons-material/CodeOutlined";
import MapOutlinedIcon from "@mui/icons-material/MapOutlined";
import TouchAppOutlinedIcon from "@mui/icons-material/TouchAppOutlined";
import TuneOutlinedIcon from "@mui/icons-material/TuneOutlined";
import BarChartOutlinedIcon from "@mui/icons-material/BarChartOutlined";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import CloseIcon from "@mui/icons-material/Close";
import StorageOutlinedIcon from "@mui/icons-material/StorageOutlined";
import FolderOutlinedIcon from "@mui/icons-material/FolderOutlined";
import SkipNextIcon from "@mui/icons-material/SkipNext";
import CheckIcon from "@mui/icons-material/Check";
import CircularProgress from "@mui/material/CircularProgress";

import { TEMPLATE_LABELS, type TemplateKey } from "../templates";
import { runDataflow } from "../utils/dataflowRunner";
import { formatColor } from "../utils/formatColors";
import {
  collectProjectTreeFiles,
  groupProjectDatasetsForDisplay,
  type ProjectDataset,
  type ProjectFolderNode,
} from "../utils/projectDatasets";
import { encodeDatasetDrag, type DatasetDragPayload } from "../utils/datasetDrag";
import { encodeComputeItemDrag } from "../utils/computeItemDrag";
import { listComputedRasterTiles } from "../utils/dataCatalog";
import type { ProjectComputeItem } from "../utils/dataflows";
import "./NodeRail.css";

interface Props {
  onAdd: (tpl: TemplateKey) => void;
  onAddPyCodeEditor: () => void;
  onOpenDataCatalog: () => void;
  onOpenComputeCatalog: () => void;
  projectDatasets: ProjectDataset[];
  projectCompute: ProjectComputeItem[];
}

// dataTransfer key + sentinel the canvas' onDrop reads to tell a dragged
// rail icon apart from any other drag source (e.g. browser text/image drags)
// - "code" isn't a TemplateKey (it maps to onAddPyCodeEditor, not onAdd), so
// it needs its own sentinel value distinct from every real TemplateKey.
export const NODE_DRAG_MIME = "application/x-scout-node";
export const PY_CODE_DRAG_VALUE = "__pyCodeEditor__";

function startDatasetDrag(e: DragEvent<HTMLElement>, payload: DatasetDragPayload) {
  e.dataTransfer.setData(NODE_DRAG_MIME, encodeDatasetDrag(payload));
  e.dataTransfer.effectAllowed = "move";
}

interface RailItem {
  key: string;
  label: string;
  icon: ReactNode;
  onClick: () => void;
  dragValue: string;
}

interface RailSection {
  title: string;
  accent: string;
  hoverBg: string;
  items: RailItem[];
}

const ICON_SX = { fontSize: 18 };

// One folder within a tree-shaped project section ("OSM Data" or "Other" -
// see groupProjectDatasetsForDisplay). The header always shows (name + file
// count), and clicking it reveals its contents underneath: any subfolders
// it actually has, rebuilt from each file's real catalog path regardless
// of whether it was added via its own folder button or as an individual
// file, plus its own direct files.
function ProjectFolderItem({
  name,
  node,
  depth = 0,
}: {
  name: string;
  node: ProjectFolderNode;
  depth?: number;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const childFolders = [...node.folders.values()].sort((a, b) => a.name.localeCompare(b.name));
  const files = [...node.files].sort((a, b) => a.name.localeCompare(b.name));
  const fileCount = collectProjectTreeFiles(node).length;

  return (
    <div className="node-rail-catalog__project-group">
      <button
        type="button"
        className="node-rail-catalog__project-item node-rail-catalog__project-item--toggle"
        onClick={() => setIsOpen((v) => !v)}
        aria-label={isOpen ? `Collapse ${name}` : `Expand ${name}`}
        title={name}
        draggable
        onDragStart={(e) =>
          startDatasetDrag(e, { kind: "folder", name, files: collectProjectTreeFiles(node) })
        }
      >
        <div className="node-rail-catalog__project-item-icon node-rail-catalog__project-item-icon--folder">
          <FolderOutlinedIcon sx={{ fontSize: 16, color: "#64748b" }} />
        </div>
        <div className="node-rail-catalog__project-item-text">
          <div className="node-rail-catalog__project-item-name">{name}</div>
          <div className="node-rail-catalog__project-item-meta">
            {fileCount} file{fileCount === 1 ? "" : "s"}
          </div>
        </div>
        {isOpen ? (
          <ChevronLeftIcon sx={{ fontSize: 15, color: "#94a3b8", flexShrink: 0 }} />
        ) : (
          <ChevronRightIcon sx={{ fontSize: 15, color: "#94a3b8", flexShrink: 0 }} />
        )}
      </button>

      {isOpen && (
        <div className="node-rail-catalog__project-group-files">
          {childFolders.map((child) => (
            <ProjectFolderItem key={child.path} name={child.name} node={child} depth={depth + 1} />
          ))}
          {files.map((f) => (
            <ProjectFileItem key={f.id} dataset={f} />
          ))}
        </div>
      )}
    </div>
  );
}

// One tile inside an expanded raster folder (see ProjectFileItem below) -
// same row look as a real dataset item, minus dragging, since nothing in
// the app references a single tile individually.
function TileProjectItem({ name }: { name: string }) {
  return (
    <div className="node-rail-catalog__project-item" title={name}>
      <div
        className="node-rail-catalog__project-item-icon"
        style={{ backgroundColor: `${formatColor("PNG")}1a` }}
      >
        <StorageOutlinedIcon sx={{ fontSize: 16, color: formatColor("PNG") }} />
      </div>
      <div className="node-rail-catalog__project-item-text">
        <div className="node-rail-catalog__project-item-name">{name}</div>
        <div className="node-rail-catalog__project-item-meta">PNG</div>
      </div>
    </div>
  );
}

// One dataset row in the flyout's project list - used for a loose file at
// a section's root (no folder to nest under) and for a computed output
// (the Computed section is deliberately flat, no folder tree at all) -
// except a computed entry that's actually a raster tile-set (see
// CatalogDataset.isDir), which renders expandable exactly like
// ProjectFolderItem above, listing its tiles via the same endpoint the map
// view itself uses to fetch them.
function ProjectFileItem({ dataset: f }: { dataset: ProjectDataset }) {
  const [isOpen, setIsOpen] = useState(false);
  const [tiles, setTiles] = useState<string[] | null>(null);
  const [loadingTiles, setLoadingTiles] = useState(false);

  if (f.isDir) {
    const toggle = () => {
      const next = !isOpen;
      setIsOpen(next);
      if (next && tiles === null && !loadingTiles) {
        setLoadingTiles(true);
        listComputedRasterTiles(f.id)
          .then(setTiles)
          .catch(() => setTiles([]))
          .finally(() => setLoadingTiles(false));
      }
    };
    return (
      <div className="node-rail-catalog__project-group">
        <button
          type="button"
          className="node-rail-catalog__project-item node-rail-catalog__project-item--toggle"
          onClick={toggle}
          aria-label={isOpen ? `Collapse ${f.name}` : `Expand ${f.name}`}
          title={f.name}
        >
          <div className="node-rail-catalog__project-item-icon node-rail-catalog__project-item-icon--folder">
            <FolderOutlinedIcon sx={{ fontSize: 16, color: "#64748b" }} />
          </div>
          <div className="node-rail-catalog__project-item-text">
            <div className="node-rail-catalog__project-item-name">{f.name}</div>
            <div className="node-rail-catalog__project-item-meta">{f.size}</div>
          </div>
          {isOpen ? (
            <ChevronLeftIcon sx={{ fontSize: 15, color: "#94a3b8", flexShrink: 0 }} />
          ) : (
            <ChevronRightIcon sx={{ fontSize: 15, color: "#94a3b8", flexShrink: 0 }} />
          )}
        </button>

        {isOpen && (
          <div className="node-rail-catalog__project-group-files">
            {loadingTiles ? (
              <div className="node-rail-catalog__project-empty">Loading…</div>
            ) : (tiles ?? []).length === 0 ? (
              <div className="node-rail-catalog__project-empty">No tiles found.</div>
            ) : (
              (tiles ?? []).map((t) => <TileProjectItem key={t} name={t} />)
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      className="node-rail-catalog__project-item"
      title={f.name}
      draggable
      onDragStart={(e) => startDatasetDrag(e, { kind: "file", dataset: f })}
    >
      <div
        className="node-rail-catalog__project-item-icon"
        style={{ backgroundColor: `${formatColor(f.format)}1a` }}
      >
        <StorageOutlinedIcon sx={{ fontSize: 16, color: formatColor(f.format) }} />
      </div>
      <div className="node-rail-catalog__project-item-text">
        <div className="node-rail-catalog__project-item-name">{f.name}</div>
        <div className="node-rail-catalog__project-item-meta">
          {f.format} · {f.size}
        </div>
      </div>
    </div>
  );
}

// One compute-catalog entry in the flyout's project list - mirrors
// ProjectFileItem above. Only items already added to the project ever show
// up here (there is no "browse everything" list in this flyout, same as
// the Data Catalog one), so anything draggable here is by definition
// already an "added" item - dragging from ComputeCatalogPanel's own browse
// list is intentionally not supported, this flyout is the only drag source.
function ComputeProjectItem({ entry }: { entry: ProjectComputeItem }) {
  return (
    <div
      className="node-rail-catalog__project-item"
      title={entry.displayName}
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData(NODE_DRAG_MIME, encodeComputeItemDrag({ entry }));
        e.dataTransfer.effectAllowed = "move";
      }}
    >
      <div
        className="node-rail-catalog__project-item-icon"
        style={{ backgroundColor: "#fee2e2" }}
      >
        <CodeOutlinedIcon sx={{ fontSize: 16 }} style={{ color: "#cb181d" }} />
      </div>
      <div className="node-rail-catalog__project-item-text">
        <div className="node-rail-catalog__project-item-name">{entry.displayName}</div>
        <div className="node-rail-catalog__project-item-meta">
          {entry.kind === "package" ? "Package" : "Script"}
        </div>
      </div>
    </div>
  );
}

export default function NodeRail({
  onAdd,
  onAddPyCodeEditor,
  onOpenDataCatalog,
  onOpenComputeCatalog,
  projectDatasets,
  projectCompute,
}: Props) {
  const { screenToFlowPosition, getNodes, getEdges } = useReactFlow();
  // Only the toggle buttons themselves open/close these - no outside-click
  // auto-close, since canvas interactions (selecting a node, renaming the
  // dataflow, etc.) all count as "outside" and shouldn't fold it away. A
  // single slot (rather than two independent booleans) keeps the Data and
  // Compute flyouts mutually exclusive - both anchor to the same position
  // (see .node-rail-catalog__flyout), so having both open at once would
  // overlap.
  const [openFlyout, setOpenFlyout] = useState<"data" | "compute" | null>(null);
  const dataFlyoutOpen = openFlyout === "data";
  const computeFlyoutOpen = openFlyout === "compute";
  // A folder added as a whole (see DataCatalogPanel's FolderRow) shows here
  // as one grouped entry instead of exploding into its individual files,
  // split into an OSM card and an "Other" card for everything else.
  const projectSections = useMemo(
    () => groupProjectDatasetsForDisplay(projectDatasets),
    [projectDatasets],
  );

  const handleBrowseDataCatalog = useCallback(() => {
    // Opens the Data Catalog sidebar without closing this flyout - they're
    // two independent panels, and closing this one on click read as the
    // flyout "folding and disappearing" rather than a deliberate action.
    onOpenDataCatalog();
  }, [onOpenDataCatalog]);

  const [runStatus, setRunStatus] = useState<
    "idle" | "running" | "success" | "failed"
  >("idle");
  const runStatusTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleRunDataflow = useCallback(async () => {
    if (runStatus === "running") return;
    if (runStatusTimeout.current) clearTimeout(runStatusTimeout.current);

    setRunStatus("running");
    let ok = false;
    try {
      const result = await runDataflow(getNodes(), getEdges());
      ok = result.ok;
    } catch {
      ok = false;
    }

    setRunStatus(ok ? "success" : "failed");
    runStatusTimeout.current = setTimeout(() => setRunStatus("idle"), 2000);
  }, [runStatus, getNodes, getEdges]);

  useEffect(() => {
    return () => {
      if (runStatusTimeout.current) clearTimeout(runStatusTimeout.current);
    };
  }, []);

  // New nodes always land at the viewport center, same as the old dropdown
  // menu - _desiredGrammarPos is the handoff createGrammarNode() reads.
  const getDropPosition = useCallback(() => {
    return screenToFlowPosition({
      x: window.innerWidth / 2,
      y: window.innerHeight / 2,
    });
  }, [screenToFlowPosition]);

  const handleAdd = useCallback(
    (tpl: TemplateKey) => {
      (window as any)._desiredGrammarPos = getDropPosition();
      onAdd(tpl);
    },
    [getDropPosition, onAdd],
  );

  const handleAddPyCodeEditor = useCallback(() => {
    (window as any)._desiredGrammarPos = getDropPosition();
    onAddPyCodeEditor();
  }, [getDropPosition, onAddPyCodeEditor]);

  // The canvas' onDrop computes the real drop position itself (from the drop
  // event's coordinates) - dragValue only needs to say *which* node to add.
  const handleDragStart = useCallback(
    (e: DragEvent<HTMLButtonElement>, dragValue: string) => {
      e.dataTransfer.setData(NODE_DRAG_MIME, dragValue);
      e.dataTransfer.effectAllowed = "move";
    },
    [],
  );

  const sections: RailSection[] = [
    {
      title: "Intelligence",
      accent: "#cb181d",
      hoverBg: "rgba(203, 24, 29, 0.1)",
      items: [
        {
          key: "data_layer",
          label: TEMPLATE_LABELS.data_layer,
          icon: <LayersOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("data_layer"),
          dragValue: "data_layer",
        },
        {
          key: "join",
          label: TEMPLATE_LABELS.join,
          icon: <MergeTypeOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("join"),
          dragValue: "join",
        },
        {
          key: "code",
          label: "Code",
          icon: <CodeOutlinedIcon sx={ICON_SX} />,
          onClick: handleAddPyCodeEditor,
          dragValue: PY_CODE_DRAG_VALUE,
        },
      ],
    },
    {
      title: "Design",
      accent: "#238b45",
      hoverBg: "rgba(35, 139, 69, 0.1)",
      items: [
        {
          key: "view",
          label: TEMPLATE_LABELS.view,
          icon: <MapOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("view"),
          dragValue: "view",
        },
      ],
    },
    {
      title: "Choice",
      accent: "#1f78b4",
      hoverBg: "rgba(31, 120, 180, 0.1)",
      items: [
        {
          key: "interaction",
          label: TEMPLATE_LABELS.interaction,
          icon: <TouchAppOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("interaction"),
          dragValue: "interaction",
        },
        {
          key: "widget",
          label: TEMPLATE_LABELS.widget,
          icon: <TuneOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("widget"),
          dragValue: "widget",
        },
        {
          key: "comparison",
          label: TEMPLATE_LABELS.comparison,
          icon: <BarChartOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("comparison"),
          dragValue: "comparison",
        },
      ],
    },
  ];

  return (
    <div className="node-rail-stack">
      <div className="node-rail">
        {sections.map((section) => (
          <div
            key={section.title}
            className="node-rail__section"
            style={
              {
                "--rail-accent": section.accent,
                "--rail-hover": section.hoverBg,
              } as CSSProperties
            }
          >
            <div className="node-rail__title">{section.title}</div>
            <div className="node-rail__grid">
              {section.items.map((item) => (
                <Tooltip key={item.key} title={item.label} placement="right" arrow>
                  <button
                    type="button"
                    className="node-rail__icon"
                    aria-label={item.label}
                    onClick={item.onClick}
                    draggable
                    onDragStart={(e) => handleDragStart(e, item.dragValue)}
                  >
                    {item.icon}
                  </button>
                </Tooltip>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="node-rail-catalog">
        <button
          type="button"
          className="node-rail-catalog__toggle"
          aria-haspopup="true"
          aria-expanded={dataFlyoutOpen}
          onClick={() => setOpenFlyout((v) => (v === "data" ? null : "data"))}
          title="Data Catalog"
        >
          <div className="node-rail-catalog__icon-row">
            <LayersOutlinedIcon sx={ICON_SX} style={{ color: "#cb181d" }} />
            <span className="node-rail-catalog__count">{projectDatasets.length}</span>
            {dataFlyoutOpen ? (
              <ChevronLeftIcon className="node-rail-catalog__chevron" sx={ICON_SX} />
            ) : (
              <ChevronRightIcon className="node-rail-catalog__chevron" sx={ICON_SX} />
            )}
          </div>
          <span className="node-rail-catalog__label">Data Catalog</span>
        </button>

        {dataFlyoutOpen && (
          <div className="node-rail-catalog__flyout">
            <button
              type="button"
              className="node-rail-catalog__flyout-close"
              aria-label="Close Data Catalog"
              onClick={() => setOpenFlyout(null)}
            >
              <CloseIcon sx={{ fontSize: 16 }} />
            </button>
            <p className="node-rail-catalog__flyout-title">
              Add or compute a dataset to use it here.
            </p>

            <div className="node-rail-catalog__project-section">
              <div className="node-rail-catalog__project-header">
                <span>Datasets in project</span>
                <span className="node-rail-catalog__project-count">
                  {projectDatasets.length}
                </span>
              </div>
              {projectDatasets.length === 0 ? (
                <div className="node-rail-catalog__project-empty">No datasets added yet.</div>
              ) : (
                <div className="node-rail-catalog__project-list">
                  {projectSections.map((section) => (
                    <div key={section.key} className="node-rail-catalog__project-source">
                      <div className="node-rail-catalog__project-source-title">
                        {section.label}
                      </div>
                      {section.kind === "flat"
                        ? section.files.map((f) => <ProjectFileItem key={f.id} dataset={f} />)
                        : [
                            ...[...section.root.folders.values()]
                              .sort((a, b) => a.name.localeCompare(b.name))
                              .map((folder) => (
                                <ProjectFolderItem key={folder.path} name={folder.name} node={folder} />
                              )),
                            ...[...section.root.files]
                              .sort((a, b) => a.name.localeCompare(b.name))
                              .map((f) => <ProjectFileItem key={f.id} dataset={f} />),
                          ]}
                    </div>
                  ))}
                </div>
              )}
            </div>

            <p className="node-rail-catalog__drag-hint">Drag a dataset onto the canvas to add it.</p>

            <button
              type="button"
              className="node-rail-catalog__browse-btn"
              onClick={handleBrowseDataCatalog}
            >
              Browse Data Catalog
            </button>
          </div>
        )}
      </div>

      <div className="node-rail-catalog">
        <button
          type="button"
          className="node-rail-catalog__toggle"
          aria-haspopup="true"
          aria-expanded={computeFlyoutOpen}
          onClick={() => setOpenFlyout((v) => (v === "compute" ? null : "compute"))}
          title="Compute Catalog"
        >
          <div className="node-rail-catalog__icon-row">
            <CodeOutlinedIcon sx={ICON_SX} style={{ color: "#cb181d" }} />
            <span className="node-rail-catalog__count">{projectCompute.length}</span>
            {computeFlyoutOpen ? (
              <ChevronLeftIcon className="node-rail-catalog__chevron" sx={ICON_SX} />
            ) : (
              <ChevronRightIcon className="node-rail-catalog__chevron" sx={ICON_SX} />
            )}
          </div>
          <span className="node-rail-catalog__label">Compute Catalog</span>
        </button>

        {computeFlyoutOpen && (
          <div className="node-rail-catalog__flyout">
            <button
              type="button"
              className="node-rail-catalog__flyout-close"
              aria-label="Close Compute Catalog"
              onClick={() => setOpenFlyout(null)}
            >
              <CloseIcon sx={{ fontSize: 16 }} />
            </button>
            <p className="node-rail-catalog__flyout-title">
              Add a model or transformation from the Compute Catalog to use it here.
            </p>

            <div className="node-rail-catalog__project-section">
              <div className="node-rail-catalog__project-header">
                <span>Models & transformations in project</span>
                <span className="node-rail-catalog__project-count">{projectCompute.length}</span>
              </div>
              {projectCompute.length === 0 ? (
                <div className="node-rail-catalog__project-empty">No models added yet.</div>
              ) : (
                <div className="node-rail-catalog__project-list">
                  {[...projectCompute]
                    .sort((a, b) => a.displayName.localeCompare(b.displayName))
                    .map((entry) => (
                      <ComputeProjectItem key={entry.id} entry={entry} />
                    ))}
                </div>
              )}
            </div>

            <p className="node-rail-catalog__drag-hint">Drag a model onto the canvas to add it.</p>

            <button
              type="button"
              className="node-rail-catalog__browse-btn"
              onClick={onOpenComputeCatalog}
            >
              Browse Compute Catalog
            </button>
          </div>
        )}
      </div>

      <Tooltip
        title={
          runStatus === "failed"
            ? "Run failed - see the failed node(s) for details"
            : "Run dataflow"
        }
        placement="right"
        arrow
      >
        <button
          type="button"
          className={`node-rail-play${
            runStatus === "failed" ? " node-rail-play--failed" : ""
          }`}
          aria-label="Run dataflow"
          onClick={() => void handleRunDataflow()}
          disabled={runStatus === "running"}
        >
          {runStatus === "running" ? (
            <CircularProgress size={20} color="inherit" />
          ) : runStatus === "success" ? (
            <CheckIcon sx={{ fontSize: 28 }} />
          ) : (
            <SkipNextIcon sx={{ fontSize: 36 }} />
          )}
        </button>
      </Tooltip>
    </div>
  );
}
