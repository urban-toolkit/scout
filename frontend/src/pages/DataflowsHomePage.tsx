import { useEffect, useMemo, useState } from "react";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogActions from "@mui/material/DialogActions";
import Button from "@mui/material/Button";
import DeleteOutlineOutlinedIcon from "@mui/icons-material/DeleteOutlineOutlined";
import {
  createDataflow,
  deleteDataflow,
  listDataflows,
  type DataflowPreviewEdge,
  type DataflowPreviewNode,
  type DataflowSummary,
} from "../utils/dataflows";
import "./DataflowsHomePage.css";

function formatRelativeTime(iso: string | null): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const seconds = Math.max(0, Math.floor((Date.now() - then) / 1000));
  const units: [string, number][] = [
    ["y", 31536000],
    ["mo", 2592000],
    ["d", 86400],
    ["h", 3600],
    ["m", 60],
  ];
  for (const [label, secondsPer] of units) {
    const value = Math.floor(seconds / secondsPer);
    if (value >= 1) return `${value}${label} ago`;
  }
  return "just now";
}

// Mirrors the accent colors NodeRail assigns to each node category, so a
// thumbnail's node colors read as part of the same system rather than a
// new one-off palette.
const NODE_TYPE_COLORS: Record<string, string> = {
  dataLayerNode: "#cb181d",
  joinNode: "#cb181d",
  pyCodeEditorNode: "#cb181d",
  viewNode: "#238b45",
  interactionNode: "#1f78b4",
  widgetNode: "#1f78b4",
  comparisonNode: "#1f78b4",
};
const DEFAULT_NODE_COLOR = "#94a3b8";
const DEFAULT_NODE_WIDTH = 180;
const DEFAULT_NODE_HEIGHT = 60;
const THUMB_WIDTH = 200;
const THUMB_HEIGHT = 96;

// A tiny schematic of the dataflow's own canvas layout - not a pixel
// screenshot, just each node's position/type and each edge, rescaled to fit
// the thumbnail. Cheap to compute since the list API already sends this
// (see backend/server.py's _dataflow_summary), not the full node payloads.
function DataflowThumbnail({
  nodes,
  edges,
}: {
  nodes: DataflowPreviewNode[];
  edges: DataflowPreviewEdge[];
}) {
  if (nodes.length === 0) {
    return <div className="dataflow-card__thumb dataflow-card__thumb--empty">Empty canvas</div>;
  }

  const centers = new Map<string, { x: number; y: number }>();
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const n of nodes) {
    const pos = n.position ?? { x: 0, y: 0 };
    const w = n.width ?? DEFAULT_NODE_WIDTH;
    const h = n.height ?? DEFAULT_NODE_HEIGHT;
    centers.set(n.id, { x: pos.x + w / 2, y: pos.y + h / 2 });
    minX = Math.min(minX, pos.x);
    minY = Math.min(minY, pos.y);
    maxX = Math.max(maxX, pos.x + w);
    maxY = Math.max(maxY, pos.y + h);
  }

  const bboxW = Math.max(maxX - minX, 1);
  const bboxH = Math.max(maxY - minY, 1);
  const padding = 12;
  const scale = Math.min(
    (THUMB_WIDTH - padding * 2) / bboxW,
    (THUMB_HEIGHT - padding * 2) / bboxH,
  );
  const offsetX = (THUMB_WIDTH - bboxW * scale) / 2;
  const offsetY = (THUMB_HEIGHT - bboxH * scale) / 2;
  const toSvg = (x: number, y: number) => ({
    x: offsetX + (x - minX) * scale,
    y: offsetY + (y - minY) * scale,
  });

  return (
    <svg
      className="dataflow-card__thumb"
      viewBox={`0 0 ${THUMB_WIDTH} ${THUMB_HEIGHT}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {edges.map((e, i) => {
        if (!e.source || !e.target) return null;
        const a = centers.get(e.source);
        const b = centers.get(e.target);
        if (!a || !b) return null;
        const pa = toSvg(a.x, a.y);
        const pb = toSvg(b.x, b.y);
        return <line key={i} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y} stroke="#cbd5e1" strokeWidth={1.5} />;
      })}
      {nodes.map((n) => {
        const pos = n.position ?? { x: 0, y: 0 };
        const w = n.width ?? DEFAULT_NODE_WIDTH;
        const h = n.height ?? DEFAULT_NODE_HEIGHT;
        const topLeft = toSvg(pos.x, pos.y);
        const color = (n.type && NODE_TYPE_COLORS[n.type]) || DEFAULT_NODE_COLOR;
        return (
          <rect
            key={n.id}
            x={topLeft.x}
            y={topLeft.y}
            width={Math.max(w * scale, 3)}
            height={Math.max(h * scale, 3)}
            rx={2}
            fill="#fff"
            stroke={color}
            strokeWidth={1.5}
          />
        );
      })}
    </svg>
  );
}

export default function DataflowsHomePage({
  onOpenDataflow,
}: {
  onOpenDataflow: (id: string) => void;
}) {
  const [dataflows, setDataflows] = useState<DataflowSummary[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [creating, setCreating] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<DataflowSummary | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listDataflows()
      .then((list) => {
        if (!cancelled) setDataflows(list);
      })
      .catch((e) => {
        if (!cancelled) setLoadError(e.message || "Failed to load dataflows.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const visibleDataflows = useMemo(() => {
    const query = search.trim().toLowerCase();
    const list = dataflows ?? [];
    if (!query) return list;
    return list.filter((d) => d.name.toLowerCase().includes(query));
  }, [dataflows, search]);

  const handleCreate = async () => {
    if (creating) return;
    setCreating(true);
    try {
      const record = await createDataflow();
      onOpenDataflow(record.id);
    } catch (e: any) {
      setLoadError(e.message || "Failed to create a new dataflow.");
      setCreating(false);
    }
  };

  const closeDeleteDialog = () => {
    if (deleting) return;
    setPendingDelete(null);
    setDeleteError(null);
  };

  const handleConfirmDelete = async () => {
    if (!pendingDelete || deleting) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteDataflow(pendingDelete.id);
      setDataflows((prev) => (prev ?? []).filter((d) => d.id !== pendingDelete.id));
      setPendingDelete(null);
    } catch (e: any) {
      setDeleteError(e.message || "Failed to delete this dataflow.");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="dataflows-page">
      <div className="dataflows-page__header">
        <div>
          <h1>Dataflows</h1>
          <p>Your dataflows. Open one to keep working on it, or start a new one.</p>
        </div>
        <button
          type="button"
          className="dataflows-page__new-btn"
          onClick={handleCreate}
          disabled={creating}
        >
          {creating ? "Creating…" : "+ New Dataflow"}
        </button>
      </div>

      <input
        type="text"
        className="dataflows-page__search"
        placeholder="Search dataflows…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      {loadError && <div className="dataflows-page__error">{loadError}</div>}

      {dataflows === null && !loadError ? (
        <div className="dataflows-page__loading">Loading dataflows…</div>
      ) : dataflows !== null && dataflows.length === 0 ? (
        <div className="dataflows-page__empty">
          No dataflows yet. Create your first one to get started.
        </div>
      ) : dataflows !== null && visibleDataflows.length === 0 ? (
        <div className="dataflows-page__empty">No dataflows match your search.</div>
      ) : (
        <div className="dataflows-page__grid">
          {visibleDataflows.map((d) => (
            <div
              key={d.id}
              className="dataflow-card"
              role="button"
              tabIndex={0}
              onClick={() => onOpenDataflow(d.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") onOpenDataflow(d.id);
              }}
            >
              <button
                type="button"
                className="dataflow-card__delete"
                aria-label={`Delete ${d.name}`}
                onClick={(e) => {
                  e.stopPropagation();
                  setDeleteError(null);
                  setPendingDelete(d);
                }}
              >
                <DeleteOutlineOutlinedIcon sx={{ fontSize: 17 }} />
              </button>
              <DataflowThumbnail nodes={d.nodesPreview} edges={d.edgesPreview} />
              <div className="dataflow-card__name">{d.name}</div>
              <div className="dataflow-card__meta">
                {d.nodeCount} node{d.nodeCount === 1 ? "" : "s"}
                {d.updatedAt ? ` · Updated ${formatRelativeTime(d.updatedAt)}` : ""}
              </div>
            </div>
          ))}
        </div>
      )}

      <Dialog open={pendingDelete !== null} onClose={closeDeleteDialog}>
        <DialogTitle sx={{ fontWeight: 700 }}>Delete "{pendingDelete?.name}"?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This will permanently delete this dataflow and everything on its canvas. This
            action can't be undone.
          </DialogContentText>
          {deleteError && (
            <DialogContentText sx={{ color: "#b91c1c", mt: 1.5 }}>
              {deleteError}
            </DialogContentText>
          )}
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={closeDeleteDialog} disabled={deleting} sx={{ textTransform: "none" }}>
            Cancel
          </Button>
          <Button
            onClick={handleConfirmDelete}
            disabled={deleting}
            color="error"
            variant="contained"
            disableElevation
            sx={{ textTransform: "none", fontWeight: 600 }}
          >
            {deleting ? "Deleting…" : "Delete"}
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}
