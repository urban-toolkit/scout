import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import Paper from "@mui/material/Paper";
import Slide from "@mui/material/Slide";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import CodeOutlinedIcon from "@mui/icons-material/CodeOutlined";
import CloseIcon from "@mui/icons-material/Close";
import UploadFileOutlinedIcon from "@mui/icons-material/UploadFileOutlined";
import RefreshOutlinedIcon from "@mui/icons-material/RefreshOutlined";
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import Tooltip from "@mui/material/Tooltip";

import {
  listComputeCatalog,
  uploadComputeItem,
  refreshComputeItem,
  renameComputeItem,
  type ComputeCatalogEntry,
  type ComputeUploadSource,
} from "../utils/computeCatalog";
import type { ProjectComputeItem } from "../utils/dataflows";

const PANEL_WIDTH = 420;

type CatalogTab = "all" | "project";

function callableSummary(entry: ComputeCatalogEntry): string {
  if (entry.callables.length === 0) return "No callable found";
  const fns = entry.callables.filter((c) => c.kind === "function").length;
  const classes = entry.callables.filter((c) => c.kind === "class").length;
  const parts: string[] = [];
  if (fns > 0) parts.push(`${fns} function${fns > 1 ? "s" : ""}`);
  if (classes > 0) parts.push(`${classes} class${classes > 1 ? "es" : ""}`);
  return parts.join(", ");
}

function ComputeItemRow({
  entry,
  inProject,
  onAdd,
  onRemove,
  onRefresh,
  refreshing,
  onRename,
}: {
  entry: ComputeCatalogEntry;
  inProject: boolean;
  onAdd: (entry: ComputeCatalogEntry) => void;
  onRemove: (id: string) => void;
  onRefresh: (entry: ComputeCatalogEntry) => void;
  refreshing: boolean;
  onRename: (entry: ComputeCatalogEntry, displayName: string) => void;
}) {
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(entry.displayName);
  const nameInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!renaming) setDraft(entry.displayName);
  }, [entry.displayName, renaming]);

  useEffect(() => {
    if (renaming) {
      nameInputRef.current?.focus();
      nameInputRef.current?.select();
    }
  }, [renaming]);

  const commitRename = () => {
    setRenaming(false);
    const trimmed = draft.trim();
    if (trimmed && trimmed !== entry.displayName) onRename(entry, trimmed);
  };

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 1,
        py: 0.75,
        px: 1,
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
          bgcolor: "#fee2e2",
        }}
      >
        <CodeOutlinedIcon sx={{ fontSize: 18, color: "#cb181d" }} />
      </Box>
      <Box sx={{ minWidth: 0, flex: 1 }}>
        {renaming ? (
          <TextField
            inputRef={nameInputRef}
            variant="standard"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                commitRename();
              } else if (e.key === "Escape") {
                e.preventDefault();
                setDraft(entry.displayName);
                setRenaming(false);
              }
            }}
            onClick={(e) => e.stopPropagation()}
            fullWidth
            sx={{ "& .MuiInputBase-input": { fontSize: 13, fontWeight: 500, py: 0 } }}
          />
        ) : (
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
            {entry.displayName}
          </Typography>
        )}
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
          {entry.kind === "package" ? "Package" : "Script"} · {callableSummary(entry)}
        </Typography>
      </Box>
      {!renaming && (
        <Tooltip title="Rename">
          <IconButton
            size="small"
            onClick={() => setRenaming(true)}
            aria-label={`Rename ${entry.displayName}`}
          >
            <EditOutlinedIcon sx={{ fontSize: 15, color: "#94a3b8" }} />
          </IconButton>
        </Tooltip>
      )}
      <Tooltip title="Re-scan this file on disk for changes">
        <span>
          <IconButton
            size="small"
            onClick={() => onRefresh(entry)}
            disabled={refreshing}
            aria-label={`Refresh ${entry.displayName}`}
          >
            {refreshing ? (
              <CircularProgress size={14} />
            ) : (
              <RefreshOutlinedIcon sx={{ fontSize: 16, color: "#94a3b8" }} />
            )}
          </IconButton>
        </span>
      </Tooltip>
      <Button
        variant="text"
        size="small"
        onClick={() => (inProject ? onRemove(entry.id) : onAdd(entry))}
        sx={{
          textTransform: "none",
          fontWeight: 600,
          fontSize: 11,
          minWidth: 0,
          width: 136,
          color: "#94a3b8",
          "&:hover": { color: "#334155", bgcolor: "transparent" },
        }}
      >
        {inProject ? "Remove from project" : "Add to project"}
      </Button>
    </Box>
  );
}

interface Props {
  open: boolean;
  onClose: () => void;
  projectCompute: ProjectComputeItem[];
  onAddToProject: (entry: ComputeCatalogEntry) => void;
  onRemoveFromProject: (id: string) => void;
  // Called after a successful refresh so the dataflow's own project-level
  // copy of this entry (a separate snapshot taken when it was added) is
  // kept in sync too, not just this panel's "Browse all" list.
  onRefreshProjectItem: (entry: ComputeCatalogEntry) => void;
}

export default function ComputeCatalogPanel({
  open,
  onClose,
  projectCompute,
  onAddToProject,
  onRemoveFromProject,
  onRefreshProjectItem,
}: Props) {
  const [tab, setTab] = useState<CatalogTab>("all");
  const [items, setItems] = useState<ComputeCatalogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshingId, setRefreshingId] = useState<string | null>(null);

  const [pendingSource, setPendingSource] = useState<ComputeUploadSource | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [uploadDescription, setUploadDescription] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const zipInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const projectIds = useMemo(() => new Set(projectCompute.map((c) => c.id)), [projectCompute]);

  const refresh = () => {
    setLoading(true);
    setError(null);
    listComputeCatalog()
      .then(setItems)
      .catch((e) => setError(e.message || "Failed to load compute catalog."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (!open) return;
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const handleRenameItem = async (entry: ComputeCatalogEntry, displayName: string) => {
    try {
      const updated = await renameComputeItem(entry.id, displayName);
      setItems((prev) => prev.map((i) => (i.id === updated.id ? updated : i)));
      onRefreshProjectItem(updated);
    } catch (e: any) {
      setError(e.message || `Failed to rename '${entry.displayName}'.`);
    }
  };

  const handleRefreshItem = async (entry: ComputeCatalogEntry) => {
    setRefreshingId(entry.id);
    try {
      const updated = await refreshComputeItem(entry.id);
      setItems((prev) => prev.map((i) => (i.id === updated.id ? updated : i)));
      onRefreshProjectItem(updated);
    } catch (e: any) {
      setError(e.message || `Failed to refresh '${entry.displayName}'.`);
    } finally {
      setRefreshingId(null);
    }
  };

  const handleFilePicked = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setPendingSource({ mode: "file", file });
    setUploadName(file.name.replace(/\.py$/i, ""));
  };

  const handleZipPicked = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setPendingSource({ mode: "zip", file });
    setUploadName(file.name.replace(/\.zip$/i, ""));
  };

  const handleFolderPicked = (e: ChangeEvent<HTMLInputElement>) => {
    const fileList = Array.from(e.target.files ?? []);
    e.target.value = "";
    if (fileList.length === 0) return;
    const relpaths = fileList.map((f) => (f as any).webkitRelativePath || f.name);
    // Strip the folder's own top-level name from every relpath so files land
    // directly under compute/<slug>/ rather than compute/<slug>/<folderName>/.
    const firstSegment = relpaths[0].split("/")[0];
    const stripped = relpaths.map((p) =>
      p.startsWith(`${firstSegment}/`) ? p.slice(firstSegment.length + 1) : p,
    );
    setPendingSource({ mode: "folder", files: fileList, relpaths: stripped });
    setUploadName(firstSegment);
  };

  const handleUploadSubmit = async () => {
    if (!pendingSource || !uploadName.trim()) return;
    setUploading(true);
    setUploadError(null);
    try {
      const created = await uploadComputeItem(uploadName.trim(), uploadDescription, pendingSource);
      setItems((prev) => [...prev.filter((i) => i.id !== created.id), created]);
      setPendingSource(null);
      setUploadName("");
      setUploadDescription("");
    } catch (e: any) {
      setUploadError(e.message || "Failed to upload model.");
    } finally {
      setUploading(false);
    }
  };

  const tabs: { key: CatalogTab; label: string }[] = [
    { key: "all", label: "Browse all" },
    { key: "project", label: "In project" },
  ];

  return (
    <Slide direction="left" in={open} mountOnEnter unmountOnExit>
      <Paper
        elevation={8}
        square
        className="compute-catalog-panel"
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
            <CodeOutlinedIcon fontSize="small" sx={{ color: "#cb181d" }} />
            <Typography sx={{ fontWeight: 600, fontSize: 15, color: "#0f172a" }}>
              Compute Catalog
            </Typography>
          </Box>
          <IconButton size="small" onClick={onClose} aria-label="Close compute catalog">
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>

        <Box sx={{ px: 2, pt: 1.5, flexShrink: 0 }}>
          <Typography variant="body2" sx={{ color: "#64748b" }}>
            Models and transformations uploaded to this project.
          </Typography>
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
              {t.key === "all" && items.length > 0 ? ` (${items.length})` : ""}
              {t.key === "project" && projectCompute.length > 0 ? ` (${projectCompute.length})` : ""}
            </Box>
          ))}
        </Box>

        <Box sx={{ flex: 1, overflowY: "auto", p: 2 }}>
          {tab === "project" ? (
            projectCompute.length === 0 ? (
              <Typography variant="body2" sx={{ color: "#94a3b8", textAlign: "center", pt: 4 }}>
                No models added to this project yet.
              </Typography>
            ) : (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                {[...projectCompute]
                  .sort((a, b) => a.displayName.localeCompare(b.displayName))
                  .map((entry) => (
                    <ComputeItemRow
                      key={entry.id}
                      entry={entry}
                      inProject
                      onAdd={onAddToProject}
                      onRemove={onRemoveFromProject}
                      onRefresh={handleRefreshItem}
                      refreshing={refreshingId === entry.id}
                      onRename={handleRenameItem}
                    />
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
          ) : items.length === 0 ? (
            <Typography variant="body2" sx={{ color: "#94a3b8", textAlign: "center", pt: 4 }}>
              Nothing uploaded yet - use "Upload model" below.
            </Typography>
          ) : (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              {[...items]
                .sort((a, b) => a.displayName.localeCompare(b.displayName))
                .map((entry) => (
                  <ComputeItemRow
                    key={entry.id}
                    entry={entry}
                    inProject={projectIds.has(entry.id)}
                    onAdd={onAddToProject}
                    onRemove={onRemoveFromProject}
                    onRefresh={handleRefreshItem}
                    refreshing={refreshingId === entry.id}
                    onRename={handleRenameItem}
                  />
                ))}
            </Box>
          )}
        </Box>

        <Box sx={{ p: 2, borderTop: "1px solid #e5e7eb", flexShrink: 0 }}>
          {pendingSource ? (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <Typography variant="caption" sx={{ color: "#64748b" }}>
                {pendingSource.mode === "folder"
                  ? `${pendingSource.files.length} file(s) selected`
                  : pendingSource.file.name}
              </Typography>
              <TextField
                size="small"
                label="Name"
                value={uploadName}
                onChange={(e) => setUploadName(e.target.value)}
                fullWidth
              />
              <TextField
                size="small"
                label="Description (optional)"
                value={uploadDescription}
                onChange={(e) => setUploadDescription(e.target.value)}
                fullWidth
              />
              {uploadError && (
                <Typography variant="caption" color="error">
                  {uploadError}
                </Typography>
              )}
              <Box sx={{ display: "flex", gap: 1 }}>
                <Button
                  fullWidth
                  variant="outlined"
                  disabled={uploading}
                  onClick={() => {
                    setPendingSource(null);
                    setUploadError(null);
                  }}
                >
                  Cancel
                </Button>
                <Button
                  fullWidth
                  variant="contained"
                  disabled={uploading || !uploadName.trim()}
                  onClick={handleUploadSubmit}
                >
                  {uploading ? "Uploading…" : "Upload"}
                </Button>
              </Box>
            </Box>
          ) : (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <input
                ref={fileInputRef}
                type="file"
                accept=".py"
                hidden
                onChange={handleFilePicked}
              />
              <input ref={zipInputRef} type="file" accept=".zip" hidden onChange={handleZipPicked} />
              <input
                ref={folderInputRef}
                type="file"
                hidden
                // @ts-expect-error - non-standard but broadly supported attrs for folder selection
                webkitdirectory=""
                directory=""
                multiple
                onChange={handleFolderPicked}
              />
              <Button
                fullWidth
                variant="outlined"
                startIcon={<UploadFileOutlinedIcon />}
                sx={{ textTransform: "none", fontWeight: 600 }}
                onClick={() => fileInputRef.current?.click()}
              >
                Upload .py file
              </Button>
              <Box sx={{ display: "flex", gap: 1 }}>
                <Button
                  fullWidth
                  variant="outlined"
                  sx={{ textTransform: "none", fontWeight: 600 }}
                  onClick={() => zipInputRef.current?.click()}
                >
                  Upload .zip
                </Button>
                <Button
                  fullWidth
                  variant="outlined"
                  sx={{ textTransform: "none", fontWeight: 600 }}
                  onClick={() => folderInputRef.current?.click()}
                >
                  Upload folder
                </Button>
              </Box>
            </Box>
          )}
        </Box>
      </Paper>
    </Slide>
  );
}
