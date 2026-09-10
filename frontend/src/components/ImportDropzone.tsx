import { useRef, useState, type DragEvent } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CloudUploadOutlinedIcon from "@mui/icons-material/CloudUploadOutlined";
import {
  resolveDroppedItems,
  resolvePickedFiles,
  resolvePickedFolder,
  type ResolvedDrop,
} from "../utils/dragDropFiles";

// One shared import entry point for both catalogs - a file, several files, a
// .zip, or a whole folder can all be dropped here (or picked via the two
// fallback links), and dragDropFiles.ts figures out which upload mode each
// resolves to, so this component itself doesn't need to know about
// DataUploadSource vs ComputeUploadSource.
export default function ImportDropzone({
  acceptHint,
  onResolved,
}: {
  acceptHint: string;
  onResolved: (drop: ResolvedDrop) => void;
}) {
  const [dragActive, setDragActive] = useState(false);
  // dragenter/dragleave fire once per descendant element crossed, not once
  // per zone - a depth counter is what keeps the highlight from flickering
  // off while the pointer is still over a child element.
  const dragDepth = useRef(0);
  const filesInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const linkSx = {
    border: "none",
    background: "none",
    p: 0,
    cursor: "pointer",
    color: "#cb181d",
    fontWeight: 600,
    fontSize: 11.5,
    fontFamily: "inherit",
  };

  return (
    <Box
      onDragEnter={(e: DragEvent) => {
        e.preventDefault();
        dragDepth.current += 1;
        setDragActive(true);
      }}
      onDragOver={(e: DragEvent) => e.preventDefault()}
      onDragLeave={(e: DragEvent) => {
        e.preventDefault();
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (dragDepth.current === 0) setDragActive(false);
      }}
      onDrop={async (e: DragEvent) => {
        e.preventDefault();
        dragDepth.current = 0;
        setDragActive(false);
        const resolved = await resolveDroppedItems(e.dataTransfer);
        if (resolved) onResolved(resolved);
      }}
      sx={{
        border: "1.5px dashed",
        borderColor: dragActive ? "#cb181d" : "#cbd5e1",
        borderRadius: 2,
        bgcolor: dragActive ? "rgba(203, 24, 29, 0.06)" : "#f8fafc",
        px: 2,
        py: 2.5,
        textAlign: "center",
        transition: "border-color 0.15s ease, background-color 0.15s ease",
      }}
    >
      <CloudUploadOutlinedIcon
        sx={{ fontSize: 26, color: dragActive ? "#cb181d" : "#94a3b8", mb: 0.5 }}
      />
      <Typography sx={{ fontSize: 13, fontWeight: 600, color: dragActive ? "#cb181d" : "#0f172a" }}>
        {dragActive ? "Drop to upload" : "Drag files or a folder here"}
      </Typography>
      <Typography sx={{ fontSize: 11.5, color: "#64748b", mb: 1.5 }}>{acceptHint}</Typography>
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 0.75 }}>
        <Box component="button" onClick={() => filesInputRef.current?.click()} sx={linkSx}>
          Browse files
        </Box>
        <Typography component="span" sx={{ color: "#94a3b8", fontSize: 11.5 }}>
          ·
        </Typography>
        <Box component="button" onClick={() => folderInputRef.current?.click()} sx={linkSx}>
          Browse a folder
        </Box>
      </Box>

      <input
        ref={filesInputRef}
        type="file"
        multiple
        hidden
        onChange={(e) => {
          const resolved = e.target.files ? resolvePickedFiles(e.target.files) : null;
          e.target.value = "";
          if (resolved) onResolved(resolved);
        }}
      />
      <input
        ref={folderInputRef}
        type="file"
        hidden
        // @ts-expect-error - non-standard but broadly supported attrs for folder selection
        webkitdirectory=""
        directory=""
        multiple
        onChange={(e) => {
          const resolved = e.target.files ? resolvePickedFolder(e.target.files) : null;
          e.target.value = "";
          if (resolved) onResolved(resolved);
        }}
      />
    </Box>
  );
}
