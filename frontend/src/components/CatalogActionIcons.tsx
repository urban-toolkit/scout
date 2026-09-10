import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import FileDownloadOutlinedIcon from "@mui/icons-material/FileDownloadOutlined";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import AddOutlinedIcon from "@mui/icons-material/AddOutlined";

// A small download affordance shared by every file/folder row in both the
// Data Catalog and Compute Catalog panels - each panel resolves its own
// download URL (dataCatalogDownloadUrl / computeCatalogDownloadUrl) and
// hands it in as `href`, so this component itself stays catalog-agnostic.
export function DownloadIconButton({ href, name }: { href: string; name: string }) {
  return (
    <Tooltip title="Download">
      <IconButton
        size="small"
        component="a"
        href={href}
        download
        onClick={(e) => e.stopPropagation()}
        aria-label={`Download ${name}`}
        sx={{ p: 0.25 }}
      >
        <FileDownloadOutlinedIcon sx={{ fontSize: 15, color: "#94a3b8" }} />
      </IconButton>
    </Tooltip>
  );
}

// The other half of every row's action pair, next to Download - a plain "+"
// when not yet in the project, flipping to a trash icon once it is (both
// tooltipped rather than labeled, since the icon alone reads ambiguously).
// Shared by Data Catalog and Compute Catalog rows alike.
export function AddRemoveIconButton({
  inProject,
  onAdd,
  onRemove,
  label,
}: {
  inProject: boolean;
  onAdd: () => void;
  onRemove: () => void;
  label: string;
}) {
  return (
    <Tooltip title={inProject ? "Remove from project" : "Add to project"}>
      <IconButton
        size="small"
        onClick={(e) => {
          e.stopPropagation();
          inProject ? onRemove() : onAdd();
        }}
        aria-label={`${inProject ? "Remove" : "Add"} ${label}`}
        sx={{ p: 0.25 }}
      >
        {inProject ? (
          <DeleteOutlineIcon sx={{ fontSize: 15, color: "#94a3b8" }} />
        ) : (
          <AddOutlinedIcon sx={{ fontSize: 15, color: "#94a3b8" }} />
        )}
      </IconButton>
    </Tooltip>
  );
}
