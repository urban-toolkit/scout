// charts/ReadOnlyTag.tsx
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import "./ReadOnlyTag.css";

// Small "read-only" pill used next to a section label wherever a JSON box
// can't be edited (Chart Gallery's and Chart Studio's "Sample data") - kept
// as one shared component so both pages present it identically rather than
// drifting into two slightly different-looking badges.
export function ReadOnlyTag() {
  return (
    <span className="readonly-tag">
      <LockOutlinedIcon sx={{ fontSize: 11 }} />
      Read-only
    </span>
  );
}
