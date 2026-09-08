import { useEffect, useRef, useState } from "react";
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import "./DataflowNameLabel.css";

interface Props {
  name: string;
  onRename: (name: string) => void;
}

// Floats over the top-left of the canvas (a sibling of NodeRail inside
// <ReactFlow>) rather than living in the Toolbar - the name belongs to the
// dataflow on the canvas, not to app-wide navigation.
export default function DataflowNameLabel({ name, onRename }: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(name);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!editing) setDraft(name);
  }, [name, editing]);

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  const startEditing = () => {
    setDraft(name);
    setEditing(true);
  };

  const commit = () => {
    setEditing(false);
    const trimmed = draft.trim();
    if (trimmed && trimmed !== name) onRename(trimmed);
  };

  if (editing) {
    return (
      <input
        ref={inputRef}
        className="dataflow-name-label__input"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commit();
          } else if (e.key === "Escape") {
            e.preventDefault();
            setDraft(name);
            setEditing(false);
          }
        }}
      />
    );
  }

  return (
    <button
      type="button"
      className="dataflow-name-label"
      onClick={startEditing}
      title="Rename dataflow"
    >
      <span className="dataflow-name-label__text">{name}</span>
      <EditOutlinedIcon sx={{ fontSize: 15 }} />
    </button>
  );
}
