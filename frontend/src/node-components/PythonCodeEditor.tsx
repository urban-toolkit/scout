// PythonCodeEditor.tsx
import { useEffect, useRef, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import { EditorView } from "@codemirror/view";
import "./PythonCodeEditor.css";

type Props = {
  value?: string;
  onChange?: (val: string) => void;
  height?: number | string; // e.g., "420px" or 420
  readOnly?: boolean;
};

export default function PythonCodeEditor({
  value = "",
  onChange,
  height = "200px",
  readOnly = false,
}: Props) {
  const [text, setText] = useState<string>(value);
  const lastApplied = useRef<string>(value);

  // keep editor text in sync with external value changes
  useEffect(() => {
    const incoming = value ?? "";
    if (incoming !== lastApplied.current) {
      lastApplied.current = incoming;
      setText(incoming);
    }
  }, [value]);

  const handleChange = (next: string) => {
    setText(next);
    lastApplied.current = next;
    onChange?.(next);
  };

  return (
    <div
      className="nodrag nowheel pyeditor-container"
      style={{
        height: typeof height === "number" ? `${height}px` : height,
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        overflow: "auto",
        background: "transparent",
        display: "grid",
        gridTemplateRows: "1fr",
      }}
    >
      <div>
        <CodeMirror
          value={text}
          onChange={handleChange}
          readOnly={readOnly}
          height="100%"
          width="100%"
          extensions={[python(), EditorView.editable.of(!readOnly)]}
          basicSetup={{
            lineNumbers: true,
            foldGutter: true,
            bracketMatching: true,
            autocompletion: false,
            highlightActiveLine: true,
            highlightActiveLineGutter: true,
          }}
        />
      </div>
    </div>
  );
}
