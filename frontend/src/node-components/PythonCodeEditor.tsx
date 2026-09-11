// PythonCodeEditor.tsx
import { useEffect, useMemo, useRef, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import { Decoration, EditorView } from "@codemirror/view";
import type { Extension } from "@codemirror/state";
import "./PythonCodeEditor.css";
import type { VariableHighlight } from "../utils/requiredVariables";

type Props = {
  value?: string;
  onChange?: (val: string) => void;
  height?: number | string; // e.g., "420px" or 420
  readOnly?: boolean;
  // Colors every occurrence of each named variable - see
  // RequiredVariablesBadge, which drives this from the same required/
  // resolved data the popover shows, so the code and the badge stay in
  // sync as one system rather than two places that can disagree.
  highlights?: VariableHighlight[];
};

const ESCAPE_RE = /[.*+?^${}()|[\]\\]/g;

function buildHighlightExtension(
  text: string,
  highlights: VariableHighlight[] | undefined,
): Extension[] {
  if (!highlights || highlights.length === 0) return [];

  const marks: { from: number; to: number; className: string }[] = [];
  for (const h of highlights) {
    if (!h.name) continue;
    const re = new RegExp(`\\b${h.name.replace(ESCAPE_RE, "\\$&")}\\b`, "g");
    const className = h.status === "resolved" ? "cm-var-resolved" : "cm-var-unmet";
    let m: RegExpExecArray | null;
    while ((m = re.exec(text))) {
      marks.push({ from: m.index, to: m.index + h.name.length, className });
    }
  }
  if (marks.length === 0) return [];

  marks.sort((a, b) => a.from - b.from || a.to - b.to);
  const ranges = marks.map((m) =>
    Decoration.mark({ class: m.className }).range(m.from, m.to),
  );
  return [EditorView.decorations.of(Decoration.set(ranges, true))];
}

export default function PythonCodeEditor({
  value = "",
  onChange,
  height = "200px",
  readOnly = false,
  highlights,
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

  const highlightExtensions = useMemo(
    () => buildHighlightExtension(text, highlights),
    [text, highlights],
  );

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
          extensions={[python(), EditorView.editable.of(!readOnly), ...highlightExtensions]}
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
