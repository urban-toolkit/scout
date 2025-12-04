// JsonCodeEditor.tsx
import { useEffect, useMemo, useRef, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { json, jsonLanguage } from "@codemirror/lang-json";
import { linter, Diagnostic, lintGutter } from "@codemirror/lint";
import { EditorView, keymap } from "@codemirror/view";
import {
  openSearchPanel,
  searchKeymap,
  search,
  highlightSelectionMatches,
} from "@codemirror/search";
import "./JsonCodeEditor.css";

type Props = {
  value?: unknown;
  onChange?: (val: unknown) => void;
  height?: number | string; // e.g., "420px" or 420
  readOnly?: boolean;
  onDiagnostics?: (diags: Diagnostic[]) => void; // <-- add this
};

export default function JsonCodeEditor({
  value = {},
  onChange,
  height = "420px",
  readOnly = false,
  onDiagnostics,
}: Props) {
  const [text, setText] = useState(() => JSON.stringify(value, null, 2));
  const lastApplied = useRef<string>(JSON.stringify(value));

  // keep editor text in sync with external value changes
  useEffect(() => {
    const incoming = JSON.stringify(value);
    if (incoming !== lastApplied.current) {
      lastApplied.current = incoming;
      setText(JSON.stringify(value, null, 2));
    }
  }, [value]);

  // Linter: mark JSON syntax errors at correct ranges + JSON.parse validation
  const jsonSyntaxLinter = useMemo(
    () =>
      linter((view): Diagnostic[] => {
        const doc = view.state.doc.toString();
        const tree = jsonLanguage.parser.parse(doc);
        const diags: Diagnostic[] = [];

        tree.iterate({
          enter(node) {
            if (node.type.isError) {
              diags.push({
                from: node.from,
                to: node.to,
                severity: "error",
                message: "JSON syntax error",
              });
            }
          },
        });

        if (diags.length === 0) {
          try {
            JSON.parse(doc);
          } catch (e: unknown) {
            const message = e instanceof Error ? e.message : "Invalid JSON"; // ✅ type-safe
            diags.push({
              from: 0,
              to: Math.min(1, doc.length),
              severity: "error",
              message,
            });
          }
        }
        onDiagnostics?.(diags);
        return diags;
      }),
    [onDiagnostics]
  );

  const handleChange = (next: string) => {
    setText(next);
    try {
      const parsed = JSON.parse(next);
      lastApplied.current = JSON.stringify(parsed);
      onChange?.(parsed);
    } catch {
      /* ignore until valid again */
    }
  };

  return (
    <div
      className="nodrag nowheel"
      style={{
        height: typeof height === "number" ? `${height}px` : height,
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        overflow: "hidden",
        background: "transparent",
        display: "grid",
        gridTemplateRows: "1fr auto",
      }}
    >
      {/* Editor */}
      <div style={{ overflow: "auto" }}>
        <CodeMirror
          value={text}
          onChange={handleChange}
          readOnly={readOnly}
          height="100%"
          width="100%"
          extensions={[
            json(),
            jsonSyntaxLinter,
            lintGutter(),
            EditorView.editable.of(!readOnly),
            search({ top: false }),
            highlightSelectionMatches(),
            keymap.of([
              {
                key: "Mod-f",
                run: (view) => {
                  view.requestMeasure();
                  return openSearchPanel(view);
                },
              },
              ...searchKeymap,
            ]),
          ]}
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
