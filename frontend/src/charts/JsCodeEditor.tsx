// JsCodeEditor.tsx
import { useEffect, useMemo, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { javascript, javascriptLanguage } from "@codemirror/lang-javascript";
import { linter, Diagnostic, lintGutter } from "@codemirror/lint";
import { EditorView, keymap } from "@codemirror/view";
import {
  openSearchPanel,
  searchKeymap,
  search,
  highlightSelectionMatches,
} from "@codemirror/search";

type Props = {
  value?: string;
  onChange?: (val: string) => void;
  height?: number | string; // e.g., "420px" or 420
  readOnly?: boolean;
  onDiagnostics?: (diags: Diagnostic[]) => void;
};

export default function JsCodeEditor({
  value = "",
  onChange,
  height = "260px",
  readOnly = false,
  onDiagnostics,
}: Props) {
  const [text, setText] = useState(value);

  // keep editor text in sync with external value changes
  useEffect(() => {
    setText((prev) => (prev === value ? prev : value));
  }, [value]);

  // Linter: mark JS syntax errors at their parse-tree ranges
  const jsSyntaxLinter = useMemo(
    () =>
      linter((view): Diagnostic[] => {
        const doc = view.state.doc.toString();
        const tree = javascriptLanguage.parser.parse(doc);
        const diags: Diagnostic[] = [];

        tree.iterate({
          enter(node) {
            if (node.type.isError) {
              diags.push({
                from: node.from,
                to: node.to,
                severity: "error",
                message: "JavaScript syntax error",
              });
            }
          },
        });

        onDiagnostics?.(diags);
        return diags;
      }),
    [onDiagnostics],
  );

  const handleChange = (next: string) => {
    setText(next);
    onChange?.(next);
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
        textAlign: "left",
      }}
    >
      <div style={{ overflow: "auto" }}>
        <CodeMirror
          value={text}
          onChange={handleChange}
          readOnly={readOnly}
          height="100%"
          width="100%"
          extensions={[
            javascript(),
            jsSyntaxLinter,
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
            autocompletion: true,
            highlightActiveLine: true,
            highlightActiveLineGutter: true,
          }}
        />
      </div>
    </div>
  );
}
