import { memo, useCallback, useState, type ChangeEvent } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import { Position, NodeResizer, useReactFlow, Handle } from "@xyflow/react";
import "./PyCodeEditorNode.css";
import restartPng from "../../assets/restart.png";
import runPng from "../../assets/run.png";
import checkPng from "../../assets/check-mark.png";
import expandPng from "../../assets/expand.png";
import { WidgetOutput } from "../../utils/types";
import PythonCodeEditor from "../../node-components/PythonCodeEditor";

export type PyCodeEditorNodeData = {
  title?: string;
  code?: string; // <-- added here
  onClose?: (id: string) => void;
  onRun?: (srcId: string, code: string) => void;
  widgetOutputs?: WidgetOutput[];
};

export type PyCodeEditorNode = Node<PyCodeEditorNodeData, "pyCodeEditorNode">;

const NODE_MIN_WIDTH = 300;
const NODE_MIN_HEIGHT = 260;

const NODE_MINIMIZED_WIDTH = 150; // tweak as you like
const NODE_MINIMIZED_HEIGHT = 48; // matches minimized bar height

const PyCodeEditorNode = memo(function PyCodeEditorNode({
  id,
  data,
}: NodeProps<PyCodeEditorNode>) {
  const rf = useReactFlow();

  const [running, setRunning] = useState(false);
  const [runningSuccess, setRunningSuccess] = useState(false);
  const [minimized, setMinimized] = useState(false);

  // NEW: store output panel data
  const [output, setOutput] = useState<{ stdout: string; stderr: string }>({
    stdout: "",
    stderr: "",
  });

  // ---------- TITLE CHANGE ----------
  const handleTitleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const nextTitle = e.target.value;
      rf.setNodes((nodes) =>
        nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, title: nextTitle } } : n
        )
      );
    },
    [id, rf]
  );

  // ---------- CLOSE ACTION ----------
  const handleClose = useCallback(() => {
    console.log(data);
    if (data?.onClose) return data.onClose(id);
    rf.setNodes((nds) => nds.filter((n) => n.id !== id));
  }, [data, id, rf]);

  // ---------- MINIMIZE TOGGLE ----------
  const handleToggleMinimize = useCallback(() => {
    setMinimized((prev) => {
      const next = !prev;

      rf.setNodes((nodes) =>
        nodes.map((n) => {
          if (n.id !== id) return n;

          if (next) {
            // Going TO minimized: snap down to a small default size
            return {
              ...n,
              width: NODE_MINIMIZED_WIDTH,
              height: NODE_MINIMIZED_HEIGHT,
            };
          } else {
            // Going back to full: enforce larger min size
            const nextWidth =
              n.width && n.width > NODE_MIN_WIDTH ? n.width : NODE_MIN_WIDTH;
            const nextHeight =
              n.height && n.height > NODE_MIN_HEIGHT
                ? n.height
                : NODE_MIN_HEIGHT;

            return {
              ...n,
              width: nextWidth,
              height: nextHeight,
            };
          }
        })
      );

      // Hide/show edges connected to this node
      rf.setEdges((eds) =>
        eds.map((e) =>
          e.source === id || e.target === id ? { ...e, hidden: next } : e
        )
      );

      return next;
    });
  }, [id, rf]);

  // ---------- RUN ACTION ----------
  const handleRun = useCallback(async () => {
    const code = data?.code ?? "";

    const widgetLines =
      (data?.widgetOutputs ?? [])
        .map((w) => {
          const val =
            typeof w.value === "string"
              ? `"${w.value}"` // string literal
              : JSON.stringify(w.value); // numbers, arrays, booleans
          return `${w.variable} = ${val}`;
        })
        .join("\n") + "\n\n";

    const finalCode = widgetLines + code;

    if (data?.onRun) {
      return data.onRun(id, finalCode);
    }

    try {
      setRunning(true);

      // Update the output panel
      setOutput({
        stdout: "",
        stderr: "",
      });

      const res = await fetch("http://127.0.0.1:5000/api/run-python", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: finalCode }),
      });

      const result = await res.json();

      // Update the output panel
      setOutput({
        stdout: result.stdout || "",
        stderr: result.stderr || "",
      });
      setRunningSuccess(true);
      setTimeout(() => setRunningSuccess(false), 2000);
    } catch (err) {
      console.error("Error running code:", err);
    } finally {
      setRunning(false);
    }
  }, [data, id]);

  // ---------- CODE CHANGE ----------
  const handleCodeChange = useCallback(
    (nextCode: string) => {
      // store inside node.data
      rf.setNodes((nodes) =>
        nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, code: nextCode } } : n
        )
      );
    },
    [id, rf]
  );

  return (
    <div className={`pcenode ${minimized ? "pcenode--minimized" : ""}`}>
      <NodeResizer
        minWidth={minimized ? NODE_MINIMIZED_WIDTH : NODE_MIN_WIDTH}
        maxWidth={Infinity}
        minHeight={minimized ? NODE_MINIMIZED_HEIGHT : NODE_MIN_HEIGHT}
        maxHeight={minimized ? NODE_MINIMIZED_HEIGHT : Infinity}
      />

      {minimized ? (
        <div className="pcenode__minimized">
          {/* Big run button */}
          <button
            type="button"
            className="pcenode__minimizedRunBtn"
            onClick={handleRun}
            disabled={running}
            aria-busy={running}
            title={running ? "Running code..." : "Run code"}
            style={{
              backgroundColor: "#f5d1d2",
              borderColor: "#cb181d",
              color: "#000",
            }}
          >
            {running ? (
              // Spinner instead of run icon
              <span className="pcenode__spinner" aria-hidden="true" />
            ) : runningSuccess ? (
              <img
                src={checkPng}
                alt="Success"
                className="pcenode__minimizedRunIcon"
              />
            ) : (
              <img
                src={runPng}
                alt="Run"
                className="pcenode__minimizedRunIcon"
              />
            )}

            <span className="pcenode__minimizedRunText">
              {running ? "Running..." : data?.title ?? "Code"}
            </span>
          </button>

          {/* Floating restore (top-left) */}
          <button
            type="button"
            className="pcenode__minimizedRestoreCircle_1 pcenode__minimizedRestoreCircle--topLeft"
            onClick={handleToggleMinimize}
          >
            <img src={expandPng} alt="Restore" />
          </button>

          {/* Floating restore (bottom-right) */}
          <button
            type="button"
            className="pcenode__minimizedRestoreCircle_2 pcenode__minimizedRestoreCircle--bottomRight"
            onClick={() => {}}
          >
            <img src={restartPng} alt="Run / update" />
          </button>
        </div>
      ) : (
        <>
          <div className="pcenode__header">
            <div className="pcenode__titleWrapper">
              <input
                type="text"
                className="pcenode__titleInput"
                value={data?.title ?? "Code"}
                onChange={handleTitleChange}
              />
            </div>

            <div className="pcenode__headerBtns">
              <button
                type="button"
                className="pcenode__iconBtn"
                onClick={handleToggleMinimize}
              >
                &#8211;
              </button>

              <button
                type="button"
                className="pcenode__iconBtn pcenode__iconBtn--close"
                onClick={handleClose}
              >
                ✕
              </button>
            </div>
          </div>

          <div className="pcenode__body">
            <div className="pcenode__editor">
              <div className="pcenode__editor-inner">
                <PythonCodeEditor
                  value={data?.code ?? ""}
                  onChange={handleCodeChange}
                />
              </div>
            </div>

            {(output.stdout || output.stderr) && (
              <div className="pcenode__output">
                {output.stdout && (
                  <pre className="pcenode__stdout">{output.stdout}</pre>
                )}
                {output.stderr && (
                  <pre className="pcenode__stderr">{output.stderr}</pre>
                )}
              </div>
            )}
          </div>

          <div className="pcenode__footer">
            <button
              type="button"
              onClick={() => {}}
              title="update"
              aria-label="update"
              className="pcenode__actionBtn"
            >
              <img
                src={restartPng}
                alt="update"
                className="pcenode__actionIcon"
              />
            </button>

            <button
              type="button"
              onClick={handleRun}
              title="Run code"
              aria-label="Run code"
              className="pcenode__actionBtn"
              disabled={running}
            >
              {running ? (
                <span className="pcenode__spinner" aria-hidden="true" />
              ) : runningSuccess ? (
                <img
                  src={checkPng}
                  alt="Success"
                  className="pcenode__actionIcon"
                />
              ) : (
                <img
                  src={runPng}
                  alt="Run Code"
                  className="pcenode__actionIcon"
                />
              )}
            </button>
          </div>
        </>
      )}

      <>
        <Handle
          type="target"
          position={Position.Top}
          id="viewport-in-1"
          className={`pcenode__handle ${
            minimized ? "pcenode__handle--hidden" : ""
          }`}
        />

        <Handle
          type="target"
          position={Position.Bottom}
          id="viewport-in-3"
          className={`pcenode__handle ${
            minimized ? "pcenode__handle--hidden" : ""
          }`}
        />

        <Handle
          type="target"
          position={Position.Left}
          id="viewport-in-2"
          className={`pcenode__handle pcenode__handle--left ${
            minimized ? "pcenode__handle--hidden" : ""
          }`}
        />

        <Handle
          type="source"
          position={Position.Right}
          id="viewport-out"
          className={`pcenode__handle pcenode__handle--right ${
            minimized ? "pcenode__handle--hidden" : ""
          }`}
        />
      </>
    </div>
  );
});

export default PyCodeEditorNode;
