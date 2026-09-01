import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import type { NodeProps, Node } from "@xyflow/react";
import {
  Handle,
  Position,
  NodeResizer,
  useReactFlow,
  useUpdateNodeInternals,
} from "@xyflow/react";

import BaseGrammarNode, {
  BaseNodeData,
} from "../../node-components/BaseGrammar";
import schema from "../../schemas/widget.json";

import "./WidgetNode.css";
import flipPng from "../../assets/restart-2.png";
import restartPng from "../../assets/restart.png";
import expandPng from "../../assets/expand.png";
import checkPng from "../../assets/check-mark.png";

// import type { ReactNode } from "react";
import type { WidgetDef, WidgetOutput } from "../../utils/types";
import { renderWidgetFromWidgetDef } from "../../utils/renderWidget";
import { registerNodeAction } from "../../utils/nodeActionRegistry";

export type WidgetNodeData = BaseNodeData & {
  mode?: "def" | "view";
  pushToken?: string;
  output?: WidgetOutput;
};

export type WidgetNode = Node<WidgetNodeData, "widgetNode">;

const NODE_MIN_WIDTH = 200;
const NODE_MIN_HEIGHT = 80;
const NODE_MINIMIZED_WIDTH = 400;
const NODE_MINIMIZED_HEIGHT = 200;

const WidgetNode = memo(function WidgetNode(props: NodeProps<WidgetNode>) {
  const { id, data, selected } = props;
  const rf = useReactFlow();
  const { setNodes, setEdges } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();

  const mode = data.mode ?? "def";
  const [minimized, setMinimized] = useState(false);
  const [widgetValue, setWidgetValue] = useState<WidgetOutput | null>(null);
  const [updateStatus, setUpdateStatus] = useState<
    "idle" | "success" | "failed"
  >("idle");
  const updateStatusTimeout = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );

  // ---------- TITLE CHANGE ----------
  const handleTitleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const nextTitle = e.target.value;
      rf.setNodes((nodes) =>
        nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, title: nextTitle } } : n,
        ),
      );
    },
    [id, rf],
  );

  useEffect(() => {
    requestAnimationFrame(() => {
      updateNodeInternals(id);
    });
  }, [id, mode, minimized, updateNodeInternals]);

  const widget: WidgetDef | undefined = useMemo(() => {
    const v: any = (data as BaseNodeData)?.value;
    return v?.widget;
  }, [data]);

  const handleClose = useCallback(() => {
    setNodes((nds) => nds.filter((n) => n.id !== id));
    setEdges((eds) => eds.filter((e) => e.source !== id && e.target !== id));
  }, [id, setNodes, setEdges]);

  const goToView = useCallback(() => {
    const token = crypto.randomUUID();

    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? {
              ...n,
              width: n.width ?? 360,
              height: n.height ?? 260,
              data: {
                ...n.data,
                mode: "view",
                pushToken: token,
              },
            }
          : n,
      ),
    );
  }, [id, setNodes]);

  const goToDef = useCallback(() => {
    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? {
              ...n,
              data: {
                ...n.data,
                mode: "def",
              } as WidgetNodeData,
            }
          : n,
      ),
    );
  }, [id, setNodes]);

  const handleFlip = useCallback(() => {
    if (mode === "def") {
      goToView();
    } else {
      goToDef();
    }
  }, [mode, goToView, goToDef]);

  const handleToggleMinimize = useCallback(() => {
    setMinimized((prev) => {
      const next = !prev;

      rf.setNodes((nodes) =>
        nodes.map((n) => {
          if (n.id !== id) return n;

          if (next) {
            return {
              ...n,
              width: NODE_MINIMIZED_WIDTH,
              height: NODE_MINIMIZED_HEIGHT,
            };
          }

          const nextWidth =
            n.width && n.width > NODE_MIN_WIDTH ? n.width : NODE_MIN_WIDTH;
          const nextHeight =
            n.height && n.height > NODE_MIN_HEIGHT ? n.height : NODE_MIN_HEIGHT;

          return {
            ...n,
            width: nextWidth,
            height: nextHeight,
          };
        }),
      );

      rf.setEdges((eds) =>
        eds.map((e) =>
          e.source === id || e.target === id ? { ...e, hidden: next } : e,
        ),
      );

      return next;
    });
  }, [id, rf]);

  // False means the push found nothing to update (e.g. not connected to any
  // Code node) - anything else (true/undefined, for onRun implementations
  // that don't report a result) counts as success.
  const runPush = useCallback((): boolean => {
    if (!data?.onRun) return false;
    return data.onRun(id) !== false;
  }, [id, data]);

  // Flashes this widget's own button, same as Code/Data Layer nodes do -
  // used for both a manual click and the "Run Dataflow" orchestrator (see
  // the registration below) so a run visibly confirms this node actually
  // did something, not just the overall play button.
  const handleRun = useCallback((): boolean => {
    const ok = runPush();

    if (updateStatusTimeout.current) clearTimeout(updateStatusTimeout.current);
    setUpdateStatus(ok ? "success" : "failed");
    updateStatusTimeout.current = setTimeout(
      () => setUpdateStatus("idle"),
      2000,
    );
    return ok;
  }, [runPush]);

  // Lets the "Run Dataflow" orchestrator push this widget's value directly
  // and await the result - see utils/nodeActionRegistry.ts.
  useEffect(
    () => registerNodeAction(id, async () => handleRun()),
    [id, handleRun],
  );

  useEffect(() => {
    return () => {
      if (updateStatusTimeout.current) clearTimeout(updateStatusTimeout.current);
    };
  }, []);

  useEffect(() => {
    if (mode !== "view") return;
    if (!widget) return;

    // location-input's "default" is just the address text field's initial
    // display string, not a valid output - only the {lat, lon} an
    // AddressAutofill selection produces is. Resetting output to that string
    // here would silently overwrite a real geocoded value (from a prior
    // selection, or seeded example data) with something downstream code
    // can't use, causing failures far from this widget. Keep whatever
    // output already exists instead, and otherwise leave it unset so a push
    // correctly reports "nothing to send" rather than corrupting data.
    if (widget.wtype === "location-input") {
      if (data.output) setWidgetValue(data.output);
      return;
    }

    const out: WidgetOutput = {
      variable: widget.variable,
      value: widget["default"],
    };

    setWidgetValue(out);

    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? {
              ...n,
              data: {
                ...(n.data as WidgetNodeData),
                output: out,
              },
            }
          : n,
      ),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, widget, data.pushToken, id, setNodes]);

  // Keep the rendered control in sync with data.output when it's set from
  // outside this component's own onChange - e.g. the Widget Agent writing a
  // new value directly via setNodes. Without this, the shared graph data
  // (and anything reading it, like the connected Code node) updates
  // correctly, but the widget's own visible selection doesn't move.
  useEffect(() => {
    if (data.output) setWidgetValue(data.output);
  }, [data.output]);

  const updateIcon = updateStatus === "success" ? checkPng : restartPng;
  const updateTitle =
    updateStatus === "success"
      ? "Updated"
      : updateStatus === "failed"
        ? "Not connected to a Code node"
        : "Update";

  if (mode === "def") {
    return (
      <>
        {!minimized ? (
          <BaseGrammarNode
            id={id}
            selected={selected}
            data={{
              ...data,
              title: data.title ?? "Widget",
              schema,
              pickInner: (v) => (v as any)?.widget,
              onClose: () => handleClose(),
              onToggleMinimize: handleToggleMinimize,
              // onRun: () => handleRun(),
              footerActions: (
                <button
                  type="button"
                  onClick={handleFlip}
                  title="Flip to view"
                  aria-label="Flip to view"
                  className="gnode__actionBtn"
                >
                  <img
                    src={flipPng}
                    alt="Flip to view"
                    className="gnode__actionIcon"
                  />
                </button>
              ),
            }}
          />
        ) : (
          <div className="gnode gnode--minimized">
            <NodeResizer
              minWidth={NODE_MINIMIZED_WIDTH}
              maxWidth={Infinity}
              minHeight={NODE_MINIMIZED_HEIGHT}
              maxHeight={NODE_MINIMIZED_HEIGHT}
            />
            <div className="gnode__minimized">
              <button type="button" className="gnode__minimizedNodeTtitleBtn">
                {data.title ?? "Widget"}
              </button>

              <button
                type="button"
                className="gnode__minimizedRestoreCircle_1 gnode__minimizedRestoreCircle--topLeft"
                onClick={handleToggleMinimize}
              >
                <img src={expandPng} alt="Restore" />
              </button>

              <button
                type="button"
                className={`gnode__minimizedRestoreCircle_2 gnode__minimizedRestoreCircle--bottomRight${
                  updateStatus === "failed"
                    ? " gnode__minimizedRestoreCircle--failed"
                    : ""
                }`}
                onClick={handleRun}
                title={updateTitle}
              >
                <img src={updateIcon} alt={updateTitle} />
              </button>
            </div>
          </div>
        )}

        <Handle
          type="source"
          position={Position.Top}
          id="widget-out-1"
          className={`gnode__handle__source ${
            minimized ? "gnode__handle--hidden" : ""
          }`}
        />

        <Handle
          type="source"
          position={Position.Bottom}
          id="widget-out-2"
          className={`gnode__handle__source ${
            minimized ? "gnode__handle--hidden" : ""
          }`}
        />

        <Handle
          type="source"
          position={Position.Right}
          id="widget-out-3"
          className={`gnode__handle__source ${
            minimized ? "gnode__handle--hidden" : ""
          }`}
        />

        <Handle
          type="source"
          position={Position.Left}
          id="widget-out-4"
          className={`gnode__handle__source ${
            minimized ? "gnode__handle--hidden" : ""
          }`}
        />
      </>
    );
  }

  return (
    <div className="wvnode">
      <NodeResizer minWidth={NODE_MIN_WIDTH} minHeight={NODE_MIN_HEIGHT} />

      {!minimized && (
        <div className="wvnode__header">
          <div className="wvnode__titleWrapper">
            <input
              type="text"
              value={data.title ?? "Widget"}
              onChange={handleTitleChange}
              className="wvnode__titleInput"
            />
          </div>

          <div className="wvnode__headerBtns">
            <button
              type="button"
              className="wvnode__iconBtn"
              onClick={handleToggleMinimize}
            >
              &#8211;
            </button>

            <button
              type="button"
              className="wvnode__iconBtn wvnode__iconBtn--close"
              onClick={handleClose}
            >
              ✕
            </button>
          </div>
        </div>
      )}

      <div className="wvnode__body">
        {renderWidgetFromWidgetDef(
          widget,
          widgetValue?.value,
          (variable, val) => {
            const out: WidgetOutput = {
              variable,
              value: val,
            };

            setWidgetValue(out);

            setNodes((nds) =>
              nds.map((n) =>
                n.id === id
                  ? {
                      ...n,
                      data: {
                        ...(n.data as WidgetNodeData),
                        output: out,
                      },
                    }
                  : n,
              ),
            );
          },
        )}
      </div>

      {!minimized && (
        <div className="wvnode__footer">
          <button
            type="button"
            onClick={handleRun}
            title={updateTitle}
            aria-label={updateTitle}
            className={`wvnode__actionBtn${
              updateStatus === "failed" ? " wvnode__actionBtn--failed" : ""
            }`}
          >
            <img
              src={updateIcon}
              alt={updateTitle}
              className="wvnode__actionIcon"
            />
          </button>

          <button
            type="button"
            onClick={handleFlip}
            title="Flip to grammar"
            aria-label="Flip to grammar"
            className="wvnode__actionBtn"
          >
            <img
              src={flipPng}
              alt="Flip to grammar"
              className="wvnode__actionIcon"
            />
          </button>
        </div>
      )}

      <Handle
        type="source"
        position={Position.Top}
        id="widget-out-1"
        className={`wvnode__handle__source ${
          minimized ? "wvnode__handle--hidden" : ""
        }`}
      />

      <Handle
        type="source"
        position={Position.Left}
        id="widget-out-2"
        className={`wvnode__handle__source ${
          minimized ? "wvnode__handle--hidden" : ""
        }`}
      />

      <Handle
        type="source"
        position={Position.Right}
        id="widget-out-3"
        className={`wvnode__handle__source ${
          minimized ? "wvnode__handle--hidden" : ""
        }`}
      />

      <Handle
        type="source"
        position={Position.Bottom}
        id="widget-out-4"
        className={`wvnode__handle__source ${
          minimized ? "wvnode__handle--hidden" : ""
        }`}
      />

      {minimized && (
        <>
          <button
            type="button"
            className="wvnode__floatingBtn wvnode__floatingBtn--topLeft"
            onClick={handleToggleMinimize}
            title="Restore widget"
          >
            <img
              src={expandPng}
              alt="restore"
              className="wvnode__floatingIcon_2"
            />
          </button>

          {widget?.wtype !== "text" && (
            <button
              type="button"
              className={`wvnode__floatingBtn wvnode__floatingBtn--bottomRight${
                updateStatus === "failed" ? " wvnode__floatingBtn--failed" : ""
              }`}
              onClick={handleRun}
              title={updateTitle}
            >
              <img
                src={updateIcon}
                alt={updateTitle}
                className="wvnode__floatingIcon"
              />
            </button>
          )}
        </>
      )}
    </div>
  );
});

export default WidgetNode;
