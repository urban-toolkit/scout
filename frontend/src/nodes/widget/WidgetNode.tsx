import { memo, useCallback, useEffect, useMemo, useState } from "react";
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

// import type { ReactNode } from "react";
import type { WidgetDef, WidgetOutput } from "../../utils/types";
import { renderWidgetFromWidgetDef } from "../../utils/renderWidget";

export type WidgetNodeData = BaseNodeData & {
  mode?: "def" | "view";
  pushToken?: string;
  output?: WidgetOutput;
};

export type WidgetNode = Node<WidgetNodeData, "widgetNode">;

const NODE_MIN_WIDTH = 300;
const NODE_MIN_HEIGHT = 180;
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

  const handleRun = useCallback(() => {
    // if (mode === "def") {
    //   goToView();
    //   return;
    // }

    // const token = crypto.randomUUID();
    // setNodes((nds) =>
    //   nds.map((n) =>
    //     n.id === id
    //       ? {
    //           ...n,
    //           data: {
    //             ...n.data,
    //             pushToken: token,
    //           },
    //         }
    //       : n,
    //   ),
    // );

    if (data?.onRun) {
      data.onRun(id);
    }
  }, [
    // mode,
    // goToView,
    id,
    // setNodes,
    data,
  ]);

  useEffect(() => {
    if (mode !== "view") return;
    if (!widget) return;

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
  }, [mode, widget, data.pushToken, id, setNodes]);

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
                className="gnode__minimizedRestoreCircle_2 gnode__minimizedRestoreCircle--bottomRight"
                onClick={handleRun}
              >
                <img src={restartPng} alt="Fetch / update" />
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
      <NodeResizer />

      {!minimized && (
        <div className="wvnode__header">
          <div className="wvnode__title">{data.title ?? "Widget"}</div>

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
            title="update"
            aria-label="update"
            className="wvnode__actionBtn"
          >
            <img src={restartPng} alt="update" className="wvnode__actionIcon" />
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
              className="wvnode__floatingBtn wvnode__floatingBtn--bottomRight"
              onClick={handleRun}
              title="update"
            >
              <img
                src={restartPng}
                alt="update"
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
