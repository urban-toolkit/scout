import { memo, useCallback, useEffect, useState } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import { Position, NodeResizer, useReactFlow, Handle } from "@xyflow/react";
import "./WidgetViewNode.css";
import restartPng from "../assets/restart.png";
import expandPng from "../assets/expand.png";
import type { WidgetDef, WidgetOutput } from "./utils/types";
import { renderWidgetFromWidgetDef } from "./utils/renderWidget";

export type WidgetViewNodeData = {
  onClose?: (id: string) => void;
  onRun?: (srcId: string, trgId?: string) => void;
  widget?: WidgetDef;
  output?: WidgetOutput;
  pushToken?: string;
};

export type WidgetViewNode = Node<WidgetViewNodeData, "widgetViewNode">;

const WidgetViewNode = memo(function WidgetViewNode({
  id,
  data,
}: NodeProps<WidgetViewNode>) {
  const rf = useReactFlow();
  const { setNodes } = useReactFlow();

  // store current widget id + value
  const [widgetValue, setWidgetValue] = useState<WidgetOutput | null>(null);
  const [minimized, setMinimized] = useState(false);

  // inside WidgetViewNode component
  const handleClose = useCallback(() => {
    if (data?.onClose) {
      // App-level callback: (nodeId: string) => void
      return data.onClose(id);
    }
  }, [data, id]);

  // ---------- MINIMIZE TOGGLE ----------
  const handleToggleMinimize = useCallback(() => {
    setMinimized((prev) => {
      const next = !prev;

      rf.setEdges((eds) =>
        eds.map((e) =>
          e.source === id || e.target === id ? { ...e, hidden: next } : e
        )
      );

      return next;
    });
  }, [id, rf]);

  const onRun = useCallback(() => {
    if (data?.onRun) {
      // you can pass widgetValue along later if you want
      console.log("WidgetViewNode onRun value:", widgetValue);
      return data.onRun(id);
    }
    console.log("WidgetViewNode onRun value:", widgetValue);
  }, [data, id, widgetValue]);

  // if the widget definition changes (or node is created), sync default
  useEffect(() => {
    const w: any = data?.widget;
    if (!w) return;

    const out: WidgetOutput = {
      id: w.id,
      variable: w["variable"],
      value: w["default-value"],
    };
    console.log("Setting widget default value:", out.value);

    setWidgetValue(out);

    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? {
              ...n,
              data: {
                ...(n.data as WidgetViewNodeData),
                output: out,
              },
            }
          : n
      )
    );
  }, [data?.widget, data?.pushToken, id, setNodes]);

  return (
    <div className="wvnode">
      <NodeResizer />
      {!minimized && (
        <div className="wvnode__header">
          <div className="wvnode__title">Widget</div>

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

      <div
        className="wvnode__body"
        // style={{
        //   border: "1px solid #1f78b4",
        //   borderRadius: "12px",
        // }}
      >
        {renderWidgetFromWidgetDef(
          data?.widget,
          widgetValue?.value,
          (wid, v, val) => {
            const out: WidgetOutput = {
              id: wid,
              variable: v,
              value: val,
            };
            setWidgetValue(out);
            console.log("Widget value changed:", wid, v, val);

            setNodes((nds) =>
              nds.map((n) =>
                n.id === id
                  ? {
                      ...n,
                      data: {
                        ...(n.data as WidgetViewNodeData),
                        output: out,
                      },
                    }
                  : n
              )
            );
          }
        )}
      </div>

      {!minimized && (
        <div className="wvnode__footer">
          <button
            type="button"
            onClick={onRun}
            title="update"
            aria-label="update"
            className="wvnode__actionBtn"
          >
            <img src={restartPng} alt="update" className="wvnode__actionIcon" />
          </button>
        </div>
      )}

      <Handle
        type="source"
        position={Position.Top}
        id="widgetView-in-1"
        className={`wvnode__handle__target ${
          minimized ? "wvnode__handle--hidden" : ""
        }`}
      />

      <Handle
        type="target"
        position={Position.Left}
        id="widgetView-in-2"
        className={`wvnode__handle__source ${
          minimized ? "wvnode__handle--hidden" : ""
        }`}
      />

      <Handle
        type="source"
        position={Position.Right}
        id="widgetView-out-1"
        className={`wvnode__handle__target ${
          minimized ? "wvnode__handle--hidden" : ""
        }`}
      />

      <Handle
        type="source"
        position={Position.Bottom}
        id="widgetView-out-2"
        className={`wvnode__handle__target ${
          minimized ? "wvnode__handle--hidden" : ""
        }`}
      />

      {/* Minimized controls: vertical buttons on the right edge */}
      {minimized && (
        <>
          {/* Top-left: respawn / restore */}
          <button
            type="button"
            className="wvnode__floatingBtn wvnode__floatingBtn--topLeft"
            onClick={handleToggleMinimize}
            title="Restore widget"
          >
            <img
              src={expandPng}
              alt="update"
              className="wvnode__floatingIcon_2"
            />
          </button>

          {/* Bottom-right: run/update with restartPng */}
          {data.widget?.type !== "text" && (
            <button
              type="button"
              className="wvnode__floatingBtn wvnode__floatingBtn--bottomRight"
              onClick={onRun}
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

export default WidgetViewNode;
