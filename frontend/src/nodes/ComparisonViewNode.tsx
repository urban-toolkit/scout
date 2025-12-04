import { memo, useCallback, useEffect, useState } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import { Position, NodeResizer, useReactFlow, Handle } from "@xyflow/react";
import "./ComparisonViewNode.css";
// import restartPng from "../assets/restart.png";
import type { ReactNode } from "react";

// Have to configure these
import type { ComparisonDef } from "./utils/types";
import { renderComparisonFromDef } from "./utils/renderComparison";

export type ComparisonViewNodeData = {
  onClose?: (id: string) => void;
  onRun?: (srcId: string, trgId?: string) => void;
  comparison?: ComparisonDef;
};

export type ComparisonViewNode = Node<
  ComparisonViewNodeData,
  "comparisonViewNode"
>;

const ComparisonViewNode = memo(function ComparisonViewNode({
  id,
  data,
}: NodeProps<ComparisonViewNode>) {
  const { getNode, setNodes, setEdges } = useReactFlow();

  const [bodyContent, setBodyContent] = useState<ReactNode | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // store current widget id + value

  // inside WidgetViewNode component
  const handleClose = useCallback(() => {
    const n = getNode(id);
    if (!n || n.type !== "comparisonViewNode") return;

    setNodes((nds) => nds.filter((nn) => nn.id !== id));
    setEdges((eds) => eds.filter((e) => e.source !== id && e.target !== id));
  }, [getNode, setNodes, setEdges, id]);

  //   useEffect(() => {
  //     console.log(data.comparison);
  //     const ctrl = new AbortController();
  //     (async () => {
  //       try {
  //         // await loadFromView(data);
  //       } catch (e: any) {
  //         // if (e?.name !== "AbortError") console.error(e);
  //       }
  //     })();

  //     return () => ctrl.abort();
  //   }, [
  //     data,
  //     // loadFromView
  //   ]);

  useEffect(() => {
    // no comparison → clear view
    if (!data.comparison) {
      setBodyContent(null);
      setError(null);
      setLoading(false);
      return;
    }

    const ctrl = new AbortController();
    setLoading(true);
    setError(null);

    (async () => {
      try {
        const content = await renderComparisonFromDef(
          data.comparison,
          ctrl.signal
        );
        if (!ctrl.signal.aborted) {
          setBodyContent(content);
        }
      } catch (e: any) {
        if (e?.name === "AbortError") return;
        console.error(e);
        setError(e.message ?? "Failed to load comparison");
      } finally {
        if (!ctrl.signal.aborted) {
          setLoading(false);
        }
      }
    })();

    return () => {
      ctrl.abort();
    };
  }, [data.comparison]);

  return (
    <div className="cvnode">
      <NodeResizer />

      <div className="cvnode__header">
        <div className="cvnode__title">Comparison</div>

        <div className="cvnode__headerBtns">
          <button
            type="button"
            className="cvnode__iconBtn cvnode__iconBtn--close"
            onClick={handleClose}
          >
            ✕
          </button>
        </div>
      </div>

      <div className="cvnode__body">
        {loading && <div>Loading…</div>}
        {error && <div className="cvnode__error">{error}</div>}
        {!loading && !error && bodyContent}
      </div>

      {/* <div className="cvnode__footer">
        <button
          type="button"
          onClick={onRun}
          title="update"
          aria-label="update"
          className="cvnode__actionBtn"
        >
          <img src={restartPng} alt="update" className="cvnode__actionIcon" />
        </button>
      </div> */}

      <Handle
        type="target"
        position={Position.Left}
        id="comparison-in-1"
        className="cvnode__handle__target"
      />

      <Handle
        type="target"
        position={Position.Bottom}
        id="comparison-in-2"
        className="cvnode__handle__target"
      />
    </div>
  );
});

export default ComparisonViewNode;
