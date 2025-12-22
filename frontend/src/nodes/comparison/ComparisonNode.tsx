import { memo, useCallback, useEffect, useMemo, useState } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import { Handle, Position, NodeResizer, useReactFlow } from "@xyflow/react";
import type { ReactNode } from "react";

import BaseGrammarNode, {
  BaseNodeData,
} from "../../node-components/BaseGrammar";
import schema from "../../schemas/comparison.json";

import type { ComparisonDef } from "../../utils/types";
import { renderComparisonFromDef } from "../../utils/renderComparison";

import "./ComparisonViewNode.css"; // reuse if you want, or make a new css

export type ComparisonNodeData = BaseNodeData & {
  mode?: "def" | "view";
  previewToken?: string;
};

export type ComparisonNode = Node<ComparisonNodeData, "comparisonNode">;

const ComparisonNode = memo(function ComparisonNode(
  props: NodeProps<ComparisonNode>
) {
  const { id, data, selected } = props;
  const { getNode, setNodes, setEdges } = useReactFlow();

  const mode = data.mode ?? "def";

  const [bodyContent, setBodyContent] = useState<ReactNode | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // close removes node + edges (same as your view node close)
  const handleClose = useCallback(() => {
    setNodes((nds) => nds.filter((n) => n.id !== id));
    setEdges((eds) => eds.filter((e) => e.source !== id && e.target !== id));
  }, [id, setNodes, setEdges]);

  // read comparison def from grammar value
  const comparison: ComparisonDef | undefined = useMemo(() => {
    const v: any = (data as BaseNodeData)?.value;
    return v?.comparison;
  }, [data]);

  const goToView = useCallback(() => {
    const token = crypto.randomUUID();
    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? {
              ...n,
              width: n.width ?? 420,
              height: n.height ?? 320,
              data: { ...n.data, mode: "view", previewToken: token },
            }
          : n
      )
    );
  }, [id, setNodes]);

  const goToDef = useCallback(() => {
    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? { ...n, data: { ...n.data, mode: "def" } as ComparisonNodeData }
          : n
      )
    );
  }, [id, setNodes]);

  // render when in view mode AND comparison changes OR previewToken changes
  useEffect(() => {
    if (mode !== "view") return;

    if (!comparison) {
      setBodyContent(null);
      setError("No comparison definition found.");
      setLoading(false);
      return;
    }

    const ctrl = new AbortController();
    setLoading(true);
    setError(null);

    (async () => {
      try {
        const content = await renderComparisonFromDef(comparison, ctrl.signal);
        if (!ctrl.signal.aborted) setBodyContent(content);
      } catch (e: any) {
        if (e?.name === "AbortError") return;
        console.error(e);
        setError(e?.message ?? "Failed to load comparison");
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    })();

    return () => ctrl.abort();
    // include previewToken so "Run" re-renders even if definition object reference is same
  }, [mode, comparison, data.previewToken]);

  if (mode === "def") {
    return (
      <>
        <BaseGrammarNode
          id={id}
          selected={selected}
          data={{
            ...data,
            title: data.title ?? "Comparison",
            schema,
            pickInner: (v) => (v as any)?.comparison,
            onClose: handleClose,
            // Add a "Run" hook: your BaseGrammarNode likely already has a run button that calls data.onRun(id)
            // If you want Run to switch to view mode, do it here by overriding onRun:
            onRun: () => goToView(),
          }}
        />

        <Handle
          type="target"
          position={Position.Left}
          id="comparison-in-1"
          className="gnode__handle__target"
        />
      </>
    );
  }

  // mode === "view"
  return (
    <div className="cvnode">
      <NodeResizer />

      <div className="cvnode__header">
        <div className="cvnode__title">{data.title ?? "Comparison"}</div>

        <div className="cvnode__headerBtns">
          <button type="button" className="cvnode__iconBtn" onClick={goToDef}>
            ←
          </button>
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

export default ComparisonNode;
