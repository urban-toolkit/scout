import type { Dispatch, SetStateAction } from "react";
import type { Edge, Node } from "@xyflow/react";

import type { BaseNodeData } from "../node-components/BaseGrammar";
import type { PyCodeEditorNodeData } from "../nodes/computation/PyCodeEditorNode";

export type AppNodeData = BaseNodeData | PyCodeEditorNodeData;

export type LoadWorkflowArgs = {
  setNodes: Dispatch<SetStateAction<Node<AppNodeData>[]>>;
  setEdges: Dispatch<SetStateAction<Edge[]>>;
  getNode: (id: string) => Node | undefined;
  onRunInteraction: (srcId: string) => boolean;
  onRunWidget: (srcId: string) => boolean;
  setIdCounter?: (next: number) => void;
};

export function attachNodeBehaviors(
  node: Node<any>,
  setNodes: Dispatch<SetStateAction<Node<AppNodeData>[]>>,
  getNode: (id: string) => Node | undefined,
  onRunInteraction: (srcId: string) => boolean,
  onRunWidget: (srcId: string) => boolean,
): Node<AppNodeData> {
  const grammarTypes = new Set([
    "dataLayerNode",
    "viewNode",
    "interactionNode",
    "widgetNode",
    "comparisonNode",
  ]);

  if (!grammarTypes.has(node.type ?? "")) {
    return node as Node<AppNodeData>;
  }

  return {
    ...node,
    data: {
      ...node.data,
      onChange: (val: any, targetId: string) => {
        setNodes((nds) =>
          nds.map((n) =>
            n.id === targetId
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    value: val,
                  },
                }
              : n,
          ),
        );
      },
      onRun: (nodeId: string) => {
        const current = getNode(nodeId);
        if (!current) return;

        if (current.type === "interactionNode") {
          return onRunInteraction(nodeId);
        } else if (current.type === "widgetNode") {
          return onRunWidget(nodeId);
        }
      },
    },
  } as Node<AppNodeData>;
}

