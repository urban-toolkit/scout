// src/nodes/InteractionNode.tsx

import { memo, useCallback } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import { Handle, Position, useReactFlow } from "@xyflow/react";
import BaseGrammarNode, { BaseNodeData } from "./BaseGrammarNode";
import schema from "../schemas/transformation.json";
import { PhysicalLayerDef } from "./utils/types";
import runPng from "../assets/run.png";
import "./BaseGrammarNode.css";

export type TransformationNodeData = BaseNodeData & {
  physical_layers?: PhysicalLayerDef[];
};

export type TransformationNode = Node<
  TransformationNodeData,
  "transformationNode"
>;

const TransformationNode = memo(function TransformationNode(
  props: NodeProps<TransformationNode>
) {
  const { id, data, selected } = props;

  const RunTransformation = useCallback(
    async (nodeData?: TransformationNodeData) => {
      const p_layers = nodeData?.physical_layers || [];
      const grammar = (nodeData as TransformationNodeData).value as any;
      if (!grammar.transformation.physical_layer) return;

      const pl = grammar.transformation.physical_layer.ref;
      const matchedLayer = p_layers.find((plDef) => plDef.id === pl);

      if (!matchedLayer) return;
      if (grammar.transformation.operation !== "rasterize") return;

      try {
        const response = await fetch(
          "http://127.0.0.1:5000/api/convert-to-raster",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(grammar.transformation),
          }
        );

        if (!response.ok) {
          throw new Error(`Server returned ${response.status}`);
        }
      } catch (err) {
        console.error("Error sending data to Flask:", err);
      }
    },
    []
  );

  return (
    <>
      <BaseGrammarNode
        id={id}
        selected={selected}
        data={{
          ...data,
          title: "Grammar • transformation",
          schema,
          pickInner: (v) => (v as any)?.transformation,
          // onClose: onCloseTransformationNode,
          footerActions: (
            <button
              type="button"
              onClick={async () => {
                await RunTransformation(data);
              }}
              title="Run transformation"
              aria-label="Run transformation"
              className="gnode__actionBtn"
            >
              <img
                src={runPng}
                alt="Run transformation"
                className="gnode__actionIcon"
              />
            </button>
          ),
        }}
      />

      <Handle
        type="target"
        position={Position.Left}
        id="transformation-in"
        className="gnode__handle gnode__handle--left"
      />

      <Handle
        type="source"
        position={Position.Right}
        id="transformation-out"
        className="gnode__handle gnode__handle--right"
      />
    </>
  );
});

export default TransformationNode;
