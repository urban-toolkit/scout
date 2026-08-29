import { memo, useCallback } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import { Handle, Position, useReactFlow } from "@xyflow/react";

import BaseGrammarNode, {
  BaseNodeData,
} from "../../node-components/BaseGrammar";
import schema from "../../schemas/join.json";

import "../../node-components/BaseGrammar.css";

export type JoinNodeData = BaseNodeData;

export type JoinNode = Node<JoinNodeData, "joinNode">;

const JoinNode = memo(function JoinNode(props: NodeProps<JoinNode>) {
  const { id, data, selected } = props;
  const rf = useReactFlow();
  const { setEdges } = useReactFlow();

  const onCloseJoinNode = useCallback(
    (nodeId: string) => {
      rf.setNodes((nds) => nds.filter((n) => n.id !== nodeId));
      setEdges((eds) =>
        eds.filter((e) => e.source !== nodeId && e.target !== nodeId),
      );
    },
    [rf, setEdges],
  );

  return (
    <>
      <BaseGrammarNode
        id={id}
        selected={selected}
        data={{
          ...data,
          title: data.title ?? "Join",
          schema,
          pickInner: (v) => (v as any)?.join,
          onClose: onCloseJoinNode,
        }}
      />

      <Handle
        type="target"
        position={Position.Left}
        id="join-in-left"
        className="gnode__handle__target"
      />

      <Handle
        type="target"
        position={Position.Top}
        id="join-in-top"
        className="gnode__handle__target"
      />

      <Handle
        type="source"
        position={Position.Bottom}
        id="join-out-bottom"
        className="gnode__handle__source"
      />

      <Handle
        type="source"
        position={Position.Right}
        id="join-out-right"
        className="gnode__handle__source"
      />
    </>
  );
});

export default JoinNode;
