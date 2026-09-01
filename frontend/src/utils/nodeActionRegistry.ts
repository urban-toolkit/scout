// A node's "run" behavior (fetch, execute, push, refresh) lives inside that
// node's own component - it's the only place with the right local state
// (loading flags, mode, etc.). The dataflow runner needs to invoke a
// specific node's action *and wait for it to finish* to respect dependency
// order, but React Flow doesn't expose component instances to call directly.
//
// This is a plain module-level registry (outside React state) that each
// runnable node registers its action into on mount and removes on unmount -
// the standard escape hatch for this kind of imperative, cross-component
// orchestration.

export type NodeAction = () => Promise<boolean>;

const registry = new Map<string, NodeAction>();

/** Call from a node's own effect: `useEffect(() => registerNodeAction(id, action), [id, action])`. */
export function registerNodeAction(nodeId: string, action: NodeAction): () => void {
  registry.set(nodeId, action);
  return () => {
    if (registry.get(nodeId) === action) registry.delete(nodeId);
  };
}

export function getNodeAction(nodeId: string): NodeAction | undefined {
  return registry.get(nodeId);
}
