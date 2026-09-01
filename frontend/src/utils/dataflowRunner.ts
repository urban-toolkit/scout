import type { Edge, Node } from "@xyflow/react";
import { getNodeAction } from "./nodeActionRegistry";

export interface DataflowResult {
  ok: boolean;
  ranNodeIds: string[];
  skippedNodeIds: string[];
  failedNodeIds: string[];
}

/**
 * Runs every node's registered action in dependency order (Kahn's algorithm
 * over the canvas's edges), one at a time. A node whose action returns false
 * or throws is marked failed; anything downstream of a failed or skipped
 * node is skipped in turn (its input can't be trusted) - but independent
 * branches elsewhere in the graph still run. Nodes with no registered
 * action (e.g. Join, which has no backend behavior yet) are no-ops that
 * pass through so their dependents still get a chance to run.
 */
export async function runDataflow(
  nodes: Node[],
  edges: Edge[],
): Promise<DataflowResult> {
  const nodeIds = nodes.map((n) => n.id);
  const idSet = new Set(nodeIds);

  const dependents = new Map<string, string[]>();
  const inDegree = new Map<string, number>();
  nodeIds.forEach((id) => {
    inDegree.set(id, 0);
    dependents.set(id, []);
  });

  for (const e of edges) {
    if (!idSet.has(e.source) || !idSet.has(e.target)) continue;
    dependents.get(e.source)!.push(e.target);
    inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1);
  }

  // Kahn's algorithm. Seeding the queue in the canvas's own node order keeps
  // the result deterministic for nodes that are equally ready to run.
  const queue: string[] = nodeIds.filter((id) => inDegree.get(id) === 0);
  const remaining = new Map(inDegree);
  const order: string[] = [];

  while (queue.length) {
    const id = queue.shift()!;
    order.push(id);
    for (const dep of dependents.get(id) ?? []) {
      const next = (remaining.get(dep) ?? 0) - 1;
      remaining.set(dep, next);
      if (next === 0) queue.push(dep);
    }
  }

  // Anything left out of `order` sits on a cycle (shouldn't normally happen
  // given the app's own connection rules) - run it last, best-effort,
  // rather than silently dropping it.
  const scheduled = new Set(order);
  for (const id of nodeIds) {
    if (!scheduled.has(id)) order.push(id);
  }

  const failed = new Set<string>();
  const skipped = new Set<string>();
  const ran: string[] = [];

  for (const id of order) {
    const upstreamBroken = edges.some(
      (e) => e.target === id && (failed.has(e.source) || skipped.has(e.source)),
    );
    if (upstreamBroken) {
      skipped.add(id);
      continue;
    }

    const action = getNodeAction(id);
    if (!action) continue;

    try {
      const ok = await action();
      if (ok) {
        ran.push(id);
      } else {
        failed.add(id);
      }
    } catch {
      failed.add(id);
    }
  }

  return {
    ok: failed.size === 0,
    ranNodeIds: ran,
    skippedNodeIds: [...skipped],
    failedNodeIds: [...failed],
  };
}
