import { BaseNodeData } from "../node-components/BaseGrammar";

import DataLayerNode, {
  DataLayerNode as DataLayerNodeType,
} from "./data-layer/DataLayerNode";

import JoinNode, {
  JoinNode as JoinNodeType,
  JoinNodeData,
} from "./join/JoinNode";

import ViewNode, {
  ViewNode as ViewNodeType,
  ViewNodeData,
} from "./view/ViewNode";

import PyCodeEditorNode, {
  PyCodeEditorNode as PyCodeEditorNodeType,
  PyCodeEditorNodeData,
} from "./computation/PyCodeEditorNode";

import InteractionNode, {
  InteractionNode as InteractionNodeType,
  InteractionNodeData,
} from "./interaction/InteractionNode";

import WidgetNode, {
  WidgetNode as WidgetNodeType,
  WidgetNodeData,
} from "./widget/WidgetNode";

import ComparisonNode, {
  ComparisonNode as ComparisonNodeType,
  ComparisonNodeData,
} from "./comparison/ComparisonNode";

// register all implemented node types
export const nodeTypes = {
  dataLayerNode: DataLayerNode,
  joinNode: JoinNode,
  viewNode: ViewNode,
  pyCodeEditorNode: PyCodeEditorNode,
  interactionNode: InteractionNode,
  comparisonNode: ComparisonNode,
  widgetNode: WidgetNode,
} as const;

// union helpers (extend as you add more)
export type AnyNode =
  | DataLayerNodeType
  | JoinNodeType
  | ViewNodeType
  | InteractionNodeType
  | PyCodeEditorNodeType
  | ComparisonNodeType
  | WidgetNodeType;

export type AnyNodeData =
  | BaseNodeData
  | JoinNodeData
  | ViewNodeData
  | InteractionNodeData
  | PyCodeEditorNodeData
  | ComparisonNodeData
  | WidgetNodeData;
