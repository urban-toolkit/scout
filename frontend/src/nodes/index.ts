import { BaseNodeData } from "../node-components/BaseGrammar";

import DataLayerNode, {
  DataLayerNode as DataLayerNodeType,
} from "./data-layer/DataLayerNode";

import ViewNode, {
  ViewNode as ViewNodeType,
  ViewNodeData,
} from "./view/ViewNode";

import ViewportNode, {
  ViewportNode as ViewportNodeType,
  ViewportNodeData,
} from "./view/ViewportNode";

import PyCodeEditorNode, {
  PyCodeEditorNode as PyCodeEditorNodeType,
  PyCodeEditorNodeData,
} from "./computation/PyCodeEditorNode";

import InteractionNode, {
  InteractionNode as InteractionNodeType,
  InteractionNodeData,
} from "./interaction/InteractionNode";

import WidgetDefNode, {
  WidgetDefNode as WidgetDefNodeType,
  WidgetDefNodeData,
} from "./widget/WidgetDefNode";

import WidgetViewNode, {
  WidgetViewNode as WidgetViewNodeType,
  WidgetViewNodeData,
} from "./widget/WidgetViewNode";

import WidgetNode, {
  WidgetNode as WidgetNodeType,
  WidgetNodeData,
} from "./widget/WidgetNode";

import ComparisonDefNode, {
  ComparisonDefNode as ComparisonDefNodeType,
  ComparisonDefNodeData,
} from "./comparison/ComparisonDefNode";

import ComparisonNode, {
  ComparisonNode as ComparisonNodeType,
  ComparisonNodeData,
} from "./comparison/ComparisonNode";

// import TransformationNode, {
//   TransformationNode as TransformationNodeType,
//   TransformationNodeData,
// } from "./TransformationNode";

import ComparisonViewNode, {
  ComparisonViewNode as ComparisonViewNodeType,
  ComparisonViewNodeData,
} from "./comparison/ComparisonViewNode";

// register all implemented node types
export const nodeTypes = {
  dataLayerNode: DataLayerNode,
  viewNode: ViewNode,
  pyCodeEditorNode: PyCodeEditorNode,
  viewportNode: ViewportNode,
  interactionNode: InteractionNode,
  widgetDefNode: WidgetDefNode,
  widgetViewNode: WidgetViewNode,
  // joinNode: JoinNode,
  // transformationNode: TransformationNode,
  comparisonDefNode: ComparisonDefNode,
  comparisonViewNode: ComparisonViewNode,
  comparisonNode: ComparisonNode,
  widgetNode: WidgetNode,
} as const;

// union helpers (extend as you add more)
export type AnyNode =
  | DataLayerNodeType
  | ViewNodeType
  | ViewportNodeType
  | InteractionNodeType
  // | TransformationNodeType
  | PyCodeEditorNodeType
  | WidgetDefNodeType
  | WidgetViewNodeType
  | ComparisonDefNodeType
  | ComparisonViewNodeType
  | ComparisonNodeType
  | WidgetNodeType;

export type AnyNodeData =
  | BaseNodeData
  | ViewNodeData
  | ViewportNodeData
  | InteractionNodeData
  // | TransformationNodeData
  | PyCodeEditorNodeData
  | WidgetDefNodeData
  | WidgetViewNodeData
  | ComparisonDefNodeData
  | ComparisonViewNodeData
  | ComparisonNodeData
  | WidgetNodeData;
