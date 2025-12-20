import { BaseNodeData } from "./BaseGrammarNode";

import DataLayerNode, {
  DataLayerNode as DataLayerNodeType,
} from "./DataLayerNode";

import ViewNode, { ViewNode as ViewNodeType, ViewNodeData } from "./ViewNode";

import ViewportNode, {
  ViewportNode as ViewportNodeType,
  ViewportNodeData,
} from "./ViewportNode";

import PyCodeEditorNode, {
  PyCodeEditorNode as PyCodeEditorNodeType,
  PyCodeEditorNodeData,
} from "./PyCodeEditorNode";

import InteractionNode, {
  InteractionNode as InteractionNodeType,
  InteractionNodeData,
} from "./InteractionNode";

import WidgetDefNode, {
  WidgetDefNode as WidgetDefNodeType,
  WidgetDefNodeData,
} from "./WidgetDefNode";

import ComparisonDefNode, {
  ComparisonDefNode as ComparisonDefNodeType,
  ComparisonDefNodeData,
} from "./ComparisonDefNode";

import WidgetViewNode, {
  WidgetViewNode as WidgetViewNodeType,
  WidgetViewNodeData,
} from "./WidgetViewNode";

// import TransformationNode, {
//   TransformationNode as TransformationNodeType,
//   TransformationNodeData,
// } from "./TransformationNode";

import ComparisonViewNode, {
  ComparisonViewNode as ComparisonViewNodeType,
  ComparisonViewNodeData,
} from "./ComparisonViewNode";

// register all implemented node types
export const nodeTypes = {
  dataLayerNode: DataLayerNode,
  viewNode: ViewNode,
  viewportNode: ViewportNode,
  pyCodeEditorNode: PyCodeEditorNode,
  interactionNode: InteractionNode,
  widgetDefNode: WidgetDefNode,
  widgetViewNode: WidgetViewNode,
  // joinNode: JoinNode,
  // transformationNode: TransformationNode,
  // choiceNode: ChoiceNode,
  comparisonDefNode: ComparisonDefNode,
  comparisonViewNode: ComparisonViewNode,
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
  | ComparisonViewNodeType;

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
  | ComparisonViewNodeData;
