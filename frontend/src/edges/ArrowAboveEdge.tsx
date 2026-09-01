import type { ComponentProps } from "react";
import {
  BaseEdge,
  BezierEdge,
  EdgeLabelRenderer,
  Position,
  getBezierPath,
} from "@xyflow/react";

// The default edge draws its arrowhead as an SVG <marker> on the same path,
// so it shares the path's stacking position - with nodes painted above edges
// (so edge lines don't cross over unrelated node cards), the marker triangle
// gets clipped by whichever node it enters. EdgeLabelRenderer's container
// sits above .react-flow__nodes (see index.css), so drawing the arrowhead
// there instead - as a small rotated SVG triangle at the target point, with
// geometry copied 1:1 from xyflow's own ArrowClosed marker so it renders
// pixel-identical to the default, just relocated - keeps the line tucked
// behind nodes while the arrowhead stays fully visible on top of them.
type Props = ComponentProps<typeof BezierEdge>;

// getBezierPath's target control point is always axis-aligned with the
// target point for Left/Right/Top/Bottom handles (see @xyflow/system's
// getControlWithCurvature), so the curve's tangent at the target - and thus
// the arrowhead's angle - is exactly determined by the target handle side.
const ANGLE_BY_TARGET_POSITION: Record<Position, number> = {
  [Position.Left]: 0,
  [Position.Right]: 180,
  [Position.Top]: 90,
  [Position.Bottom]: -90,
};

// Copied from @xyflow/react's own ArrowClosedSymbol: a triangle whose tip
// sits at local (0,0) - the path's exact endpoint - pointing along +x, with
// its flat back at x=-5. viewBox spans -10..10 on both axes to match the
// marker's own <marker viewBox="-10 -10 20 20">.
const ARROW_POINTS = "-5,-4 0,0 -5,4 -5,-4";

// The default marker is sized via markerUnits="strokeWidth": actual pixel
// size = markerWidth/markerHeight attr (20, set in defaultEdgeOptions) times
// the edge's own stroke-width. Reproducing that here keeps the size in sync
// if either ever changes.
const MARKER_WIDTH_ATTR = 20;

export default function ArrowAboveEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
}: Props) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const angle = ANGLE_BY_TARGET_POSITION[targetPosition];
  const color = style?.stroke ?? "#888";
  const strokeWidth =
    typeof style?.strokeWidth === "number" ? style.strokeWidth : 1;
  const boxSize = MARKER_WIDTH_ATTR * strokeWidth;
  const half = boxSize / 2;

  return (
    <>
      <BaseEdge id={id} path={edgePath} style={style} />
      <EdgeLabelRenderer>
        <div
          style={{
            position: "absolute",
            width: boxSize,
            height: boxSize,
            transform: `translate(${targetX - half}px, ${targetY - half}px) rotate(${angle}deg)`,
            pointerEvents: "none",
          }}
        >
          <svg
            width={boxSize}
            height={boxSize}
            viewBox="-10 -10 20 20"
            style={{ overflow: "visible" }}
          >
            <polygon points={ARROW_POINTS} fill={color} />
          </svg>
        </div>
      </EdgeLabelRenderer>
    </>
  );
}
