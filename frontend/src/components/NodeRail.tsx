import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode, CSSProperties, DragEvent } from "react";
import { useReactFlow } from "@xyflow/react";
import Tooltip from "@mui/material/Tooltip";
import LayersOutlinedIcon from "@mui/icons-material/LayersOutlined";
import MergeTypeOutlinedIcon from "@mui/icons-material/MergeTypeOutlined";
import CodeOutlinedIcon from "@mui/icons-material/CodeOutlined";
import MapOutlinedIcon from "@mui/icons-material/MapOutlined";
import TouchAppOutlinedIcon from "@mui/icons-material/TouchAppOutlined";
import TuneOutlinedIcon from "@mui/icons-material/TuneOutlined";
import BarChartOutlinedIcon from "@mui/icons-material/BarChartOutlined";
import DeleteOutlineOutlinedIcon from "@mui/icons-material/DeleteOutlineOutlined";
import SkipNextIcon from "@mui/icons-material/SkipNext";
import CheckIcon from "@mui/icons-material/Check";
import CircularProgress from "@mui/material/CircularProgress";

import { TEMPLATE_LABELS, type TemplateKey } from "../templates";
import { runDataflow } from "../utils/dataflowRunner";
import "./NodeRail.css";

interface Props {
  onAdd: (tpl: TemplateKey) => void;
  onAddPyCodeEditor: () => void;
  onClear: () => void;
}

// dataTransfer key + sentinel the canvas' onDrop reads to tell a dragged
// rail icon apart from any other drag source (e.g. browser text/image drags)
// - "code" isn't a TemplateKey (it maps to onAddPyCodeEditor, not onAdd), so
// it needs its own sentinel value distinct from every real TemplateKey.
export const NODE_DRAG_MIME = "application/x-scout-node";
export const PY_CODE_DRAG_VALUE = "__pyCodeEditor__";

interface RailItem {
  key: string;
  label: string;
  icon: ReactNode;
  onClick: () => void;
  dragValue: string;
}

interface RailSection {
  title: string;
  accent: string;
  hoverBg: string;
  items: RailItem[];
}

const ICON_SX = { fontSize: 18 };

export default function NodeRail({
  onAdd,
  onAddPyCodeEditor,
  onClear,
}: Props) {
  const { screenToFlowPosition, getNodes, getEdges } = useReactFlow();

  const [runStatus, setRunStatus] = useState<
    "idle" | "running" | "success" | "failed"
  >("idle");
  const runStatusTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleRunDataflow = useCallback(async () => {
    if (runStatus === "running") return;
    if (runStatusTimeout.current) clearTimeout(runStatusTimeout.current);

    setRunStatus("running");
    let ok = false;
    try {
      const result = await runDataflow(getNodes(), getEdges());
      ok = result.ok;
    } catch {
      ok = false;
    }

    setRunStatus(ok ? "success" : "failed");
    runStatusTimeout.current = setTimeout(() => setRunStatus("idle"), 2000);
  }, [runStatus, getNodes, getEdges]);

  useEffect(() => {
    return () => {
      if (runStatusTimeout.current) clearTimeout(runStatusTimeout.current);
    };
  }, []);

  // New nodes always land at the viewport center, same as the old dropdown
  // menu - _desiredGrammarPos is the handoff createGrammarNode() reads.
  const getDropPosition = useCallback(() => {
    return screenToFlowPosition({
      x: window.innerWidth / 2,
      y: window.innerHeight / 2,
    });
  }, [screenToFlowPosition]);

  const handleAdd = useCallback(
    (tpl: TemplateKey) => {
      (window as any)._desiredGrammarPos = getDropPosition();
      onAdd(tpl);
    },
    [getDropPosition, onAdd],
  );

  const handleAddPyCodeEditor = useCallback(() => {
    (window as any)._desiredGrammarPos = getDropPosition();
    onAddPyCodeEditor();
  }, [getDropPosition, onAddPyCodeEditor]);

  // The canvas' onDrop computes the real drop position itself (from the drop
  // event's coordinates) - dragValue only needs to say *which* node to add.
  const handleDragStart = useCallback(
    (e: DragEvent<HTMLButtonElement>, dragValue: string) => {
      e.dataTransfer.setData(NODE_DRAG_MIME, dragValue);
      e.dataTransfer.effectAllowed = "move";
    },
    [],
  );

  const sections: RailSection[] = [
    {
      title: "Intelligence",
      accent: "#cb181d",
      hoverBg: "rgba(203, 24, 29, 0.1)",
      items: [
        {
          key: "data_layer",
          label: TEMPLATE_LABELS.data_layer,
          icon: <LayersOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("data_layer"),
          dragValue: "data_layer",
        },
        {
          key: "join",
          label: TEMPLATE_LABELS.join,
          icon: <MergeTypeOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("join"),
          dragValue: "join",
        },
        {
          key: "code",
          label: "Code",
          icon: <CodeOutlinedIcon sx={ICON_SX} />,
          onClick: handleAddPyCodeEditor,
          dragValue: PY_CODE_DRAG_VALUE,
        },
      ],
    },
    {
      title: "Design",
      accent: "#238b45",
      hoverBg: "rgba(35, 139, 69, 0.1)",
      items: [
        {
          key: "view",
          label: TEMPLATE_LABELS.view,
          icon: <MapOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("view"),
          dragValue: "view",
        },
      ],
    },
    {
      title: "Choice",
      accent: "#1f78b4",
      hoverBg: "rgba(31, 120, 180, 0.1)",
      items: [
        {
          key: "interaction",
          label: TEMPLATE_LABELS.interaction,
          icon: <TouchAppOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("interaction"),
          dragValue: "interaction",
        },
        {
          key: "widget",
          label: TEMPLATE_LABELS.widget,
          icon: <TuneOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("widget"),
          dragValue: "widget",
        },
        {
          key: "comparison",
          label: TEMPLATE_LABELS.comparison,
          icon: <BarChartOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("comparison"),
          dragValue: "comparison",
        },
      ],
    },
  ];

  return (
    <div className="node-rail-stack">
      <div className="node-rail">
        {sections.map((section) => (
          <Fragment key={section.title}>
            <div
              className="node-rail__section"
              style={
                {
                  "--rail-accent": section.accent,
                  "--rail-hover": section.hoverBg,
                } as CSSProperties
              }
            >
              <div className="node-rail__title">{section.title}</div>
              <div className="node-rail__grid">
                {section.items.map((item) => (
                  <Tooltip key={item.key} title={item.label} placement="right" arrow>
                    <button
                      type="button"
                      className="node-rail__icon"
                      aria-label={item.label}
                      onClick={item.onClick}
                      draggable
                      onDragStart={(e) => handleDragStart(e, item.dragValue)}
                    >
                      {item.icon}
                    </button>
                  </Tooltip>
                ))}
              </div>
            </div>
            <div className="node-rail__divider" />
          </Fragment>
        ))}

        <div
          className="node-rail__section"
          style={
            {
              "--rail-accent": "#64748b",
              "--rail-hover": "rgba(100, 116, 139, 0.1)",
            } as CSSProperties
          }
        >
          <div className="node-rail__grid">
            <Tooltip title="Clear canvas" placement="right" arrow>
              <button
                type="button"
                className="node-rail__icon"
                aria-label="Clear canvas"
                onClick={onClear}
              >
                <DeleteOutlineOutlinedIcon sx={ICON_SX} />
              </button>
            </Tooltip>
          </div>
        </div>
      </div>

      <Tooltip
        title={
          runStatus === "failed"
            ? "Run failed - see the failed node(s) for details"
            : "Run dataflow"
        }
        placement="right"
        arrow
      >
        <button
          type="button"
          className={`node-rail-play${
            runStatus === "failed" ? " node-rail-play--failed" : ""
          }`}
          aria-label="Run dataflow"
          onClick={() => void handleRunDataflow()}
          disabled={runStatus === "running"}
        >
          {runStatus === "running" ? (
            <CircularProgress size={20} color="inherit" />
          ) : runStatus === "success" ? (
            <CheckIcon sx={{ fontSize: 28 }} />
          ) : (
            <SkipNextIcon sx={{ fontSize: 36 }} />
          )}
        </button>
      </Tooltip>
    </div>
  );
}
