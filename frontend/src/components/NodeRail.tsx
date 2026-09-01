import { Fragment, useCallback } from "react";
import type { ReactNode, CSSProperties } from "react";
import { useReactFlow } from "@xyflow/react";
import Tooltip from "@mui/material/Tooltip";
import LayersOutlinedIcon from "@mui/icons-material/LayersOutlined";
import MergeTypeOutlinedIcon from "@mui/icons-material/MergeTypeOutlined";
import CodeOutlinedIcon from "@mui/icons-material/CodeOutlined";
import MapOutlinedIcon from "@mui/icons-material/MapOutlined";
import TouchAppOutlinedIcon from "@mui/icons-material/TouchAppOutlined";
import TuneOutlinedIcon from "@mui/icons-material/TuneOutlined";
import BarChartOutlinedIcon from "@mui/icons-material/BarChartOutlined";
import WbSunnyOutlinedIcon from "@mui/icons-material/WbSunnyOutlined";
import WaterDropOutlinedIcon from "@mui/icons-material/WaterDropOutlined";
import AltRouteOutlinedIcon from "@mui/icons-material/AltRouteOutlined";
import DeleteOutlineOutlinedIcon from "@mui/icons-material/DeleteOutlineOutlined";

import { TEMPLATE_LABELS, type TemplateKey } from "../templates";
import "./NodeRail.css";

interface Props {
  onAdd: (tpl: TemplateKey) => void;
  onAddPyCodeEditor: () => void;
  onClear: () => void;
  onLoadShadowWorkflow: () => void;
  onLoadFloodingWorkflow: () => void;
  onLoadWeatherRoutingWorkflow: () => void;
}

interface RailItem {
  key: string;
  label: string;
  icon: ReactNode;
  onClick: () => void;
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
  onLoadShadowWorkflow,
  onLoadFloodingWorkflow,
  onLoadWeatherRoutingWorkflow,
}: Props) {
  const { screenToFlowPosition } = useReactFlow();

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
        },
        {
          key: "join",
          label: TEMPLATE_LABELS.join,
          icon: <MergeTypeOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("join"),
        },
        {
          key: "code",
          label: "Code",
          icon: <CodeOutlinedIcon sx={ICON_SX} />,
          onClick: handleAddPyCodeEditor,
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
        },
        {
          key: "widget",
          label: TEMPLATE_LABELS.widget,
          icon: <TuneOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("widget"),
        },
        {
          key: "comparison",
          label: TEMPLATE_LABELS.comparison,
          icon: <BarChartOutlinedIcon sx={ICON_SX} />,
          onClick: () => handleAdd("comparison"),
        },
      ],
    },
    {
      title: "Examples",
      accent: "#64748b",
      hoverBg: "rgba(100, 116, 139, 0.1)",
      items: [
        {
          key: "shadow",
          label: "Shadow",
          icon: <WbSunnyOutlinedIcon sx={ICON_SX} />,
          onClick: onLoadShadowWorkflow,
        },
        {
          key: "flooding",
          label: "Flooding",
          icon: <WaterDropOutlinedIcon sx={ICON_SX} />,
          onClick: onLoadFloodingWorkflow,
        },
        {
          key: "routing",
          label: "Routing",
          icon: <AltRouteOutlinedIcon sx={ICON_SX} />,
          onClick: onLoadWeatherRoutingWorkflow,
        },
      ],
    },
  ];

  return (
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
  );
}
