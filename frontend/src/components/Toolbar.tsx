import { useState } from "react";
import type { MouseEvent, ReactNode } from "react";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import WbSunnyOutlinedIcon from "@mui/icons-material/WbSunnyOutlined";
import WaterDropOutlinedIcon from "@mui/icons-material/WaterDropOutlined";
import AltRouteOutlinedIcon from "@mui/icons-material/AltRouteOutlined";
import "./Toolbar.css";

interface Props {
  onLoadShadowWorkflow: () => void;
  onLoadFloodingWorkflow: () => void;
  onLoadWeatherRoutingWorkflow: () => void;
}

interface ProjectItem {
  key: string;
  label: string;
  icon: ReactNode;
  onClick: () => void;
}

const ICON_SX = { fontSize: 18 };

export default function Toolbar({
  onLoadShadowWorkflow,
  onLoadFloodingWorkflow,
  onLoadWeatherRoutingWorkflow,
}: Props) {
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  const open = Boolean(anchorEl);

  const projects: ProjectItem[] = [
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
  ];

  const handleOpenProjects = (e: MouseEvent<HTMLElement>) =>
    setAnchorEl(e.currentTarget);
  const handleCloseProjects = () => setAnchorEl(null);
  const handleSelectProject = (item: ProjectItem) => {
    handleCloseProjects();
    item.onClick();
  };

  return (
    <header className="toolbar">
      <div className="toolbar__brand">
        <img src="/scout.png" alt="Scout" className="toolbar__logo" />
      </div>

      <nav className="toolbar__nav">
        <button
          type="button"
          className="toolbar__nav-item"
          onClick={handleOpenProjects}
          aria-haspopup="true"
          aria-expanded={open}
        >
          Projects
          <KeyboardArrowDownIcon sx={ICON_SX} />
        </button>
        <Menu
          anchorEl={anchorEl}
          open={open}
          onClose={handleCloseProjects}
          anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
        >
          {projects.map((item) => (
            <MenuItem key={item.key} onClick={() => handleSelectProject(item)}>
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText>{item.label}</ListItemText>
            </MenuItem>
          ))}
        </Menu>

        {/* Placeholder nav items - no pages/routes yet, wired up later */}
        <button type="button" className="toolbar__nav-item" onClick={() => {}}>
          Chart Gallery
        </button>
        <button type="button" className="toolbar__nav-item" onClick={() => {}}>
          Chart Studio
        </button>
      </nav>
    </header>
  );
}
