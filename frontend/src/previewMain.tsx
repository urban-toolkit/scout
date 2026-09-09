import { useState } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import ComputeConfigDialog from "./components/ComputeConfigDialog";
import type { ProjectComputeItem } from "./utils/dataflows";

const entry: ProjectComputeItem = {
  id: "routing",
  displayName: "Routing model",
  description: "",
  kind: "package",
  createdAt: new Date().toISOString(),
  callables: [
    {
      kind: "class",
      name: "DataLoader",
      importPath: "routing.scripts.load_static.DataLoader",
      sourceFile: "scripts/load_static.py",
      docstring: "",
      ctorParams: [{ name: "time", hasDefault: true, defaultRepr: "18", type: null }],
      methods: [],
    },
    {
      kind: "function",
      name: "calculate_weather_route",
      importPath: "routing.scripts.weather_routing.calculate_weather_route",
      sourceFile: "scripts/weather_routing.py",
      docstring: "",
      params: [{ name: "graph_path", hasDefault: false, defaultRepr: null, type: "str" }],
    },
  ],
};

function Preview() {
  const [open, setOpen] = useState(true);
  return (
    <div style={{ width: "100vw", height: "100vh", background: "#f1f5f9" }}>
      <ComputeConfigDialog
        open={open}
        onClose={() => setOpen(false)}
        entry={entry}
        onCreate={(r) => console.log("created", r)}
      />
    </div>
  );
}

const container = document.querySelector("#app");
const root = createRoot(container as HTMLElement);
root.render(<Preview />);
