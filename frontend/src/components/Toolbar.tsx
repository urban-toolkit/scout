import "./Toolbar.css";

interface Props {
  onNavigateHome: () => void;
  onOpenDataflows: () => void;
  onOpenChartGallery: () => void;
  // Ungated (unlike inDataflow's props below) - the AI chat panel is
  // available on every page, not just the canvas (see App.tsx), so its
  // trigger lives here alongside Dataflows/Chart Gallery rather than inside
  // the inDataflow-only group.
  onOpenChat: () => void;
  // Only relevant once a dataflow is actually open - the home page has no
  // canvas context yet, so it omits these along with inDataflow. Data/Compute
  // are dataflow-scoped (each dataflow has its own project list), so unlike
  // onOpenChat they can't be ungated the same way.
  inDataflow?: boolean;
  onOpenDataCatalog?: () => void;
  onOpenComputeCatalog?: () => void;
}

export default function Toolbar({
  onNavigateHome,
  onOpenDataflows,
  onOpenChartGallery,
  onOpenChat,
  inDataflow = false,
  onOpenDataCatalog,
  onOpenComputeCatalog,
}: Props) {
  return (
    <header className="toolbar">
      <button
        type="button"
        className="toolbar__brand"
        onClick={onNavigateHome}
        aria-label="Go to home"
      >
        <img src="/scout.png" alt="Scout" className="toolbar__logo" />
      </button>

      <nav className="toolbar__nav">
        <button
          type="button"
          className="toolbar__nav-item"
          onClick={onOpenDataflows}
        >
          Dataflows
        </button>
        {inDataflow && (
          <>
            <button
              type="button"
              className="toolbar__nav-item"
              onClick={onOpenDataCatalog}
            >
              Data
            </button>
            <button
              type="button"
              className="toolbar__nav-item"
              onClick={onOpenComputeCatalog}
            >
              Compute
            </button>
          </>
        )}
        <button
          type="button"
          className="toolbar__nav-item"
          onClick={onOpenChat}
        >
          Agent
        </button>
        <button
          type="button"
          className="toolbar__nav-item"
          onClick={onOpenChartGallery}
        >
          Chart Gallery
        </button>
      </nav>
    </header>
  );
}
