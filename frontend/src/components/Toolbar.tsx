import "./Toolbar.css";

interface Props {
  onNavigateHome: () => void;
  onOpenDataflows: () => void;
  onOpenChartGallery: () => void;
  // Only relevant once a dataflow is actually open - the home page has no
  // canvas/chart-studio context yet, so it omits these along with inDataflow.
  inDataflow?: boolean;
  onOpenChartStudio?: () => void;
  onOpenDataCatalog?: () => void;
}

export default function Toolbar({
  onNavigateHome,
  onOpenDataflows,
  onOpenChartGallery,
  inDataflow = false,
  onOpenChartStudio,
  onOpenDataCatalog,
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
        <button
          type="button"
          className="toolbar__nav-item"
          onClick={onOpenChartGallery}
        >
          Chart Gallery
        </button>
        {inDataflow && (
          <>
            <button
              type="button"
              className="toolbar__nav-item"
              onClick={onOpenChartStudio}
            >
              Chart Studio
            </button>
            <button
              type="button"
              className="toolbar__nav-item"
              onClick={onOpenDataCatalog}
            >
              Data
            </button>
          </>
        )}
      </nav>
    </header>
  );
}
