import {
  memo,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import type { NodeProps, Node } from "@xyflow/react";
import { Position, NodeResizer, useReactFlow, Handle } from "@xyflow/react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./ViewportNode.css";
import restartPng from "../../assets/restart.png";
import persistPng from "../../assets/update-data.png";
import mapPng from "../../assets/map.png";
import checkPng from "../../assets/check-mark.png";
import * as d3 from "d3";
import { ViewDef, InteractionDef } from "../../utils/types";
import { renderLayers } from "../../utils/renderViewLayers";

export type ViewportNodeData = {
  title?: string;
  center?: [number, number];
  onClose?: (id: string) => void;
  onRun?: (srcId: string, trgId?: string) => void;
  view?: ViewDef[];
  interactions?: InteractionDef[];
};

export type ViewportNode = Node<ViewportNodeData, "viewportNode">;

const ViewportNode = memo(function ViewportNode({
  id,
  data,
}: NodeProps<ViewportNode>) {
  const [persisting, setPersisting] = useState(false);
  const [persistSuccess, setPersistSuccess] = useState(false);
  const [showBasemap, setShowBasemap] = useState(false);
  const baseLayerRef = useRef<L.TileLayer | null>(null);

  const { getEdges, setEdges } = useReactFlow();
  const mapRef = useRef<HTMLDivElement | null>(null);
  const leafletRef = useRef<L.Map | null>(null);

  const pendingRef = useRef<Record<string, any>>({});
  const wasDraggedRef = useRef(false);

  // ---- D3 / SVG overlay refs ----
  const svgLayerRef = useRef<L.SVG | null>(null);
  const overlaySvgRef = useRef<d3.Selection<
    SVGSVGElement,
    unknown,
    null,
    undefined
  > | null>(null);
  const gRootRef = useRef<d3.Selection<
    SVGGElement,
    unknown,
    null,
    undefined
  > | null>(null);
  const gByTagRef = useRef<
    Map<string, d3.Selection<SVGGElement, unknown, null, undefined>>
  >(new Map());
  const rf = useReactFlow();

  // ---------- TITLE CHANGE ----------
  const handleTitleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const nextTitle = e.target.value;
      rf.setNodes((nodes) =>
        nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, title: nextTitle } } : n
        )
      );
    },
    [id, rf]
  );

  const shouldHandleClick = useCallback(() => {
    if (wasDraggedRef.current) {
      // A drag/zoom happened since last stable state → ignore this click
      wasDraggedRef.current = false;
      return false;
    }
    // No drag since last time → treat as a real click
    return true;
  }, []);

  const onPersist = useCallback(async () => {
    const entries = Object.values(pendingRef.current) as {
      ref: string;
      geojson: any;
    }[];

    if (!entries.length) return;

    setPersisting(true);
    setPersistSuccess(false);

    try {
      const tasks = entries.map(({ ref, geojson }) =>
        fetch("http://127.0.0.1:5000/api/update-data-layer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ref,
            geojson,
          }),
        })
      );

      await Promise.allSettled(tasks);
      pendingRef.current = {};

      setPersistSuccess(true);
      setTimeout(() => setPersistSuccess(false), 2000);
    } finally {
      setPersisting(false);
    }

    // clear pending once sent
    pendingRef.current = {};
  }, []);

  // Leaflet-aware D3 path generator
  const makeLeafletPath = useCallback((map: L.Map) => {
    const projectPoint = function (this: any, x: number, y: number) {
      const point = map.latLngToLayerPoint([y, x]); // [lat, lon] -> layer point
      this.stream.point(point.x, point.y);
    };
    const transform = d3.geoTransform({ point: projectPoint as any });
    return d3.geoPath(transform as any);
  }, []);

  const getOrCreateTagGroup = useCallback((tag: string) => {
    const m = gByTagRef.current;
    if (m.has(tag)) return m.get(tag)!;
    const gRoot = gRootRef.current!;
    const g = gRoot.append("g").attr("class", `tag-${tag}`);
    m.set(tag, g);
    return g;
  }, []);

  const clearAllSvgLayers = useCallback(() => {
    gByTagRef.current.forEach((g) => g.remove());
    gByTagRef.current.clear();
  }, []);

  // Redraw all feature paths when the map moves/zooms
  const redrawAll = useCallback(() => {
    const map = leafletRef.current;
    if (!map) return;

    gByTagRef.current.forEach((g) => {
      const geomType = (g as any)._geomType;
      const isPointLayer = (g as any)._isPointLayer;
      const configuredRadius = (g as any)._pointRadius;

      const path = makeLeafletPath(map);

      if (isPointLayer && typeof (path as any).pointRadius === "function") {
        const pointRadius = configuredRadius ?? 4;
        (path as any).pointRadius(pointRadius);
      }

      g.selectAll<SVGPathElement, any>("path.geom, path.geom-border").attr(
        "d",
        path as any
      );
    });
  }, [makeLeafletPath]);

  const loadFromView = useCallback(
    async (nodeData?: ViewportNodeData) => {
      const map = leafletRef.current;

      // If map not ready OR no spec -> clear drawings and bail
      if (!map) {
        return;
      }

      const views = nodeData?.view ?? [];
      const interactions = nodeData?.interactions ?? [];

      await renderLayers({
        id,
        map,
        views,
        interactions,
        clearAllSvgLayers,
        makeLeafletPath,
        getOrCreateTagGroup,
        onDirty: ({ ref, featureCollection }) => {
          const key = ref;
          pendingRef.current[key] = {
            ref,
            geojson: featureCollection,
          };
        },
        shouldHandleClick,
      });

      map.invalidateSize();
      // paths will auto-reproject because we listen for map move/zoom in init
    },
    [
      clearAllSvgLayers,
      getOrCreateTagGroup,
      id,
      makeLeafletPath,
      shouldHandleClick,
    ]
  );

  // Init Leaflet + SVG overlay once
  useEffect(() => {
    if (!mapRef.current || leafletRef.current) return;
    const map = L.map(mapRef.current, {
      attributionControl: false,
      preferCanvas: true,
    });

    setTimeout(() => map.invalidateSize(), 0);

    const center: [number, number] = data?.center ?? [41.881, -87.63];
    const zoom = 14;
    map.setView(center, zoom);

    const baseLayer = L.tileLayer(
      // https://wiki.openstreetmap.org/wiki/Raster_tile_providers
      "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
      {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors",
        opacity: 0,
      }
    ).addTo(map);

    baseLayerRef.current = baseLayer;

    // SVG overlay
    const svgLayer = L.svg().addTo(map);
    const overlaySvg = d3.select(svgLayer._container as SVGSVGElement);
    const gRoot = overlaySvg.append("g").attr("class", "d3-layer");

    leafletRef.current = map;
    svgLayerRef.current = svgLayer;
    overlaySvgRef.current = overlaySvg;
    gRootRef.current = gRoot;

    // reproject on pan/zoom

    const onMoveStart = () => {
      wasDraggedRef.current = true;
    };
    const onMoveEnd = () => {
      // Do NOT reset here.
      // We want the next click to see that a drag happened.
    };

    map.on("movestart", onMoveStart);
    map.on("moveend", onMoveEnd);
    map.on("zoomstart", onMoveStart);
    map.on("zoomend", onMoveEnd);

    const onMove = () => redrawAll();
    map.on("zoom viewreset moveend", onMove);

    leafletRef.current = map;

    return () => {
      try {
        map.off("zoom viewreset move", onMove);
        map.off("movestart", onMoveStart);
        map.off("moveend", onMoveEnd);
        map.off("zoomstart", onMoveStart);
        map.off("zoomend", onMoveEnd);
        clearAllSvgLayers();
        // remove the appended root
        gRootRef.current?.remove();
        gRootRef.current = null;
        overlaySvgRef.current = null;

        if (svgLayerRef.current) {
          map.removeLayer(svgLayerRef.current);
          svgLayerRef.current = null;
        }
        map.remove();
      } catch {
        /* ignore */
      } finally {
        leafletRef.current = null;
      }
    };
  }, [data?.center, clearAllSvgLayers, redrawAll]);

  useEffect(() => {
    if (baseLayerRef.current) {
      baseLayerRef.current.setOpacity(showBasemap ? 0.5 : 0);
    }
  }, [showBasemap]);

  // Keep map sized
  useEffect(() => {
    if (!leafletRef.current) return;
    const observer = new ResizeObserver(() =>
      leafletRef.current?.invalidateSize()
    );
    if (mapRef.current) observer.observe(mapRef.current);
    return () => observer.disconnect();
  }, []);

  // Auto-run when `data` changes (any change)
  useEffect(() => {
    if (!leafletRef.current) return;

    const ctrl = new AbortController();
    (async () => {
      try {
        await loadFromView(data);
      } catch (e: any) {
        if (e?.name !== "AbortError") console.error(e);
      }
    })();

    return () => ctrl.abort();
  }, [data, loadFromView]);

  const onClose = useCallback(() => {
    loadFromView(undefined); // clear before closing

    if (data?.onClose) return data.onClose(id);
    rf.setNodes((nds) => nds.filter((n) => n.id !== id));

    const curEdges = getEdges();

    // All targets currently connected FROM this view node
    const targetIds = curEdges
      .filter((e) => e.source === id)
      .map((e) => e.target);

    // Since we are not using the transformation node anymore, this logic here stays empty for now!!

    // 3) Remove all edges touching the closed view node
    setEdges((eds) => eds.filter((e) => e.source !== id && e.target !== id));
  }, [data, id, rf, getEdges, setEdges, loadFromView]);

  const onRun = useCallback(() => {
    if (data?.onRun) return data.onRun(id);
    // manual refresh if needed
    // console.log(data);
    loadFromView(data);
  }, [data, loadFromView, id]);

  return (
    <div className="vpnode">
      <NodeResizer minWidth={300} minHeight={260} />

      <div className="vpnode__header">
        <div className="vpnode__titleWrapper">
          <input
            type="text"
            className="pcenode__titleInput"
            value={data?.title ?? "View"}
            onChange={handleTitleChange}
          />
        </div>
        <button
          type="button"
          className="vpnode__iconBtn vpnode__iconBtn--close"
          onClick={onClose}
        >
          ✕
        </button>
      </div>

      <div className="vpnode__body">
        <div
          ref={mapRef}
          className="vpnode__map nodrag nowheel"
          aria-label={`Leaflet map for ${id}`}
          onPointerDown={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
          onTouchStart={(e) => e.stopPropagation()}
        />
      </div>

      <div className="vpnode__footer">
        <button
          type="button"
          onClick={onRun}
          title="update"
          aria-label="update"
          className="vpnode__actionBtn"
        >
          <img src={restartPng} alt="update" className="vpnode__actionIcon" />
        </button>
        <button
          type="button"
          onClick={onPersist}
          title="Save edits"
          aria-label="Save edits"
          className="vpnode__actionBtn"
          disabled={persisting}
        >
          {persisting ? (
            <span className="vpnode__spinner" />
          ) : persistSuccess ? (
            <img src={checkPng} alt="Success" className="vpnode__actionIcon" />
          ) : (
            <img
              src={persistPng}
              alt="Save edits"
              className="vpnode__actionIcon"
            />
          )}
        </button>

        <button
          type="button"
          onClick={() => setShowBasemap((b) => !b)}
          title="update"
          aria-label="update"
          className="vpnode__actionBtn"
        >
          <img src={mapPng} alt="update" className="vpnode__actionIcon" />
        </button>
      </div>

      <Handle
        type="target"
        position={Position.Left}
        id="viewport-in-1"
        className="vpnode__handle vpnode__handle--left"
      />

      <Handle
        type="target"
        position={Position.Top}
        id="viewport-in-3"
        className="vpnode__handle "
      />

      <Handle
        type="source"
        position={Position.Right}
        id="viewport-out"
        className="vpnode__handle vpnode__handle--right"
      />

      <Handle
        type="target"
        position={Position.Bottom}
        id="viewport-in-2"
        className="vpnode__handle vpnode__handle--bottom"
      />
    </div>
  );
});

export default ViewportNode;
