// utils/comparisonPieChart.tsx
import * as d3 from "d3";
import { useEffect, useRef, useState } from "react";

export function ComparisonPieChart({
  values,
  metric,
  props,
}: {
  values: Record<string, number>;
  metric: string;
  props?: Record<string, any>;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const [width, setWidth] = useState(260);
  const height = 220; // reduced so there's less vertical space

  const entries = Object.entries(values);
  const labels = entries.map(([k]) => k);

  const color = d3
    .scaleOrdinal<string, string>()
    .domain(labels)
    .range(d3.schemeTableau10);

  // Responsive observer
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setWidth(entry.contentRect.width);
      }
    });

    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Draw chart
  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl || entries.length === 0) return;

    d3.select(svgEl).selectAll("*").remove();

    const svg = d3.select(svgEl).attr("width", width).attr("height", height);

    const margin = 10;
    const innerWidth = width - margin * 2;
    const innerHeight = height - margin * 2;

    // Use full innerHeight for radius, minus a tiny padding
    const radius = Math.min(innerWidth, innerHeight) / 2;

    // Center the pie; slightly shift up to leave a hair of space below
    const g = svg
      .append("g")
      .attr("transform", `translate(${width / 2}, ${height / 2 + 6})`);

    const pie = d3
      .pie<[string, number]>()
      .value(([, v]) => v)
      .sort(null);

    const arc = d3
      .arc<d3.PieArcDatum<[string, number]>>()
      .innerRadius(radius * 0.45)
      .outerRadius(radius);

    const arcs = pie(entries);

    // Slices
    g.selectAll("path")
      .data(arcs)
      .enter()
      .append("path")
      .attr("d", arc as any)
      .attr("fill", (d) => color(d.data[0]))
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.2);

    // Value labels (2 decimals, white, bigger, bold)
    g.selectAll("text.slice-label")
      .data(arcs)
      .enter()
      .append("text")
      .attr("class", "slice-label")
      .attr("transform", (d) => `translate(${arc.centroid(d)})`)
      .attr("text-anchor", "middle")
      .style("font-family", "Inter, sans-serif")
      .style("font-size", "15px")
      // .style("font-weight", "600")
      .style("fill", "#ffffff")
      .text((d) => d.data[1].toFixed(2));
  }, [entries, color, width, height]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        gap: 6,
      }}
    >
      {/* Centered legend row: Scenario + legend items together */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          gap: 16,
          padding: "0 6px",
          flexWrap: "wrap",
        }}
      >
        {/* Scenario label */}
        <div
          style={{
            fontSize: 16,
            fontFamily: "Inter, sans-serif",
            fontWeight: 500,
          }}
        >
          Scenario
        </div>

        {/* Legend items */}
        <div
          style={{
            display: "flex",
            gap: 14,
            flexWrap: "wrap",
            alignItems: "center",
            fontFamily: "Inter, sans-serif",
            fontSize: 16,
            fontWeight: 500,
          }}
        >
          {entries.map(([label]) => (
            <div
              key={label}
              style={{ display: "flex", alignItems: "center", gap: 6 }}
            >
              <span
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: 2,
                  backgroundColor: color(label),
                }}
              />
              <span>{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Pie chart */}
      <svg ref={svgRef} style={{ display: "block", margin: "0 auto" }} />

      {/* Metric label below chart */}
      <div
        style={{
          marginTop: 4,
          textAlign: "center",
          fontSize: 16,
          fontFamily: "Inter, sans-serif",
          fontWeight: 500,
        }}
      >
        {metric} ({props?.unit ?? ""})
      </div>
    </div>
  );
}
