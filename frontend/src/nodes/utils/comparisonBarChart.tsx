import * as d3 from "d3";
import React, { useEffect, useRef, useState } from "react";

export function ComparisonBarChart({
  values,
  metric,
  unit,
}: {
  values: Record<string, number>;
  metric: string;
  unit: string;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const [width, setWidth] = useState(400);
  const height = 220;

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setWidth(entry.contentRect.width);
      }
    });

    resizeObserver.observe(el);
    return () => resizeObserver.disconnect();
  }, []);

  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return;

    const entries = Object.entries(values);

    d3.select(svgEl).selectAll("*").remove();

    const margin = { top: 20, right: 0, bottom: 50, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3.select(svgEl).attr("width", width).attr("height", height);

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleBand<string>()
      .domain(entries.map(([k]) => k))
      .range([0, innerWidth])
      .padding(0.35);

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(entries, ([, v]) => v) ?? 0])
      .nice()
      .range([innerHeight, 0]);

    // Bars
    g.selectAll("rect")
      .data(entries)
      .enter()
      .append("rect")
      .attr("x", ([k]) => x(k) ?? 0)
      .attr("y", ([, v]) => y(v))
      .attr("width", x.bandwidth())
      .attr("height", ([, v]) => innerHeight - y(v))
      .attr("fill", "#3182bd")
      .attr("rx", 4)
      .attr("ry", 4);

    // X axis
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .style("font-size", "15px")
      .style("font-family", "Inter, sans-serif");

    // X label
    svg
      .append("text")
      .attr("x", width / 2 + 30)
      .attr("y", height - 10)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-family", "Inter, sans-serif")
      .text("Scenario");

    // Y axis
    g.append("g")
      .call(d3.axisLeft(y).ticks(3))
      .selectAll("text")
      .style("font-size", "15px")
      .style("font-family", "Inter, sans-serif");

    // 💡 LEFT-SIDE VERTICAL METRIC LABEL
    svg
      .append("text")
      .attr("transform", `rotate(-90)`)
      .attr("x", -height / 2)
      .attr("y", 10) // closer to axis; adjust if needed
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-family", "Inter, sans-serif")
      .text(`${metric} (${unit})`);
  }, [values, metric, width, unit]);

  return (
    <div ref={containerRef} style={{ width: "100%", overflow: "hidden" }}>
      <svg ref={svgRef} />
    </div>
  );
}
