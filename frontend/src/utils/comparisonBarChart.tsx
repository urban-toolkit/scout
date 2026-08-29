import * as d3 from "d3";
import { useEffect, useRef, useState } from "react";

export function ComparisonBarChart({
  values,
  axis,
  axisLabel,
  props,
}: {
  values: Record<string, number>;
  axis: "x" | "y";
  axisLabel: string;
  props?: {
    unit?: string;
    labelY?: string;
    color?: string | string[] | Record<string, string>;
    colors?: string[] | Record<string, string>;
    [key: string]: any;
  };
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const [width, setWidth] = useState(400);
  const height = 220;
  const entries = Object.entries(values);
  const isHorizontal = axis === "x";
  const colorProp = props?.color ?? props?.colors;
  const barColors = entries.map(([label], index) => {
    if (typeof colorProp === "string") {
      return colorProp;
    }

    if (Array.isArray(colorProp) && colorProp.length > 0) {
      return colorProp[index % colorProp.length];
    }

    if (colorProp && typeof colorProp === "object" && !Array.isArray(colorProp)) {
      return colorProp[label] ?? d3.schemeTableau10[index % d3.schemeTableau10.length];
    }

    return d3.schemeTableau10[index % d3.schemeTableau10.length];
  });

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

    d3.select(svgEl).selectAll("*").remove();

    const margin = isHorizontal
      ? { top: 20, right: 10, bottom: 50, left: 90 }
      : { top: 20, right: 0, bottom: 50, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3.select(svgEl).attr("width", width).attr("height", height);

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    if (isHorizontal) {
      const x = d3
        .scaleLinear()
        .domain([0, d3.max(entries, ([, v]) => v) ?? 0])
        .nice()
        .range([0, innerWidth]);

      const y = d3
        .scaleBand<string>()
        .domain(entries.map(([k]) => k))
        .range([0, innerHeight])
        .padding(0.35);

      g.selectAll("rect")
        .data(entries)
        .enter()
        .append("rect")
        .attr("x", 0)
        .attr("y", ([k]) => y(k) ?? 0)
        .attr("width", ([, v]) => x(v))
        .attr("height", y.bandwidth())
        .attr("fill", (_d, i) => barColors[i])
        .attr("rx", 4)
        .attr("ry", 4);

      g.append("g")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x).ticks(3))
        .selectAll("text")
        .style("font-size", "15px")
        .style("font-family", "Inter, sans-serif");

      g.append("g")
        .call(d3.axisLeft(y))
        .selectAll("text")
        .style("font-size", "15px")
        .style("font-family", "Inter, sans-serif");

      svg
        .append("text")
        .attr("x", width / 2 + 30)
        .attr("y", height - 10)
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .style("font-family", "Inter, sans-serif")
        .text(props?.labelX ?? `${axisLabel} ${props?.unit ?? ""}`.trim());

      g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", -70)
        .attr("x", -height / 2.5)
        .attr("text-anchor", "middle")
        .style("font-size", "15px")
        .style("font-family", "Inter, sans-serif")
        .text("Scenario");

      return;
    }

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
      .attr("fill", (_d, i) => barColors[i])
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

    // Y-axis title
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -60)
      .attr("x", -height / 2.5)
      .attr("text-anchor", "middle")
      .style("font-size", "15px")
      .style("font-family", "Inter, sans-serif")
      .text(props?.labelY ?? `${axisLabel} ${props?.unit ?? ""}`.trim());
  }, [entries, axisLabel, width, props, barColors, isHorizontal]);

  return (
    <div ref={containerRef} style={{ width: "100%", overflow: "hidden" }}>
      <svg ref={svgRef} />
    </div>
  );
}
