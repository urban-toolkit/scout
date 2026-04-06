# SCOUT Grammar

This page describes the core grammar components used in **SCOUT**. The grammar is organized into three groups:

- **Intelligence**: `data_layer`, `join`
- **Design**: `view`
- **Choice**: `interaction`, `widget`, `comparison`

---

## `data_layer`

A `data_layer` represents an urban dataset or model output that other nodes can render, transform, interact with, or compare.

#### Structure

```txt
data_layer := (id, source, feature, roi, attributes?, filters?)
```

#### Fields

- `id`: Unique id of the data layer that downstream nodes can reference.
- `source`: Where the data comes from, such as OpenStreetMap or a local file.
- `feature`: Type of data layer such as `buildings`, `streets`, `parks`, `water`.
- `roi`: The region of interest.
- `attributes`: Fields to use for each `feature`, such as building height, or road speed.
- `filters`: Conditions used to filter the data based on `attributes` of `feature`. e.g., _height >= 50m_.

#### Example

```json
{
  "data_layer": {
    "id": "A",
    "source": "osm",
    "dtype": "physical",
    "roi": {
      "datafile": "chicago",
      "type": "bbox",
      "value": [-87.66, 41.86, -87.64, 41.88]
    },
    "osm_features": [
      {
        "feature": "buildings",
        "attributes": ["height"]
      }
    ]
  }
}
```

#### Usage

A `data_layer` usually feeds into a `view` node for visualization, or a `join` node for transformation. SCOUT supports common geospatial formats such as GeoTIFF, GeoJSON, and Feather; the underlying format is inferred from filename extension, so authors do not need to annotate it in the grammar.

---

## `join`

A `join` combines two data layers and derive a new layer from their relationship.

#### Structure

```txt
join := (id, ref_left, ref_right, op, aggr)
```

#### Fields

- `id`: Name of the derived output from join operation.
- `ref_left`: Reference to the first input layer.
- `ref_right`: Reference to the second input layer.
- `op`: The relationship between the two layers, such as `intersects`, `contains`, `within`, `nearest`, or `overlaps`.
- `aggr`: How matched values are summarized, such as `count`, `sum`, `mean`, `min`, or `max`.

#### Example

```json
{
  "join": {
    "id": "A_buildings_mean_flood_depth",
    "ref_left": "A_buildings",
    "ref_right": "flood_depth",
    "op": "contains",
    "aggr": "mean"
  }
}
```

#### Usage

The output of a `join` can be rendered in a `view`, or sent into a `computation node`. For example, average flood depth that intersects per building's footprint can be visualized in a `view` or sent to a `computation node` for modeling purposes.

---

## `view`

A `view` defines how SCOUT renders data layers (either defined or derived).

- _Defined layers_ are created through `data_layer`.
- _Derived layers_ are produced through operations such as `join`, or as scenario outputs from _computation node_.
- A `view` can show a single layer, overlay multiple layers, or compare two scenarios such as baseline versus intervention.

#### Structure

```txt
view := (ref, style?)+ | (ref_base, ref_comp, style?)
```

#### Fields

- `ref`: Reference to a `data_layer` or a derived output to render.
- `ref_base`: Reference to the baseline scenario layer.
- `ref_comp`: Reference to the intervention scenario layer.
- `style`: Visual encoding properties such as fill, stroke, opacity, color scale, line width.

#### Single-Layer Example

<table>
  <tr>
    <td>
      <pre><code>{
  "view": [
    {
      "ref": "A_buildings",
      "style": {
        "fill": {
          "feature": "height",
          "range": [0, 550],
          "colormap": "blues"
        },
        "stroke-color": "#333333",
        "opacity": 1
      }
    }
  ]
}</code></pre>
    </td>
    <td><img src="images/buildings.png" width="400"/></td>
  </tr>
</table>

#### Multi-Layer Example

<table>
  <tr>
    <td>
      <pre><code>
{
  "view": [
    {
      "ref": "A_buildings",
      "style": {
        "fill": {
          "feature": "height",
          "range": [0, 550],
          "colormap": "blues"
        },
        "stroke-color": "#333333",
        "opacity": 1
      }
    },
    {
      "ref": "A_roads",
      "style": {
        "stroke": {
          "color": "green",
          "width": 1.5
        },
        "opacity": 1
      }
    }
  ]
}
</code></pre>
    </td>
    <td><img src="images/buildings_roads.png" width="400"/></td>
  </tr>
</table>

### Compare two scenarios Example

<table>
  <tr>
    <td>
      <pre><code>
{
  "view": [
    {
      "ref_base": "B",
      "ref_comp": "A",
      "style": {
        "opacity": 1,
        "colormap": "blues"
      }
    }
  ]
}
</code></pre>
    </td>
    <td><img src="images/flood_compare.png" width="400"/></td>
  </tr>
</table>

### Usage

A `view` usually consumes outputs from Intelligence nodes and reacts to Choice nodes. For example, a widget can change a floodwall height parameter, the model can produce a new flood layer, and the view can update to show the resulting scenario.

---

<table>
  <tr>
    <td bgcolor="#D2E4F0"><strong>Choice</strong></td>
  </tr>
</table>

## Interaction

An `interaction` defines how users directly manipulate a layer or scenario view.

Interactions are useful when scenario alternatives depend on geometry edits, feature selection, or attribute changes.

### Structure

```txt
interaction := (ref, itype, action, attribute?, condition?)
```

### Fields

- `ref`: Reference to the layer or view being manipulated.
- `itype`: The trigger, such as `click`, `draw`, `drag`, `select`, or `brush`.
- `action`: What happens after the trigger, such as `remove_feature`, `add_feature`, `edit_geometry`, or `update_attribute`.
- `attribute`: The field being changed when the interaction edits data values.
- `condition`: Optional condition that limits when the interaction is active.

### Example

```txt
interaction(
  ref: buildings_intervention,
  itype: click,
  action: remove_feature
)
```

```txt
interaction(
  ref: street_network,
  itype: select,
  action: update_attribute,
  attribute: speed_limit
)
```

### Usage

An `interaction` can feed edited layers into computation nodes and then into updated `view` or `comparison` outputs. For example, removing a high-rise building can trigger a shadow model and update sunlight-access comparison metrics.

## Widget

A `widget` exposes a model or scenario parameter as a user-facing control.

Widgets help technical authors turn complex dataflow pipelines into dashboards that non-technical users can operate.

### Structure

```txt
widget := (wtype, variable, choices?, default?, props?)
```

### Fields

- `wtype`: The control type, such as `radio`, `checkbox`, `dropdown`, `slider`, `number_input`, or `location_input`.
- `variable`: The scenario or model parameter controlled by the widget.
- `choices`: Available options for discrete or stepped inputs.
- `default`: The initial value.
- `props`: Display and validation options, such as label, units, min, max, or step.

### Example

```txt
widget(
  wtype: slider,
  variable: floodwall_height,
  choices: [0.5, 1.0, 1.5, 2.0],
  default: 1.5,
  props: { label: "Floodwall height", unit: "m" }
)
```

```txt
widget(
  wtype: checkbox,
  variable: nature_based_solutions,
  choices: [wetlands, green_roofs, retention_ponds],
  default: [wetlands]
)
```

```txt
widget(
  wtype: location_input,
  variable: region_of_interest,
  default: chicago_loop
)
```

### Usage

Widget values feed into computation nodes, model nodes, or filters. When a user changes a widget, SCOUT can rerun the relevant part of the pipeline and update views and comparisons.

## Comparison

A `comparison` defines how SCOUT summarizes differences across scenarios.

Comparisons make trade-offs visible by turning scenario outputs into charts, tables, or metrics that can support decisions.

### Structure

```txt
comparison := (x*, y*, key+, chart, props?)
```

### Fields

- `x`: The independent dimension, such as timeline, scenario, neighborhood, or intervention type.
- `y`: The measured outcome, such as average flood depth, shadow duration, travel time, or exposed population.
- `key`: Grouping fields used for comparison.
- `chart`: The representation, such as `bar`, `line`, `table`, `pie`, or `scatter`.
- `props`: Display options such as title, units, sorting, labels, or aggregation.

### Example

```txt
comparison(
  x: projection_timeline,
  y: avg_flood_depth,
  key: [scenario],
  chart: line,
  props: { title: "Projected flood depth by scenario", unit: "m" }
)
```

```txt
comparison(
  x: scenario,
  y: mean_shadow_duration,
  key: [park_area],
  chart: bar,
  props: { title: "Sunlight access comparison", unit: "hours" }
)
```

```txt
comparison(
  x: intervention,
  y: exposed_buildings,
  key: [neighborhood],
  chart: table
)
```

### Usage

A `comparison` usually sits downstream of model outputs, joined layers, or scenario views. It helps users decide which scenario performs better under the criteria they care about.
