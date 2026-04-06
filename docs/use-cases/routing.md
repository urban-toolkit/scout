# Routing Analysis

The routing analysis use case demonstrates how SCOUT can support transportation decisions when route choice depends on weather and user priorities. The workflow begins by fetching a road network for a Chicago region from OpenStreetMap. Users then provide an origin and destination through location widgets, choose a routing mode, set the number of alternate routes, and select a start time.

Weather-related preferences are exposed through rain and wind weight sliders. These widget values are passed into a weather-routing model, which generates alternate route outputs for the selected trip. This setup lets users explore how changes in environmental conditions or routing priorities can produce different mobility scenarios without directly editing the model code.

The generated routes are rendered together in a map view with distinct styling for each route, along with origin and destination markers. SCOUT also creates comparison charts for travel time, distance, rain exposure, and wind exposure, allowing users to evaluate the trade-offs between alternatives.

This use case highlights SCOUT's ability to combine physical network data, parameterized model execution, spatial visualization, and multi-metric comparison for transportation planning and scenario exploration.
