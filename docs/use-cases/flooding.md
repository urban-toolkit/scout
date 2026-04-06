# Flooding Analysis

The flooding analysis use case demonstrates how SCOUT can support climate resilience planning through side-by-side scenario comparison. Users define a study area with top-left and bottom-right coordinate inputs, choose a projection timeline, and configure two scenarios using checkbox widgets for nature-based solutions.

Each scenario is passed to a flood simulation model that generates a projected flood output for the selected region and timeline. In the example workflow, one scenario can represent a baseline or limited-intervention case, while the other can include a collection of nature-based solutions such as bioswales, infiltration trenches, permeable pavements, retention ponds, and constructed wetlands.

SCOUT renders each flood projection as a map layer using a blue colormap, then creates a comparison view between the two scenario outputs. A comparison node summarizes the median flood depth in meters, giving users both a spatial view of where flood depth changes and a quantitative measure of how the intervention scenario differs from the baseline.

This use case highlights SCOUT's role as a decision-support environment for evaluating flood mitigation strategies, connecting stakeholder-controlled scenario parameters to model results, visual outputs, and summary metrics.
