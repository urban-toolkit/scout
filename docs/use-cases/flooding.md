# Flooding Analysis

This use case demonstrates how SCOUT supports flood-risk analysis by comparing a baseline scenario with an intervention scenario incorporating nature-based solutions (NbS). The workflow begins by configuring scenarios through `widget` nodes, where the baseline is defined with no NbS and the intervention activates multiple solutions. Additional `widget` nodes specify shared parameters such as the projection timeline and region of interest, ensuring consistent inputs across scenarios. These parameters are passed to `computation` nodes, which access precomputed flood simulations and generate scenario-specific flood-depth outputs. The resulting flood-depth rasters are visualized in separate `view` nodes, enabling spatial comparison of projected flood patterns. A derived difference layer highlights areas of change between two scenarios and is visualized through an additional `view`. Finally, a `comparison` node displays the difference between two scenarios for median flood depth, allowing users to evaluate the effectiveness of nature-based solutions across scenarios.

![image](images/teaser_scout.png)

### Open in live application

[Live use case]()
