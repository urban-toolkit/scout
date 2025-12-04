import numpy as np
import networkx as nx
import torch
from torch_geometric.nn import SAGEConv

import torch.nn.functional as F
import os
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np

from torch_geometric.data import Data
    


class NodeRegressor(torch.nn.Module):
    
    """
    Simple GraphSAGE-based regressor for node-level prediction of weather variables.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels=5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        out = self.lin(x)
        return out


def classic_weight_calculations(
    G, lats, lons, rain_ds, time, heat_ds, wind_speed_ds, wind_dir_ds, humidity_ds,
    rain_weight=0.85834, heat_weight=0.02850,
    wind_weight=0.09648, humidity_weight=0.01668
) -> None:
    """
    Apply weather-based penalties to edge weights in the graph according to
    
    This method iterates trough each edge and calculates weights based on weather data and zone.
    
    Default penalty lambda based calculated previously with entropy weights.
    rain_weight: 0.85834
    heat_weight: 0.02850
    wind_weight: 0.09648
    rh_weight: 0.01668
    
    """
    calculated_zones = []

    print("Calculating weights with classic method")

    for u, v, k, data in G.edges(keys=True, data=True):
        y1, x1 = G.nodes[u]['y'], G.nodes[u]['x']
        y2, x2 = G.nodes[v]['y'], G.nodes[v]['x']
        lat, lon = (y1 + y2) / 2, (x1 + x2) / 2

        t_uv = data.get("travel_time", 1)

        r_term = w_term = tau_term = h_term = 0


        # If rain is selected as a weight, calculate rain penalty
        if (rain_ds is not None) and (rain_weight is not None):
            rain = rain_ds.variables['RAIN'][time + G[u][v][k]["zone"], :, :]
            if G[u][v][k]["zone"] not in calculated_zones:
                calculated_zones.append(G[u][v][k]["zone"])
            rain_mm = get_element_at_point(lat, lon, rain, lats, lons)
            
            r_term = (rain_weight * rain_mm + t_uv * (1 - rain_weight))
            data['rain_weight'] = r_term

        # If heat index is selected as a weight, calculate heat penalty
        if (heat_ds is not None) and (heat_weight is not None):
            heat_index = heat_ds.variables['T2'][time + data['zone'] - 1, :, :]
            heat_at_point = get_element_at_point(lat, lon, heat_index, lats, lons)
            tau_term = (heat_weight * heat_at_point + t_uv * (1 - heat_weight))
            data['heat_weight'] = tau_term

        # If humidity is selected as a weight, calculate humidity penalty
        if (humidity_ds is not None) and (humidity_weight is not None):
            relative_humidity = humidity_ds.variables['RH2'][time + data['zone'] - 1, :, :]
            rh_at_point = get_element_at_point(lat, lon, relative_humidity, lats, lons)
            h_term = (humidity_weight * rh_at_point + t_uv * (1 - humidity_weight))
            data['humidity_weight'] = h_term

        # If wind is selected as a weight, calculate wind penalty
        if (wind_dir_ds is not None) and (wind_speed_ds is not None) and (wind_weight is not None):
            wind_speed = wind_speed_ds.variables['WSPD10'][time + data['zone'] - 1, :, :]
            wind_direction = wind_dir_ds.variables['WDIR10'][time + data['zone'] - 1, :, :]

            wind_spd_at_point = get_element_at_point(lat, lon, wind_speed, lats, lons)
            wind_dir_at_point = get_element_at_point(lat, lon, wind_direction, lats, lons)

            # This is faster than computing sin/cos each time
            sin_lut = np.sin(np.radians(np.arange(360)))
            cos_lut = np.cos(np.radians(np.arange(360)))

            crosswind = wind_spd_at_point * sin_lut[int(wind_dir_at_point) % 360]
            headwind = wind_spd_at_point * cos_lut[int(wind_dir_at_point) % 360]

            w_term = (wind_weight * (abs(crosswind) + max(0, headwind)) + t_uv * (1 - wind_weight))
            data['wind_weight'] = w_term

        
        data['total_weight'] = t_uv * (1 + r_term + w_term + tau_term + h_term)



def build_grid_kdtree(lats_arr, lons_arr):
    """Builds a k-d tree from flattened grid coordinates for fast nearest-neighbor lookup."""
    grid_points = np.vstack((lats_arr.flatten(), lons_arr.flatten())).T
    return cKDTree(grid_points)


def get_values_at_points_kdtree(grid_vals, kdtree, points_latlon):
    """Gets weather values from a grid for specific points using a pre-built k-d tree."""
    _, indices = kdtree.query(points_latlon, k=1)
    flat_vals = grid_vals.flatten()
    return flat_vals[indices]


def GNN_weight_calculations(G, 
                            lats, 
                            lons, 
                            rain_ds,
                            rain_data, 
                            time, 
                            heat_ds, 
                            heat_data, 
                            wind_speed_ds, 
                            wind_speed_data, 
                            wind_dir_ds, 
                            wind_dir_data, 
                            humidity_ds, 
                            humidity_data, 
                            trip_time_seconds, 
                            rain_weight= 0.85834, 
                            heat_weight=0.02850, 
                            wind_weight=0.09648, 
                            humidity_weight=0.01668) -> None:
    """
    Use a GNN to predict rain in any given node and use that vector to apply weights to the graph edges as a matrix operation.

    This modifies the weights for travel_time into a new variable "rain_weight", so that the routing algorithm can optimize for the fastest route given current weather events.

    Parameters:
    G (networkx.MultiDiGraph): The street graph.
    rain_ds (NC): 3D array of rain data (time, lat, lon).
    heat_ds (NC): 3D array of heat data (time, lat, lon).
    wind_speed_ds (NC): 3D array of wind speed data (time, lat, lon).
    wind_dir_ds (NC): 3D array of wind direction data (time, lat, lon).
    humidity_ds (NC): 3D array of humidity data (time, lat, lon).
    rain_data (np.ndarray): Raw rain data for training.
    heat_data (np.ndarray): Raw heat data for training.
    wind_speed_data (np.ndarray): Raw wind speed data for training.
    wind_dir_data (np.ndarray): Raw wind direction data for training.
    humidity_data (np.ndarray): Raw humidity data for training.
    time (int): Starting time index for weather data.
    trip_time_seconds (list): List of trip times in seconds for which to stitch datasets.
    lats (np.ndarray): 2D array of latitudes corresponding to the rain data
    lons (np.ndarray): 2D array of longitudes corresponding to the rain data
    rain_penalty_lambda (float): Scaling factor for rain penalty.

    Default penalty lambda based calculated previously with entropy weights.
    rain_weight: 0.85834
    heat_weight: 0.02850
    wind_weight: 0.09648
    rh_weight: 0.01668

    """

    print("Stitching datasets")
    rain_stitched = stitchDataset(G, rain_ds, starting_time=time, trip_time_seconds=trip_time_seconds, variable_name='RAIN')
    heat_stitched = stitchDataset(G, heat_ds, starting_time=time, trip_time_seconds=trip_time_seconds, variable_name='T2')
    wind_speed_stitched = stitchDataset(G, wind_speed_ds, starting_time=time, trip_time_seconds=trip_time_seconds, variable_name='WSPD10')
    wind_dir_stitched = stitchDataset(G, wind_dir_ds, starting_time=time, trip_time_seconds=trip_time_seconds, variable_name='WDIR10')
    humidity_stitched = stitchDataset(G, humidity_ds, starting_time=time, trip_time_seconds=trip_time_seconds, variable_name='RH2')
    print("Datasets Stitched")
    
    # Extract 2D spatial grids
    lats_raw = rain_ds.variables['XLAT'][:]
    lons_raw = rain_ds.variables['XLONG'][:]
    if lats_raw.ndim == 3:
        lats_2d = np.array(lats_raw[0, :, :])
        lons_2d = np.array(lons_raw[0, :, :])
    else:
        lats_2d = np.array(lats_raw)
        lons_2d = np.array(lons_raw)
    print("2D lats/lons extracted")
    node_ids = list(G.nodes)
    print("Preparing coords")
    coords = np.array([[G.nodes[n]['y'], G.nodes[n]['x']] for n in node_ids])  # lat, lon
    print("coords prepared")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Preparing node features and edges")
    # We build our variables to input to the GNN
    x, edge_index, node_ids = prepare_node_features_and_edges(
        G, rain_stitched, heat_stitched, wind_speed_stitched, wind_dir_stitched, humidity_stitched,
        lats_2d, lons_2d, coords
    )
    x = x.to(device)
    edge_index = edge_index.to(device)


    if os.path.exists("rain_model.pth"): # Pre trained model included in repo
        model = NodeRegressor(x.shape[1], hidden_channels=64).to(device)
        checkpoint = torch.load("rain_model.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained GNN model.")
    else:
        model = train_GNN_model(G, rain_data, heat_data, wind_speed_data, wind_dir_data, humidity_data, coords, node_ids, lons, lats)
        print("Trained new GNN model.")
        model = model.to(device)

    # Model makes predictions
    model.eval()
    with torch.no_grad():
        preds = model(x, edge_index).cpu().numpy()
    preds = np.nan_to_num(preds, nan=0.0)
   
    rain_pred_node = np.clip(preds[:, 0], 0, None)
    heat_pred_node = preds[:, 1]
    humidity_pred_node = np.clip(preds[:, 2], 0, 1)  
    wind_speed_pred_node = np.clip(preds[:, 3], 0, None)
    wind_dir_pred_node = preds[:, 4]  # allow negative for direction


    
    
    # Vectorized edge weight assignment
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    edges_list = list(G.edges(keys=True))
    u_idx = np.array([id_to_idx[u] for u, _, _ in edges_list])
    v_idx = np.array([id_to_idx[v] for _, v, _ in edges_list])

    # Base edge lengths
    travel_lengths = []
    for u, v, k in edges_list:
        data = G.get_edge_data(u, v)
        if isinstance(data, dict):
            k0 = list(data.keys())[0]
            d = data[k0]
        else:
            d = data
        travel_lengths.append(d.get("travel_time", 1.0))
    travel_lengths = np.array(travel_lengths, dtype=np.float32)

    # Create edge index
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    edges_list = list(G.edges(keys=True))
    u_idx = np.array([id_to_idx[u] for u, _, _ in edges_list])
    v_idx = np.array([id_to_idx[v] for _, v, _ in edges_list])

    travel_lengths = np.array([G[u][v][k].get("travel_time", 1.0) for u, v, k in edges_list], dtype=np.float32)

    # Get predicted weather values at edge target nodes
    rain_pred_at_v = rain_pred_node[v_idx]
    heat_pred_at_v = heat_pred_node[v_idx]
    humidity_pred_at_v = humidity_pred_node[v_idx]
    wind_speed_pred_at_v = wind_speed_pred_node[v_idx]
    wind_dir_pred_at_v = wind_dir_pred_node[v_idx]

    # Calculate all edge weights in a vectorized manner
    rain_w = abs((rain_pred_at_v * rain_weight) + (travel_lengths * (1 - rain_weight)))
    heat_w = abs((heat_pred_at_v * heat_weight) + (travel_lengths * (1 - heat_weight)))
    humidity_w = abs((humidity_pred_at_v * humidity_weight) + (travel_lengths * (1 - humidity_weight)))
    wind_w = abs((wind_speed_pred_at_v * wind_weight) + (travel_lengths * (1 - wind_weight)))
    wind_dir_w = (wind_dir_pred_at_v * wind_weight) + (travel_lengths * (1 - wind_weight))

    total_w = (
        (rain_weight * rain_pred_at_v) +
        (wind_weight * wind_speed_pred_at_v) +
        (heat_weight * heat_pred_at_v) +
        (humidity_weight * humidity_pred_at_v) +
        (travel_lengths * (1 - (rain_weight + wind_weight + heat_weight + humidity_weight)))
    )

    rain_weights_dict = dict(zip(edges_list, rain_w))
    heat_weights_dict = dict(zip(edges_list, heat_w))
    humidity_weights_dict = dict(zip(edges_list, humidity_w))
    wind_weights_dict = dict(zip(edges_list, wind_w))
    wind_dir_weights_dict = dict(zip(edges_list, wind_dir_w))
    total_weights_dict = dict(zip(edges_list, total_w))

    # Perform bulk assignment of the new weights to the graph edges
    nx.set_edge_attributes(G, rain_weights_dict, "rain_weight")
    nx.set_edge_attributes(G, heat_weights_dict, "heat_weight")
    nx.set_edge_attributes(G, humidity_weights_dict, "humidity_weight")
    nx.set_edge_attributes(G, wind_weights_dict, "wind_weight")
    nx.set_edge_attributes(G, wind_dir_weights_dict, "wind_dir_weight")
    nx.set_edge_attributes(G, total_weights_dict, "total_weight")
    
    print("Edge weights computed.")


def get_element_at_point(lat, lon, rain_grid, lats, lons):
    """Get the rain value at a specific lat/lon point from the rain grid."""
    i = np.argmin(np.abs(lats[:, 0] - lat))
    j = np.argmin(np.abs(lons[0, :] - lon))
    return rain_grid[i, j]


def train_GNN_model(G, rain_grid, heat_grid, wind_speed_grid, wind_dir_grid, humidity_grid, coords, node_ids, lons, lats, hidden_dim=64, lr=1e-3, epochs=300, out_channels=5):
    """
    Trains a GNN model to predict weather variables at graph nodes.
    When trained first time, the training data is the same as the input data.
    """
    
    
    grid_kdtree = build_grid_kdtree(lats, lons)

    rain_nodes = get_values_at_points_kdtree(rain_grid, grid_kdtree, coords)
    rain_nodes = np.nan_to_num(rain_nodes, nan=np.nanmedian(rain_nodes))
    
    heat_nodes = get_values_at_points_kdtree(heat_grid, grid_kdtree, coords)
    heat_nodes = np.nan_to_num(heat_nodes, nan=np.nanmedian(heat_nodes))
    
    wind_speed_nodes = get_values_at_points_kdtree(wind_speed_grid, grid_kdtree, coords)
    wind_speed_nodes = np.nan_to_num(wind_speed_nodes, nan=np.nanmedian(wind_speed_nodes))
    
    wind_dir_nodes = get_values_at_points_kdtree(wind_dir_grid, grid_kdtree, coords)
    wind_dir_nodes = np.nan_to_num(wind_dir_nodes, nan=np.nanmedian(wind_dir_nodes))
    
    humidity_nodes = get_values_at_points_kdtree(humidity_grid, grid_kdtree, coords)
    humidity_nodes = np.nan_to_num(humidity_nodes, nan=np.nanmedian(humidity_nodes))
    
    # Build edge_index
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    edges = []
    for u_n, v_n, data in G.edges(data=True):
        edges.append([id_to_idx[u_n], id_to_idx[v_n]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    if edge_index.numel() == 0:
        raise RuntimeError("Edge index is empty!")

    x_np = np.column_stack([
        coords[:, 0],  # lat
        coords[:, 1],  # lon
        rain_nodes,
        heat_nodes,
        wind_speed_nodes,
        wind_dir_nodes,
        humidity_nodes,
    ])
    x = torch.tensor(x_np, dtype=torch.float)

    y_np = np.column_stack([
        rain_nodes,
        heat_nodes,
        humidity_nodes,
        wind_speed_nodes,
        wind_dir_nodes,
    ])
    y = torch.tensor(y_np, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)

    # Simple train/val split: random nodes (since we treat each node as sample)
    num_nodes = data.num_nodes
    perm = np.random.permutation(num_nodes)
    train_n = int(0.8 * num_nodes)
    train_idx = torch.tensor(perm[:train_n], dtype=torch.long)
    val_idx = torch.tensor(perm[train_n:], dtype=torch.long)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    model = NodeRegressor(in_channels=data.num_node_features, hidden_channels=hidden_dim, out_channels=out_channels).to(device)
    data = data.to(device)
    in_ch = data.num_node_features
    out_ch = data.y.shape[1]  

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch==1:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = loss_fn(val_out[val_idx], data.y[val_idx])
            print(f"Epoch {epoch:04d} train_loss={loss.item():.6f} val_loss={val_loss.item():.6f}")
    
    model_path = "rain_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'in_channels': in_ch,
        'out_channels': out_ch,
    }, model_path)
    return model



def prepare_node_features_and_edges(G, rain_grid, heat_grid, wind_speed_grid, wind_dir_grid, humidity_grid, lats, lons, coords):

    grid_kdtree = build_grid_kdtree(lats, lons)
    
    node_ids = list(G.nodes)


    if rain_grid is not None: 
        rain_nodes = get_values_at_points_kdtree(rain_grid, grid_kdtree, coords)
        rain_nodes = np.nan_to_num(rain_nodes, nan=np.nanmedian(rain_nodes))
    else:
        rain_nodes = np.zeros(len(coords))

    if heat_grid is not None:
        heat_nodes = get_values_at_points_kdtree(heat_grid, grid_kdtree, coords)
        heat_nodes = np.nan_to_num(heat_nodes, nan=np.nanmedian(heat_nodes))
    else:
        heat_nodes = np.zeros(len(coords))

    if wind_speed_grid is not None:
        wind_speed_nodes = get_values_at_points_kdtree(wind_speed_grid, grid_kdtree, coords)
        wind_speed_nodes = np.nan_to_num(wind_speed_nodes, nan=np.nanmedian(wind_speed_nodes))
    else:
        wind_speed_nodes = np.zeros(len(coords))

    if wind_dir_grid is not None:
        wind_dir_nodes = get_values_at_points_kdtree(wind_dir_grid, grid_kdtree, coords)
        wind_dir_nodes = np.nan_to_num(wind_dir_nodes, nan=np.nanmedian(wind_dir_nodes))
    else:
        wind_dir_nodes = np.zeros(len(coords))

    if humidity_grid is not None:
        humidity_nodes = get_values_at_points_kdtree(humidity_grid, grid_kdtree, coords)
        humidity_nodes = np.nan_to_num(humidity_nodes, nan=np.nanmedian(humidity_nodes))
    else:
        humidity_nodes = np.zeros(len(coords))
    
    # Node features (lat, lon, and weather data)
    x_np = np.column_stack([
        coords[:, 0],
        coords[:, 1],
        rain_nodes,
        heat_nodes,
        wind_speed_nodes,
        wind_dir_nodes,
        humidity_nodes,
    ])
    x = torch.tensor(x_np, dtype=torch.float)

    # Build edge_index
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    edges = []
    for u_n, v_n, data in G.edges(data=True):
        edges.append([id_to_idx[u_n], id_to_idx[v_n]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        raise RuntimeError("Edge index is empty!")
    
    return x, edge_index, node_ids


def stitchDataset(G, dataset, trip_time_seconds, variable_name, starting_time=17):
    """Stitch together time slices from a dataset based on zones in the graph G."""
    import numpy as np
    
    # Handle empty trip times (e.g. start/end are same node)
    if not trip_time_seconds:
        # Just load the single starting time slice
        slice_data = np.array(dataset.variables[variable_name][starting_time, :, :])
        return slice_data
    
    # Preload time slices
    tdim = dataset.variables[variable_name].shape[0]
    loaded_data = []
    for t in range(len(trip_time_seconds)):
        idx = min(starting_time + t, tdim - 1)
        slice_data = np.array(dataset.variables[variable_name][idx, :, :])
        loaded_data.append(slice_data)

    zone_polygons = create_zone_polygons(G)

    # Extract 2D spatial grids
    lats_raw = dataset.variables['XLAT'][:]
    lons_raw = dataset.variables['XLONG'][:]
    
    if lats_raw.ndim == 3:
        lats = np.array(lats_raw[0, :, :])
        lons = np.array(lons_raw[0, :, :])
    else:
        lats = np.array(lats_raw)
        lons = np.array(lons_raw)

    stitched = np.zeros_like(loaded_data[0], dtype=np.float32)

    # Assign each zone its corresponding time slice
    for zone, polygon in zone_polygons.items():
        minx, miny, maxx, maxy = polygon.bounds
        zone_mask = (lats >= miny) & (lats <= maxy) & (lons >= minx) & (lons <= maxx)
        
        if not np.any(zone_mask):
            continue
        
        time_index = zone - 1
        if time_index < 0 or time_index >= len(loaded_data):
            continue
        
        stitched[zone_mask] = loaded_data[time_index][zone_mask]

    return stitched
    
def create_zone_polygons(G):
    """Create a convex hull polygon for each zone based on edge endpoints."""
    zones = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        zone = data.get("zone", 0)
        if zone == 0:
            continue
        zones.setdefault(zone, []).append(Point(G.nodes[u]["x"], G.nodes[u]["y"]))
        zones.setdefault(zone, []).append(Point(G.nodes[v]["x"], G.nodes[v]["y"]))

    zone_polygons = {}
    for zone, points in zones.items():
        unique_pts = list({(p.x, p.y): p for p in points}.values())
        if len(unique_pts) >= 3:
            zone_polygons[zone] = gpd.GeoSeries(unique_pts).unary_union.convex_hull
        elif len(unique_pts) == 2:
            zone_polygons[zone] = gpd.GeoSeries(unique_pts).unary_union.buffer(0.001)
        elif len(unique_pts) == 1:
            zone_polygons[zone] = unique_pts[0].buffer(0.001)
    return zone_polygons