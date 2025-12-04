import re
import geopandas as gpd
import pandas as pd
from geopy.geocoders import Nominatim
import osmium
import subprocess
import pickle, gzip
from pyrosm import OSM
import osmnx as ox
import networkx as nx
import os
os.environ['USE_PYGEOS'] = '0'

LEVEL_HEIGHT = 3.4

def download_osm_data(input_filename, location, output_filename):
    filename = 'data/%s.osm.pbf' % (output_filename)
    input = 'data/%s.osm.pbf' % (input_filename)
    
    geolocator = Nominatim(user_agent='uic')
    location = geolocator.geocode(location).raw

    south, north, west, east = map(float, location['boundingbox'])
    bbox = f"{west},{south},{east},{north}"

    print("Downloading OSM data for %s with bbox %s" % (location['display_name'], bbox))

    subprocess.run([
        "osmium", "extract",
        "-b", bbox,
        "-o", filename,
        "--overwrite",
        input
    ])

    print("Download complete. Data saved to %s" % (filename))

    return


def extract_roads(output_filename):
    pbf_path = 'data/%s.osm.pbf' % (output_filename)
    osm = OSM(pbf_path)

    nodes_gdf, edges_gdf = osm.get_network(
        network_type="driving",
        nodes=True,  
        extra_attributes=["maxspeed", "lanes", "name", "oneway"]
    )

    G = nx.MultiDiGraph()

    for _, row in nodes_gdf.iterrows():
        nid = int(row["id"])
        # OSMnx convention: x = lon, y = lat
        G.add_node(
            nid,
            x=float(row["lon"]),
            y=float(row["lat"]),
            # keep anything else if you care
            # timestamp=row["timestamp"],
            # visible=row["visible"],
        )

    for _, row in edges_gdf.iterrows():
        u = int(row["u"])
        v = int(row["v"])

        # copy all edge attributes except u,v
        data = row.to_dict()
        data.pop("u", None)
        data.pop("v", None)

        # pyrosm typically already gives 'length' (in meters) and 'geometry'
        G.add_edge(u, v, **data)

    G.graph["crs"] = "EPSG:4326"

    G = ox.add_edge_speeds(G)        # adds edge attribute 'speed_kph'
    G = ox.add_edge_travel_times(G)  # adds edge attribute 'travel_time' (seconds)
    G = ox.distance.add_edge_lengths(G)   # adds 'length' attribute in meters

    # convert to geodataframe
    edges = ox.convert.graph_to_gdfs(G, nodes=False, edges=True)
    edges = edges[['length', 'speed_kph', 'travel_time', 'geometry', 'width', 'name']]

    os.makedirs('data/%s' % (output_filename), exist_ok=True)
    edges.to_feather('data/%s/roads.feather' % (output_filename), compression='lz4')

    with gzip.open("./data/%s/roads.pkl.gz" % output_filename, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved road data to ./data/%s/roads.feather and ./data/%s/roads.pkl.gz" % (output_filename, output_filename))

    return

# https://wiki.openstreetmap.org/wiki/Simple_3D_buildings#Other_roof_tags

def _feet_to_meters(s):
    r = re.compile(r"([0-9]*\.?[0-9]+)'([0-9]*\.?[0-9]+)?\"?")
    m = r.findall(s)[0]
    if len(m[0]) > 0 and len(m[1]) > 0:
        m = float(m[0]) + float(m[1]) / 12.0
    elif len(m[0]) > 0:
        m = float(m[0])
    return m * 0.3048


def _get_height(tags):
    if 'height' in tags:
        # already accounts for roof
        if '\'' in tags['height'] or '\"' in tags['height']:
            return _feet_to_meters(tags['height'])
        r = re.compile(r"[-+]?\d*\.\d+|\d+")
        return float(r.findall(tags['height'])[0])
    if 'levels' in tags:
        roof_height = 0
        if 'roof_height' in tags:
            if '\'' in tags['roof_height'] or '\"' in tags['roof_height']:
                roof_height = _feet_to_meters(tags['roof_height'])
            else:
                r = re.compile(r"[-+]?\d*\.\d+|\d+")
                roof_height = float(r.findall(tags['roof_height'])[0])

        # does not account for roof height
        height = float(tags['levels']) * LEVEL_HEIGHT
        if 'roof_levels' in tags and roof_height == 0:
            height += float(tags['roof_levels']) * LEVEL_HEIGHT
        return height
    return 7.0


def _get_min_height(tags):
    if 'min_height' in tags:
        # already accounts for roof
        if '\'' in tags['min_height'] or '\"' in tags['min_height']:
            return _feet_to_meters(tags['min_height'])
        r = re.compile(r"[-+]?\d*\.\d+|\d+")
        return float(r.findall(tags['min_height'])[0])
    if 'min_level' in tags:
        height = float(tags['min_level']) * LEVEL_HEIGHT
        return height
    return 0.0


class BuildingHandler(osmium.SimpleHandler):

    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.geometry = []       # WKB bytes
        self.height = []
        self.min_height = []
        self.osm_id = []         # numeric id
        self.osm_type = []       # 'W' or 'R'
        self.wkbfab = osmium.geom.WKBFactory()

    def get_gdf(self):
        geom = gpd.GeoSeries.from_wkb(self.geometry, crs='EPSG:4326')
        gdf = gpd.GeoDataFrame({
            'osm_id': self.osm_id,
            'osm_type': self.osm_type,
            'min_height': pd.Series(self.min_height, dtype='float'),
            'height': pd.Series(self.height, dtype='float'),
            'geometry': geom
        }, index=geom.index)
        return gdf

    def area(self, a):
        id = int(a.orig_id())
        osm_type = 'W' if a.from_way() else 'R'

        tags = a.tags
        # Qualifiers
        if not ('building' in tags or 'building:part' in tags or tags.get('type', None) == 'building'):
            return
        # Disqualifiers
        if (tags.get('location', None) == 'underground' or 'bridge' in tags):
            return
        try:
            poly = self.wkbfab.create_multipolygon(a)
            height = _get_height(tags)
            min_height = _get_min_height(tags)

            self.geometry.append(poly)
            self.height.append(height)
            self.min_height.append(min_height)
            self.osm_id.append(id)
            self.osm_type.append(osm_type)
            
        except Exception as e:
            print(e)
            print(a)

def save_buildings_geojson(handler: BuildingHandler, out_path: str) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame (with osm_id, osm_type, height, min_height, geometry)
    and save it as GeoJSON. Returns the GeoDataFrame.
    """
    gdf = handler.get_gdf()
    gdf.to_file(out_path, driver="GeoJSON")
    return

def extract_buildings(output_filename):
    pbf_path = 'data/%s.osm.pbf' % (output_filename)
    handler = BuildingHandler()
    handler.apply_file(pbf_path, locations=True)

    gdf = handler.get_gdf()
    # Ensure directory exists
    os.makedirs('data/%s' % (output_filename), exist_ok=True)
    gdf.to_feather('data/%s/buildings.feather' % (output_filename), compression='lz4')

    print("Saved building data to ./data/%s/buildings.feather" % (output_filename))

    return