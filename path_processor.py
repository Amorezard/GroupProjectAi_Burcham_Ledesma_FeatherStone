import numpy as np
import cv2
import networkx as nx
import math
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString
import os

# Set matplotlib backend to non-interactive to avoid threading issues
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt

class PathProcessor:
    def __init__(self, campus_data):
        """
        Initialize path processor with campus data
        
        Args:
            campus_data (dict): The campus data dictionary containing buildings and paths
        """
        self.campus_data = campus_data
        self.graph = None
        self.osm_graph = None
        self.grid_size = 0.0001  # Approximately 10m grid size
        self.path_types = {
            'main_road': {'weight': 1.2, 'priority': 1},
            'sidewalk': {'weight': 1.0, 'priority': 2},
            'shortcut': {'weight': 0.8, 'priority': 3}, 
            'accessible': {'weight': 1.1, 'priority': 2},
            'stairs': {'weight': 1.5, 'priority': 4}
        }
        
    def detect_paths_from_image(self, image_path, lat_bounds, lng_bounds):
        """
        Detect paths from an aerial image using computer vision
        
        Args:
            image_path (str): Path to the aerial image
            lat_bounds (tuple): (min_lat, max_lat) bounds of the image
            lng_bounds (tuple): (min_lng, max_lng) bounds of the image
            
        Returns:
            list: Detected paths as nodes
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to identify pathways (assuming paths are lighter)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours of potential paths
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contour points to geographic coordinates
        paths = []
        for contour in contours:
            # Filter out small contours (noise)
            if cv2.contourArea(contour) < 100:
                continue
                
            path_nodes = []
            for i, point in enumerate(contour):
                # Skip points to reduce density
                if i % 10 != 0:
                    continue
                    
                x, y = point[0]
                
                # Convert pixel coordinates to geographic coordinates
                img_height, img_width = gray.shape
                lng = lng_bounds[0] + (x / img_width) * (lng_bounds[1] - lng_bounds[0])
                lat = lat_bounds[0] + ((img_height - y) / img_height) * (lat_bounds[1] - lat_bounds[0])
                
                path_nodes.append({
                    "id": f"detected_path_{len(paths)}_{i}",
                    "lat": lat,
                    "lng": lng
                })
                
            if len(path_nodes) >= 2:
                paths.append({
                    "from": path_nodes[0]["id"],
                    "to": path_nodes[-1]["id"],
                    "type": "detected",
                    "nodes": path_nodes
                })
                
        return paths
    
    def create_path_graph(self, include_existing=True, include_terrain=False):
        """
        Create a graph representing all paths on campus
        
        Args:
            include_existing (bool): Whether to include existing paths
            include_terrain (bool): Whether to include terrain data
            
        Returns:
            networkx.Graph: Graph representing campus paths
        """
        G = nx.Graph()
        
        # Add buildings as nodes
        for building in self.campus_data['buildings']:
            G.add_node(
                building['id'],
                pos=(building['lat'], building['lng']),
                type='building',
                name=building['name'],
                description=building.get('description', '')
            )
        
        # Add existing paths
        if include_existing:
            for path in self.campus_data['paths']:
                path_type = path.get('type', 'sidewalk')
                
                # Set edge weight based on path type
                weight_factor = self.path_types.get(path_type, {'weight': 1.0})['weight']
                
                prev_node = None
                for node in path['nodes']:
                    node_id = node['id']
                    
                    # Add node if it doesn't exist
                    if node_id not in G:
                        G.add_node(
                            node_id,
                            pos=(node['lat'], node['lng']),
                            type='path_node',
                            path_type=path_type
                        )
                    
                    if prev_node:
                        # Calculate actual distance
                        lat1, lng1 = G.nodes[prev_node]['pos']
                        lat2, lng2 = G.nodes[node_id]['pos']
                        distance = self.calculate_distance(lat1, lng1, lat2, lng2)
                        
                        # If terrain is included, adjust weight based on slope
                        if include_terrain and 'elevation' in node and 'elevation' in G.nodes[prev_node]:
                            # Calculate slope
                            elev_diff = abs(node['elevation'] - G.nodes[prev_node]['elevation'])
                            horiz_dist = distance
                            
                            # Calculate slope percentage
                            if horiz_dist > 0:
                                slope_pct = (elev_diff / horiz_dist) * 100
                                
                                # Non-linear penalty for steep slopes (square the slope percentage)
                                slope_factor = 1 + (slope_pct / 10)**2
                                weight_factor *= slope_factor
                        
                        # Add edge with appropriate weight
                        G.add_edge(
                            prev_node, 
                            node_id,
                            weight=distance * weight_factor,
                            distance=distance,
                            path_type=path_type
                        )
                    
                    prev_node = node_id
                
                # Connect first and last nodes to buildings if this is a building path
                if path.get('from') and path.get('to'):
                    first_node = path['nodes'][0]['id']
                    last_node = path['nodes'][-1]['id']
                    
                    if path['from'] in G.nodes:
                        from_building = path['from']
                        lat1, lng1 = G.nodes[from_building]['pos']
                        lat2, lng2 = G.nodes[first_node]['pos']
                        distance = self.calculate_distance(lat1, lng1, lat2, lng2)
                        
                        G.add_edge(
                            from_building, 
                            first_node,
                            weight=distance * weight_factor,
                            distance=distance,
                            path_type=path_type
                        )
                    
                    if path['to'] in G.nodes:
                        to_building = path['to']
                        lat1, lng1 = G.nodes[last_node]['pos']
                        lat2, lng2 = G.nodes[to_building]['pos']
                        distance = self.calculate_distance(lat1, lng1, lat2, lng2)
                        
                        G.add_edge(
                            last_node,
                            to_building,
                            weight=distance * weight_factor,
                            distance=distance,
                            path_type=path_type
                        )
        
        self.graph = G
        return G
    
    def analyze_path_types(self):
        """
        Analyze existing paths to detect their types based on properties
        
        Returns:
            dict: Updated paths with detected types
        """
        updated_paths = []
        
        for path in self.campus_data['paths']:
            # Start with default type if not already set
            if 'type' not in path:
                path_type = 'sidewalk'  # Default type
                
                # Check the name of from/to buildings to guess if it's a main road
                if path.get('from') and path.get('to'):
                    from_node = path['from']
                    to_node = path['to']
                    
                    # If connected to campus_entrance, likely a main road
                    if from_node == 'campus_entrance' or to_node == 'campus_entrance':
                        path_type = 'main_road'
                
                # Check the nodes for properties that might indicate a type
                if len(path['nodes']) >= 2:
                    # Calculate straight-line vs actual path length ratio
                    first_node = path['nodes'][0]
                    last_node = path['nodes'][-1]
                    
                    straight_dist = self.calculate_distance(
                        first_node['lat'], first_node['lng'],
                        last_node['lat'], last_node['lng']
                    )
                    
                    actual_dist = 0
                    for i in range(1, len(path['nodes'])):
                        prev = path['nodes'][i-1]
                        curr = path['nodes'][i]
                        actual_dist += self.calculate_distance(
                            prev['lat'], prev['lng'],
                            curr['lat'], curr['lng']
                        )
                    
                    ratio = straight_dist / actual_dist if actual_dist > 0 else 1
                    
                    # If the path is very direct (>0.95 ratio), it might be a shortcut
                    if ratio > 0.95:
                        path_type = 'shortcut'
                
                # Update path with detected type
                path['type'] = path_type
            
            updated_paths.append(path)
        
        return updated_paths
    
    def generate_grid_paths(self, resolution=10):
        """
        Generate a grid of paths covering the campus area
        
        Args:
            resolution (int): Number of grid cells in each dimension
            
        Returns:
            list: Generated grid paths
        """
        # Find bounds of campus
        min_lat = min(b['lat'] for b in self.campus_data['buildings'])
        max_lat = max(b['lat'] for b in self.campus_data['buildings'])
        min_lng = min(b['lng'] for b in self.campus_data['buildings'])
        max_lng = max(b['lng'] for b in self.campus_data['buildings'])
        
        # Add some padding
        padding = 0.001  # Approximately 100m
        min_lat -= padding
        max_lat += padding
        min_lng -= padding
        max_lng += padding
        
        # Create grid
        lat_step = (max_lat - min_lat) / resolution
        lng_step = (max_lng - min_lng) / resolution
        
        grid_paths = []
        
        # Create horizontal grid lines
        for i in range(resolution + 1):
            lat = min_lat + i * lat_step
            
            nodes = []
            for j in range(resolution + 1):
                lng = min_lng + j * lng_step
                node_id = f"grid_h_{i}_{j}"
                
                nodes.append({
                    "id": node_id,
                    "lat": lat,
                    "lng": lng
                })
            
            grid_paths.append({
                "type": "grid_connection",
                "nodes": nodes
            })
        
        # Create vertical grid lines
        for j in range(resolution + 1):
            lng = min_lng + j * lng_step
            
            nodes = []
            for i in range(resolution + 1):
                lat = min_lat + i * lat_step
                node_id = f"grid_v_{i}_{j}"
                
                nodes.append({
                    "id": node_id,
                    "lat": lat,
                    "lng": lng
                })
            
            grid_paths.append({
                "type": "grid_connection",
                "nodes": nodes
            })
        
        return grid_paths
    
    def prioritize_paths(self, paths):
        """
        Assign priorities to paths based on their properties
        
        Args:
            paths (list): List of paths to prioritize
            
        Returns:
            list: Paths with priority values
        """
        prioritized_paths = []
        
        for path in paths:
            # Start with the default priority from path type
            path_type = path.get('type', 'sidewalk')
            priority = self.path_types.get(path_type, {'priority': 2})['priority']
            
            # Adjust priority based on other factors
            
            # 1. Connection to important buildings increases priority
            important_buildings = ['sakowich', 'mcquade', 'campus_entrance']
            if path.get('from') in important_buildings or path.get('to') in important_buildings:
                priority -= 1  # Higher priority (lower number)
            
            # 2. Length - shorter paths get higher priority for efficiency
            if len(path['nodes']) >= 2:
                length = 0
                for i in range(1, len(path['nodes'])):
                    prev = path['nodes'][i-1]
                    curr = path['nodes'][i]
                    length += self.calculate_distance(
                        prev['lat'], prev['lng'],
                        curr['lat'], curr['lng']
                    )
                
                # Shorter paths get higher priority
                if length < 100:  # Less than 100m
                    priority -= 0.5
                elif length > 500:  # More than 500m
                    priority += 0.5
            
            # Add priority to path
            path_copy = path.copy()
            path_copy['priority'] = priority
            prioritized_paths.append(path_copy)
        
        # Sort by priority (lower is better)
        prioritized_paths.sort(key=lambda x: x.get('priority', 99))
        
        return prioritized_paths
    
    def identify_missing_connections(self):
        """
        Identify missing connections between buildings/areas
        
        Returns:
            list: Suggested new paths to add
        """
        if not self.graph:
            self.create_path_graph()
        
        G = self.graph
        
        # Create proximity graph for all buildings
        buildings = [(b['id'], b['lat'], b['lng']) for b in self.campus_data['buildings']]
        
        suggested_paths = []
        
        # For each building, check paths to nearby buildings
        for i, (building_id, lat, lng) in enumerate(buildings):
            # Find closest buildings
            nearby = []
            for j, (other_id, other_lat, other_lng) in enumerate(buildings):
                if building_id != other_id:
                    distance = self.calculate_distance(lat, lng, other_lat, other_lng)
                    if distance < 200:  # Buildings within 200m
                        nearby.append((other_id, distance))
            
            # Sort by distance
            nearby.sort(key=lambda x: x[1])
            
            # Check for the closest 3 buildings
            for other_id, distance in nearby[:3]:
                # Calculate shortest path distance using current graph
                try:
                    path = nx.shortest_path(G, building_id, other_id, weight='weight')
                    path_length = 0
                    for k in range(len(path) - 1):
                        u, v = path[k], path[k + 1]
                        path_length += G[u][v].get('distance', 0)
                    
                    # If path is significantly longer than direct distance, suggest new path
                    if path_length > distance * 1.5:
                        # Get building positions
                        b1_pos = next(b for b in self.campus_data['buildings'] if b['id'] == building_id)
                        b2_pos = next(b for b in self.campus_data['buildings'] if b['id'] == other_id)
                        
                        # Create a simple direct path
                        suggested_paths.append({
                            "from": building_id,
                            "to": other_id,
                            "type": "suggested_shortcut",
                            "nodes": [
                                {
                                    "id": f"sugg_{building_id}_{other_id}_1",
                                    "lat": b1_pos['lat'],
                                    "lng": b1_pos['lng']
                                },
                                {
                                    "id": f"sugg_{building_id}_{other_id}_2",
                                    "lat": (b1_pos['lat'] + b2_pos['lat']) / 2,
                                    "lng": (b1_pos['lng'] + b2_pos['lng']) / 2
                                },
                                {
                                    "id": f"sugg_{building_id}_{other_id}_3",
                                    "lat": b2_pos['lat'],
                                    "lng": b2_pos['lng']
                                }
                            ],
                            "description": f"Suggested shortcut between {building_id} and {other_id}"
                        })
                except nx.NetworkXNoPath:
                    # No path exists, definitely suggest new path
                    b1_pos = next(b for b in self.campus_data['buildings'] if b['id'] == building_id)
                    b2_pos = next(b for b in self.campus_data['buildings'] if b['id'] == other_id)
                    
                    suggested_paths.append({
                        "from": building_id,
                        "to": other_id,
                        "type": "suggested_connection",
                        "nodes": [
                            {
                                "id": f"sugg_{building_id}_{other_id}_1",
                                "lat": b1_pos['lat'],
                                "lng": b1_pos['lng']
                            },
                            {
                                "id": f"sugg_{building_id}_{other_id}_2",
                                "lat": (b1_pos['lat'] + b2_pos['lat']) / 2,
                                "lng": (b1_pos['lng'] + b2_pos['lng']) / 2
                            },
                            {
                                "id": f"sugg_{building_id}_{other_id}_3",
                                "lat": b2_pos['lat'],
                                "lng": b2_pos['lng']
                            }
                        ],
                        "description": f"Suggested connection between {building_id} and {other_id}"
                    })
        
        return suggested_paths
    
    def calculate_distance(self, lat1, lng1, lat2, lng2):
        """
        Calculate distance between two points in meters using the Haversine formula
        
        Args:
            lat1 (float): Latitude of first point
            lng1 (float): Longitude of first point
            lat2 (float): Latitude of second point
            lng2 (float): Longitude of second point
            
        Returns:
            float: Distance in meters
        """
        # Earth's radius in meters
        earth_radius = 6371000
        
        # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(lat1)
        lng1 = math.radians(lng1)
        lat2 = math.radians(lat2)
        lng2 = math.radians(lng2)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = earth_radius * c
        
        return distance
    
    def load_osm_data(self, north=None, south=None, east=None, west=None):
        """
        Load OpenStreetMap data for the campus area
        
        Args:
            north, south, east, west: Bounding box coordinates (if None, determined from campus data)
            
        Returns:
            networkx.MultiDiGraph: OSM graph for the campus area
        """
        # If bounding box not provided, use Merrimack College coordinates
        if north is None or south is None or east is None or west is None:
            # Merrimack College bounding box coordinates - EXPANDED to capture more paths
            north = 42.676500  # Increased from 42.674598
            south = 42.661500  # Decreased from 42.663485
            east = -71.111000  # Increased from -71.113261
            west = -71.129500  # Decreased from -71.127175
            
            # Only calculate from campus data if explicitly requested by setting all params to None
            if all(param is None for param in [north, south, east, west]):
                # Get all building coordinates
                lats = [b['lat'] for b in self.campus_data['buildings']]
                lngs = [b['lng'] for b in self.campus_data['buildings']]
                
                # Add padding to ensure we capture the entire campus
                padding = 0.002  # Increased from 0.001 for better coverage
                north = max(lats) + padding
                south = min(lats) - padding
                east = max(lngs) + padding
                west = min(lngs) - padding
        
        try:
            # Configure OSMnx with more aggressive settings
            import osmnx.settings as ox_settings
            ox_settings.max_query_area_size = 50000000  # 50 sq km, much larger to prevent subdivision
            ox_settings.timeout = 300  # Increased from 180 to 300 seconds
            ox_settings.useful_tags_way = ['highway', 'name', 'footway', 'surface', 'path', 'foot', 'service', 'access']  # Added more tags
            ox_settings.log_console = True  # Show progress in console
            
            print(f"Downloading OSM data for area: N:{north}, S:{south}, E:{east}, W:{west}")
            
            # Improved filter to capture more path types - especially footpaths and trails
            custom_filter = (
                '["highway"]["area"!~"yes"]'
                '["highway"~"^(primary|secondary|tertiary|residential|service|footway|path|pedestrian|track|cycleway|steps|corridor|bridleway)$"]'
                '["access"!~"private"]'
            )
            
            # Try different approaches to get more comprehensive data
            try:
                # First try polygon-based approach - better for irregular campus shapes
                from shapely.geometry import Polygon
                
                # Create a polygon boundary around the campus with buffer
                campus_poly = Polygon([
                    (west, north), (east, north), 
                    (east, south), (west, south)
                ])
                
                # Download using polygon boundary
                print(f"Attempting polygon-based download")
                self.osm_graph = ox.graph_from_polygon(
                    campus_poly,
                    network_type='all',  # Get all network types, not just 'walk'
                    simplify=True,
                    custom_filter=custom_filter
                )
            except Exception as e:
                print(f"Polygon-based approach failed: {e}, trying point-based approach...")
                
                # Try with a point and distance approach
                center_lat = (north + south) / 2
                center_lng = (east + west) / 2
                
                # Calculate distance in meters
                from math import radians, cos, sin, asin, sqrt
                def haversine(lat1, lon1, lat2, lon2):
                    # Convert decimal degrees to radians
                    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                    # Haversine formula
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * asin(sqrt(a))
                    r = 6371000  # Radius of earth in meters
                    return c * r
                
                # Find distance from center to farthest corner
                distances = [
                    haversine(center_lat, center_lng, north, east),
                    haversine(center_lat, center_lng, north, west),
                    haversine(center_lat, center_lng, south, east),
                    haversine(center_lat, center_lng, south, west)
                ]
                max_dist = max(distances) * 1.1  # Add 10% buffer
                
                try:
                    print(f"Attempting point-based download from ({center_lat}, {center_lng}) with {max_dist:.1f}m radius")
                    self.osm_graph = ox.graph_from_point(
                        (center_lat, center_lng), 
                        dist=max_dist,
                        network_type='all',  # Get all network types
                        simplify=True,
                        custom_filter=custom_filter
                    )
                except Exception as e:
                    print(f"Point-based approach failed: {e}, trying bbox approach...")
                    # Fall back to bbox approach with all network types
                    self.osm_graph = ox.graph_from_bbox(
                        north=north, south=south, east=east, west=west,
                        network_type='all',  # Get all network types, not just walk
                        simplify=True,
                        truncate_by_edge=True,
                        clean_periphery=True,
                        custom_filter=custom_filter
                    )
            
            # Convert to undirected graph for pathfinding
            # In OSMnx 2.0.2, to_undirected is in the convert module
            from osmnx.convert import to_undirected
            self.osm_graph = to_undirected(self.osm_graph)
            
            # Remove isolated nodes and self-loops
            self.osm_graph = ox.utils_graph.remove_isolated_nodes(self.osm_graph)
            self.osm_graph = nx.classes.filters.selfloop_edges(self.osm_graph)
            
            print(f"Successfully downloaded OSM graph with {len(self.osm_graph.nodes)} nodes and {len(self.osm_graph.edges)} edges")
            
            # Save a visualization of the graph for debugging
            try:
                self.visualize_osm_graph(save_path="static/osm_graph.png")
            except Exception as vis_e:
                print(f"Unable to generate visualization: {vis_e}")
            
            # After getting OSM data, enhance by detecting paths from aerial imagery
            try:
                self.augment_osm_with_aerial_imagery()
            except Exception as img_e:
                print(f"Unable to augment OSM data with aerial imagery: {img_e}")
            
            return self.osm_graph
            
        except Exception as e:
            print(f"Error loading OSM data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def augment_osm_with_aerial_imagery(self):
        """
        Augment OSM data with paths detected from aerial imagery
        """
        try:
            # Check if we have aerial imagery of campus
            import os
            
            # First, try to load from a predefined aerial image
            aerial_image_path = 'static/merrimack_aerial.jpg'
            if not os.path.exists(aerial_image_path):
                # If no aerial image exists, download it
                self.download_aerial_imagery(aerial_image_path)
            
            if os.path.exists(aerial_image_path):
                # Define bounds for the aerial image
                north, south = 42.676500, 42.661500
                east, west = -71.111000, -71.129500
                lat_bounds = (south, north)
                lng_bounds = (west, east)
                
                # Detect paths from aerial image
                detected_paths = self.detect_paths_from_image(aerial_image_path, lat_bounds, lng_bounds)
                
                # Convert detected paths to graph nodes and edges
                self.add_detected_paths_to_osm_graph(detected_paths)
                
                print(f"Added {len(detected_paths)} paths from aerial imagery")
            else:
                print("No aerial imagery available for path detection")
        
        except Exception as e:
            print(f"Error augmenting OSM data with aerial imagery: {e}")
            import traceback
            traceback.print_exc()
    
    def download_aerial_imagery(self, output_path):
        """
        Download aerial imagery of the campus area
        
        Args:
            output_path (str): Path to save the downloaded aerial image
        """
        try:
            # If we need to download aerial imagery, we can use various sources
            # For example, using the Google Static Maps API or similar service
            # This is a placeholder - in practice, you would use an API key and proper service
            
            import os
            import requests
            from io import BytesIO
            from PIL import Image
            
            # Define the campus area
            center_lat = (42.676500 + 42.661500) / 2
            center_lng = (-71.111000 + (-71.129500)) / 2
            
            # Check if the campus PDF map can be converted to an image
            pdf_path = "merrimack-college-map.pdf"
            if os.path.exists(pdf_path):
                try:
                    # Convert PDF to image
                    from pdf2image import convert_from_path
                    pages = convert_from_path(pdf_path, dpi=300)
                    if pages:
                        # Save the first page as aerial image
                        pages[0].save(output_path, 'JPEG')
                        print(f"Converted campus map PDF to image: {output_path}")
                        return
                except Exception as pdf_e:
                    print(f"Error converting PDF to image: {pdf_e}")
            
            # If PDF conversion failed, we could fallback to a web service
            # This is just placeholder code - real implementation would use proper API access
            print("No aerial imagery could be downloaded - place a high-resolution image in static/merrimack_aerial.jpg")
            
        except Exception as e:
            print(f"Error downloading aerial imagery: {e}")
    
    def detect_paths_from_image(self, image_path, lat_bounds, lng_bounds):
        """
        Enhanced path detection from an aerial image using computer vision
        
        Args:
            image_path (str): Path to the aerial image
            lat_bounds (tuple): (min_lat, max_lat) bounds of the image
            lng_bounds (tuple): (min_lng, max_lng) bounds of the image
            
        Returns:
            list: Detected paths as nodes
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create a copy for visualization
        debug_image = image.copy()
        
        # Apply adaptive thresholding to better handle lighting variations
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Apply edge detection
        edges = cv2.Canny(opening, 50, 150)
        
        # Dilate edges to connect nearby lines
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Use HoughLinesP to detect line segments (potential paths)
        lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
        
        # Collect detected paths
        paths = []
        
        if lines is not None:
            # Group lines into path segments
            path_segments = self.group_line_segments(lines, 10)  # 10 pixel threshold for grouping
            
            # Create paths from grouped segments
            for i, segment_group in enumerate(path_segments):
                path_nodes = []
                
                # Sort segments to form a continuous path
                ordered_segments = self.order_segments(segment_group)
                
                for j, segment in enumerate(ordered_segments):
                    # Convert pixel coordinates to geographic coordinates
                    img_height, img_width = gray.shape
                    min_lat, max_lat = lat_bounds
                    min_lng, max_lng = lng_bounds
                    
                    # Start point
                    x1, y1 = segment[0]
                    lng1 = min_lng + (x1 / img_width) * (max_lng - min_lng)
                    lat1 = max_lat - (y1 / img_height) * (max_lat - min_lat)
                    
                    # For the first segment, add the starting point
                    if j == 0:
                        path_nodes.append({
                            "id": f"detected_path_{i}_{j*2}",
                            "lat": lat1,
                            "lng": lng1
                        })
                    
                    # End point
                    x2, y2 = segment[1]
                    lng2 = min_lng + (x2 / img_width) * (max_lng - min_lng)
                    lat2 = max_lat - (y2 / img_height) * (max_lat - min_lat)
                    
                    path_nodes.append({
                        "id": f"detected_path_{i}_{j*2+1}",
                        "lat": lat2,
                        "lng": lng2
                    })
                    
                    # Draw the detected path on debug image
                    cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add path if it has at least 2 nodes
                if len(path_nodes) >= 2:
                    paths.append({
                        "from": path_nodes[0]["id"],
                        "to": path_nodes[-1]["id"],
                        "type": "detected",
                        "nodes": path_nodes
                    })
        
        # Save debug image
        cv2.imwrite('static/debug/path_detection.jpg', debug_image)
        
        return paths
    
    def group_line_segments(self, lines, threshold):
        """
        Group line segments that may belong to the same path
        
        Args:
            lines: Detected lines from HoughLinesP
            threshold: Distance threshold for grouping
            
        Returns:
            list: Groups of line segments
        """
        segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segments.append([(x1, y1), (x2, y2)])
        
        # Group segments that are close to each other
        groups = []
        grouped = set()
        
        for i, segment1 in enumerate(segments):
            if i in grouped:
                continue
            
            group = [segment1]
            grouped.add(i)
            
            for j, segment2 in enumerate(segments):
                if j in grouped:
                    continue
                
                # Check if any endpoints are close
                if self.segments_are_close(segment1, segment2, threshold):
                    group.append(segment2)
                    grouped.add(j)
            
            groups.append(group)
        
        return groups
    
    def segments_are_close(self, segment1, segment2, threshold):
        """
        Check if two line segments are close to each other
        
        Args:
            segment1, segment2: Line segments as [(x1,y1), (x2,y2)]
            threshold: Distance threshold
            
        Returns:
            bool: True if segments are close
        """
        # Get endpoints
        p1, p2 = segment1
        p3, p4 = segment2
        
        # Calculate distances between endpoints
        d1 = self.euclidean_distance(p1, p3)
        d2 = self.euclidean_distance(p1, p4)
        d3 = self.euclidean_distance(p2, p3)
        d4 = self.euclidean_distance(p2, p4)
        
        # Return True if any pair of endpoints is close
        return min(d1, d2, d3, d4) < threshold
    
    def euclidean_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def order_segments(self, segments):
        """
        Order line segments to form a continuous path
        
        Args:
            segments: List of line segments
            
        Returns:
            list: Ordered segments
        """
        if not segments:
            return []
        
        # Start with the first segment
        ordered = [segments[0]]
        remaining = segments[1:]
        
        while remaining:
            # Get the last endpoint of the current path
            last_segment = ordered[-1]
            last_point = last_segment[1]
            
            # Find the closest segment
            closest_idx = -1
            closest_dist = float('inf')
            should_flip = False
            
            for i, segment in enumerate(remaining):
                # Check distance to start point
                d1 = self.euclidean_distance(last_point, segment[0])
                # Check distance to end point (might need to flip the segment)
                d2 = self.euclidean_distance(last_point, segment[1])
                
                if d1 < closest_dist:
                    closest_dist = d1
                    closest_idx = i
                    should_flip = False
                
                if d2 < closest_dist:
                    closest_dist = d2
                    closest_idx = i
                    should_flip = True
            
            if closest_idx == -1:
                break
            
            # Add the closest segment to the ordered list
            next_segment = remaining.pop(closest_idx)
            if should_flip:
                next_segment = [next_segment[1], next_segment[0]]
            
            ordered.append(next_segment)
        
        return ordered
    
    def add_detected_paths_to_osm_graph(self, detected_paths):
        """
        Add detected paths from aerial imagery to the OSM graph
        
        Args:
            detected_paths: List of detected paths
        """
        if not self.osm_graph:
            print("No OSM graph available to add detected paths")
            return
        
        # Track added nodes to avoid duplicates
        added_nodes = set()
        
        # Add each detected path to the graph
        for path in detected_paths:
            prev_node_id = None
            
            for node in path['nodes']:
                node_id = node['id']
                
                # Add node if not already in the graph
                if node_id not in self.osm_graph.nodes and node_id not in added_nodes:
                    self.osm_graph.add_node(
                        node_id,
                        y=node['lat'],  # OSMnx uses y for lat, x for lng
                        x=node['lng'],
                        osmid=node_id
                    )
                    added_nodes.add(node_id)
                
                # Connect to previous node
                if prev_node_id:
                    # Calculate distance between nodes
                    lat1, lng1 = node['lat'], node['lng']
                    prev_node = next((n for n in path['nodes'] if n['id'] == prev_node_id), None)
                    
                    if prev_node:
                        lat2, lng2 = prev_node['lat'], prev_node['lng']
                        distance = self.calculate_distance(lat1, lng1, lat2, lng2)
                        
                        # Add edge with distance as weight
                        self.osm_graph.add_edge(
                            prev_node_id,
                            node_id,
                            length=distance * 1000,  # Convert to meters
                            highway='path',
                            oneway=False,
                            source_type='detected'
                        )
                
                prev_node_id = node_id
    
    def determine_path_type_from_osm(self, edge):
        """
        Determine path type based on OSM tags - Enhanced to handle more path types
        
        Args:
            edge: Edge data from OSM
            
        Returns:
            str: Path type
        """
        # Get highway type
        highway = edge.get('highway', '')
        
        # Look for specific features that might indicate path types
        surface = str(edge.get('surface', '')).lower()
        is_designated = edge.get('foot', '') == 'designated'
        name = str(edge.get('name', '')).lower()
        service = str(edge.get('service', '')).lower()
        
        # Check for specific path types
        if highway in ['motorway', 'trunk', 'primary', 'secondary']:
            return 'main_road'
        elif highway in ['tertiary', 'residential', 'service']:
            if service in ['parking_aisle', 'driveway']:
                return 'access_road'
            return 'road'
        elif highway in ['footway', 'path', 'pedestrian']:
            # Differentiate between different types of pedestrian paths
            if surface in ['asphalt', 'paved', 'concrete']:
                return 'sidewalk'
            elif surface in ['gravel', 'unpaved', 'dirt', 'grass']:
                return 'trail'
            elif 'trail' in name or 'hiking' in name:
                return 'trail'
            return 'sidewalk'
        elif highway in ['steps', 'stairway']:
            return 'stairs'
        elif highway in ['cycleway']:
            return 'accessible'  # Assuming bike paths are accessible
        elif highway in ['track', 'bridleway']:
            return 'trail'
        elif highway in ['corridor']:
            return 'indoor'
        elif is_designated or 'footpath' in name:
            return 'sidewalk'
        elif (highway in ['service', 'track']) and ('access_road' in name or 'service' in name):
            return 'access_road'
        elif highway in ['track'] or 'track' in name or surface in ['unpaved', 'dirt', 'gravel']:
            return 'shortcut'
        
        # Default to sidewalk for unknown types
        return 'sidewalk'
    
    def visualize_osm_graph(self, save_path=None):
        """
        Create a visualization of the OSM graph for debugging purposes
        
        Args:
            save_path (str): Path to save the visualization image
        """
        if self.osm_graph is None:
            print("No OSM graph to visualize")
            return
            
        try:
            # Try using OSMnx built-in plotting function
            try:
                fig, ax = ox.plot_graph(self.osm_graph, figsize=(12, 10), node_size=10, 
                                        show=False, close=False, edge_linewidth=0.5)
            except Exception as e:
                print(f"Error with OSMnx plot_graph: {e}")
                print("Using fallback matplotlib visualization")
                
                # Fallback: Create a matplotlib plot manually
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Plot nodes
                for node, data in self.osm_graph.nodes(data=True):
                    if 'x' in data and 'y' in data:
                        ax.scatter(data['x'], data['y'], c='blue', s=5, alpha=0.7)
                    
                # Plot edges
                for u, v, data in self.osm_graph.edges(data=True):
                    u_data = self.osm_graph.nodes[u]
                    v_data = self.osm_graph.nodes[v]
                    
                    if 'x' in u_data and 'y' in u_data and 'x' in v_data and 'y' in v_data:
                        ax.plot([u_data['x'], v_data['x']], [u_data['y'], v_data['y']], 
                                c='gray', linewidth=0.5, alpha=0.5)
            
            # Overlay buildings if available
            if self.campus_data and 'buildings' in self.campus_data:
                for building in self.campus_data['buildings']:
                    ax.scatter(building['lng'], building['lat'], c='red', s=80, zorder=3, 
                               alpha=0.8, edgecolor='black')
                    ax.annotate(building.get('name', building['id']), 
                                (building['lng'], building['lat']), 
                                fontsize=8, ha='center', va='bottom', 
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Set title and limits
            plt.title('OSM Graph with Campus Buildings')
            
            # Improve axis labels and ticks
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            
            # Save if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            
            plt.close()
        except Exception as e:
            print(f"Error visualizing OSM graph: {e}")
            import traceback
            traceback.print_exc()
    
    def find_path_on_osm(self, start_lat, start_lng, end_lat, end_lng):
        """
        Find a path between two points using OpenStreetMap data
        
        Args:
            start_lat (float): Start latitude
            start_lng (float): Start longitude
            end_lat (float): End latitude
            end_lng (float): End longitude
            
        Returns:
            list: Path points
        """
        if self.osm_graph is None:
            print("Loading OSM data...")
            self.load_osm_data()
            
        if self.osm_graph is None:
            print("Failed to load OSM data")
            return []
            
        try:
            print(f"Finding path from ({start_lat}, {start_lng}) to ({end_lat}, {end_lng})")
            
            # Find nearest nodes to start and end points
            start_nodes = ox.nearest_nodes(self.osm_graph, X=[start_lng], Y=[start_lat], return_dist=True)
            end_nodes = ox.nearest_nodes(self.osm_graph, X=[end_lng], Y=[end_lat], return_dist=True)
            
            if isinstance(start_nodes, tuple) and len(start_nodes) == 2:
                # Newer OSMnx versions might return (nodes, distances)
                start_node = start_nodes[0][0]
                start_dist = start_nodes[1][0]
                print(f"Start point nearest node: {start_node} (distance: {start_dist:.2f}m)")
            else:
                # Older versions just return node IDs
                start_node = start_nodes[0]
                print(f"Start point nearest node: {start_node}")
                
            if isinstance(end_nodes, tuple) and len(end_nodes) == 2:
                end_node = end_nodes[0][0]
                end_dist = end_nodes[1][0]
                print(f"End point nearest node: {end_node} (distance: {end_dist:.2f}m)")
            else:
                end_node = end_nodes[0]
                print(f"End point nearest node: {end_node}")
            
            # Check if nodes exist in the graph
            if start_node not in self.osm_graph.nodes:
                print(f"Start node {start_node} not found in graph")
                return []
                
            if end_node not in self.osm_graph.nodes:
                print(f"End node {end_node} not found in graph")
                return []
            
            # Check if end node is reachable from start node
            if not nx.has_path(self.osm_graph, start_node, end_node):
                print(f"No path exists between nodes {start_node} and {end_node}")
                
                # Try to find alternative paths by using nodes up to 200m away
                print("Attempting to find alternative paths with more distant nodes...")
                
                # Get all nodes sorted by proximity to start and end
                start_coords = (start_lat, start_lng)
                end_coords = (end_lat, end_lng)
                
                # Find 5 closest nodes to start and end
                start_alternatives = self.find_alternative_nodes(start_coords, 5, 300)
                end_alternatives = self.find_alternative_nodes(end_coords, 5, 300)
                
                if not start_alternatives or not end_alternatives:
                    return []
                
                # Try all combinations until a path is found
                for alt_start in start_alternatives:
                    for alt_end in end_alternatives:
                        if nx.has_path(self.osm_graph, alt_start, alt_end):
                            print(f"Found alternative path from node {alt_start} to {alt_end}")
                            path_nodes = nx.shortest_path(self.osm_graph, alt_start, alt_end, weight='length')
                            
                            # Extract coordinates for each node in the path
                            path_points = []
                            
                            # Add the actual start point as first node
                            path_points.append({
                                "id": "osm_start",
                                "lat": start_lat,
                                "lng": start_lng,
                                "type": "path_node"
                            })
                            
                            # Add OSM path nodes
                            for node_id in path_nodes:
                                node = self.osm_graph.nodes[node_id]
                                path_points.append({
                                    "id": f"osm_{node_id}",
                                    "lat": node['y'],
                                    "lng": node['x'],
                                    "type": "osm_node" 
                                })
                            
                            # Add the actual end point as last node
                            path_points.append({
                                "id": "osm_end",
                                "lat": end_lat,
                                "lng": end_lng,
                                "type": "path_node"
                            })
                            
                            return path_points
                
                print("Could not find any alternative path")
                return []
            else:
                # Find shortest path
                path_nodes = nx.shortest_path(self.osm_graph, start_node, end_node, weight='length')
                print(f"Found path with {len(path_nodes)} nodes")
            
            # Calculate total distance
            total_distance = 0
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i+1]
                if 'length' in self.osm_graph.edges[u, v, 0]:
                    total_distance += self.osm_graph.edges[u, v, 0]['length']
            
            print(f"Total path distance: {total_distance:.2f} meters")
            
            # Extract coordinates for each node in the path
            path_points = []
            
            # Add the actual start point as first node
            path_points.append({
                "id": "osm_start",
                "lat": start_lat,
                "lng": start_lng,
                "type": "path_node"
            })
            
            # Add OSM path nodes
            for node_id in path_nodes:
                node = self.osm_graph.nodes[node_id]
                path_points.append({
                    "id": f"osm_{node_id}",
                    "lat": node['y'],
                    "lng": node['x'],
                    "type": "osm_node"
                })
            
            # Add the actual end point as last node
            path_points.append({
                "id": "osm_end",
                "lat": end_lat,
                "lng": end_lng,
                "type": "path_node"
            })
            
            # Debug display path characteristics
            print(f"Path contains {len(path_points)} points")
            
            return path_points
            
        except Exception as e:
            print(f"Error finding path on OSM: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    def find_alternative_nodes(self, coords, n_nodes=5, max_distance=200):
        """Find alternative nodes near a location for pathfinding
        
        Args:
            coords (tuple): (lat, lng) coordinates
            n_nodes (int): Number of nodes to find
            max_distance (float): Maximum distance in meters
            
        Returns:
            list: List of node IDs
        """
        if self.osm_graph is None:
            return []
            
        lat, lng = coords
        
        # Get all nodes and their coords
        nodes = []
        for node_id, data in self.osm_graph.nodes(data=True):
            if 'y' in data and 'x' in data:
                dist = self.calculate_distance(lat, lng, data['y'], data['x']) 
                if dist <= max_distance:
                    nodes.append((node_id, dist))
        
        # Sort by distance and take top n
        nodes.sort(key=lambda x: x[1])
        return [node_id for node_id, _ in nodes[:n_nodes]]
    
    def extract_paths_from_osm(self):
        """
        Extract paths from OpenStreetMap data and convert to our path format
        
        Returns:
            list: Paths in the campus data format
        """
        if self.osm_graph is None:
            self.load_osm_data()
            
        if self.osm_graph is None:
            print("Failed to load OSM data")
            return []
            
        # Extract edges and their geometry from the OSM graph
        paths = []
        edge_data = ox.graph_to_gdfs(self.osm_graph, nodes=False, edges=True)
        
        # Process each edge to create path objects
        for idx, edge in edge_data.iterrows():
            # Skip edges without geometry
            if 'geometry' not in edge or edge.geometry is None:
                continue
                
            # Extract coordinates from the LineString geometry
            if isinstance(edge.geometry, LineString):
                path_nodes = []
                coords = list(edge.geometry.coords)
                
                # Create nodes for each point in the LineString
                for i, (lng, lat) in enumerate(coords):
                    node_id = f"osm_{edge['osmid']}_{i}"
                    path_nodes.append({
                        "id": node_id,
                        "lat": lat,
                        "lng": lng
                    })
                
                # Determine path type from OSM tags
                path_type = self.determine_path_type_from_osm(edge)
                
                # Create path object
                if len(path_nodes) >= 2:
                    # Handle NaN value in name - convert to null
                    name = edge.get('name', '')
                    if name is None or (isinstance(name, float) and math.isnan(name)):
                        name = None
                    
                    path = {
                        "from": path_nodes[0]["id"],
                        "to": path_nodes[-1]["id"],
                        "type": path_type,
                        "nodes": path_nodes,
                        "osm_id": str(edge['osmid']),
                        "name": name
                    }
                    
                    paths.append(path)
        
        print(f"Extracted {len(paths)} paths from OpenStreetMap")
        return paths
    
    def integrate_osm_paths(self):
        """
        Integrate OpenStreetMap paths with existing paths in campus data
        
        Returns:
            list: Combined paths
        """
        # Get existing paths
        existing_paths = self.campus_data['paths']
        
        # Extract OSM paths
        osm_paths = self.extract_paths_from_osm()
        
        # Generate grid paths for areas without OSM coverage
        grid_paths = self.generate_grid_paths(resolution=15)
        
        # Combine paths
        combined_paths = existing_paths + osm_paths + grid_paths
        
        # De-duplicate paths that might overlap
        unique_paths = self.deduplicate_paths(combined_paths)
        
        # Update campus data
        self.campus_data['paths'] = unique_paths
        
        return unique_paths
    
    def deduplicate_paths(self, paths, threshold=10):
        """
        Remove duplicate or very similar paths
        
        Args:
            paths: List of paths
            threshold: Distance threshold in meters for considering paths as duplicates
            
        Returns:
            list: De-duplicated paths
        """
        unique_paths = []
        path_signatures = set()
        
        for path in paths:
            if len(path['nodes']) < 2:
                continue
            
            # Create a simplified signature for the path
            start_node = path['nodes'][0]
            end_node = path['nodes'][-1]
            
            # Round coordinates to reduce precision (helps with deduplication)
            start_lat = round(start_node['lat'], 5)
            start_lng = round(start_node['lng'], 5)
            end_lat = round(end_node['lat'], 5)
            end_lng = round(end_node['lng'], 5)
            
            # Create signature
            signature = f"{start_lat}_{start_lng}_{end_lat}_{end_lng}"
            
            # Check if we already have a very similar path
            if signature in path_signatures:
                continue
            
            # Check distances to existing paths
            duplicate = False
            for existing_path in unique_paths:
                if self.paths_are_similar(path, existing_path, threshold):
                    duplicate = True
                    break
            
            if not duplicate:
                path_signatures.add(signature)
                unique_paths.append(path)
        
        return unique_paths
    
    def paths_are_similar(self, path1, path2, threshold):
        """
        Check if two paths are similar based on node proximity
        
        Args:
            path1, path2: Paths to compare
            threshold: Distance threshold in meters
            
        Returns:
            bool: True if paths are similar
        """
        # Check start and end points
        start1 = path1['nodes'][0]
        end1 = path1['nodes'][-1]
        start2 = path2['nodes'][0]
        end2 = path2['nodes'][-1]
        
        # Calculate distances
        start_dist = self.calculate_distance(
            start1['lat'], start1['lng'],
            start2['lat'], start2['lng']
        )
        
        end_dist = self.calculate_distance(
            end1['lat'], end1['lng'],
            end2['lat'], end2['lng']
        )
        
        # Check if both start and end points are close
        if start_dist < threshold/1000 and end_dist < threshold/1000:
            return True
        
        # Also check reversed path
        rev_start_dist = self.calculate_distance(
            start1['lat'], start1['lng'],
            end2['lat'], end2['lng']
        )
        
        rev_end_dist = self.calculate_distance(
            end1['lat'], end1['lng'],
            start2['lat'], start2['lng']
        )
        
        if rev_start_dist < threshold/1000 and rev_end_dist < threshold/1000:
            return True
        
        return False
    
    def generate_grid_paths(self, resolution=15):
        """
        Generate a denser grid of paths covering the campus area
        
        Args:
            resolution (int): Number of grid cells in each dimension, increased for better coverage
            
        Returns:
            list: Generated grid paths
        """
        # Find bounds of campus
        min_lat = min(b['lat'] for b in self.campus_data['buildings'])
        max_lat = max(b['lat'] for b in self.campus_data['buildings'])
        min_lng = min(b['lng'] for b in self.campus_data['buildings'])
        max_lng = max(b['lng'] for b in self.campus_data['buildings'])
        
        # Add some padding
        padding = 0.002  # Increased from 0.001 for better coverage
        min_lat -= padding
        max_lat += padding
        min_lng -= padding
        max_lng += padding
        
        # Create grid
        lat_step = (max_lat - min_lat) / resolution
        lng_step = (max_lng - min_lng) / resolution
        
        grid_paths = []
        
        # Create horizontal grid lines
        for i in range(resolution + 1):
            lat = min_lat + i * lat_step
            
            nodes = []
            for j in range(resolution + 1):
                lng = min_lng + j * lng_step
                node_id = f"grid_h_{i}_{j}"
                
                nodes.append({
                    "id": node_id,
                    "lat": lat,
                    "lng": lng
                })
            
            grid_paths.append({
                "type": "grid_connection",
                "nodes": nodes
            })
        
        # Create vertical grid lines
        for j in range(resolution + 1):
            lng = min_lng + j * lng_step
            
            nodes = []
            for i in range(resolution + 1):
                lat = min_lat + i * lat_step
                node_id = f"grid_v_{i}_{j}"
                
                nodes.append({
                    "id": node_id,
                    "lat": lat,
                    "lng": lng
                })
            
            grid_paths.append({
                "type": "grid_connection",
                "nodes": nodes
            })
        
        # Create diagonal grid lines for better connectivity
        for i in range(resolution):
            for j in range(resolution):
                # First diagonal (top-left to bottom-right)
                nodes = []
                node_id1 = f"grid_d1_{i}_{j}"
                node_id2 = f"grid_d1_{i+1}_{j+1}"
                
                nodes.append({
                    "id": node_id1,
                    "lat": min_lat + i * lat_step,
                    "lng": min_lng + j * lng_step
                })
                
                nodes.append({
                    "id": node_id2,
                    "lat": min_lat + (i+1) * lat_step,
                    "lng": min_lng + (j+1) * lng_step
                })
                
                grid_paths.append({
                    "type": "grid_connection",
                    "nodes": nodes
                })
                
                # Second diagonal (bottom-left to top-right)
                nodes = []
                node_id1 = f"grid_d2_{i+1}_{j}"
                node_id2 = f"grid_d2_{i}_{j+1}"
                
                nodes.append({
                    "id": node_id1,
                    "lat": min_lat + (i+1) * lat_step,
                    "lng": min_lng + j * lng_step
                })
                
                nodes.append({
                    "id": node_id2,
                    "lat": min_lat + i * lat_step,
                    "lng": min_lng + (j+1) * lng_step
                })
                
                grid_paths.append({
                    "type": "grid_connection",
                    "nodes": nodes
                })
        
        return grid_paths
    
    def find_nearest_osm_nodes(self, lat, lng, n=3):
        """
        Find the nearest OSM nodes to a given location
        
        Args:
            lat (float): Latitude
            lng (float): Longitude
            n (int): Number of nearest nodes to return
            
        Returns:
            list: Nearest OSM node IDs
        """
        if self.osm_graph is None:
            self.load_osm_data()
            
        if self.osm_graph is None:
            return None
            
        # Find nearest nodes
        nearest_nodes = ox.nearest_nodes(self.osm_graph, X=[lng], Y=[lat], return_dist=False)
        
        return nearest_nodes
    
    def create_path_graph_with_osm(self, include_existing=True, include_terrain=False):
        """
        Create a path graph incorporating OpenStreetMap data
        
        Args:
            include_existing (bool): Whether to include existing paths
            include_terrain (bool): Whether to include terrain data
            
        Returns:
            networkx.Graph: Combined graph
        """
        # First create a graph with existing paths
        G = self.create_path_graph(include_existing, include_terrain)
        
        # If we have an OSM graph, integrate it
        if self.osm_graph is not None:
            try:
                # Extract node information from OSM graph
                for node_id, node_data in self.osm_graph.nodes(data=True):
                    osm_node_id = f"osm_{node_id}"
                    
                    # Ensure node has the required coordinates
                    if 'y' not in node_data or 'x' not in node_data:
                        continue
                        
                    # Add OSM node to our graph if not already present
                    if osm_node_id not in G:
                        G.add_node(
                            osm_node_id,
                            pos=(node_data['y'], node_data['x']),
                            type='osm_node'
                        )
                
                # Extract edge information from OSM graph
                for u, v, edge_data in self.osm_graph.edges(data=True):
                    osm_u = f"osm_{u}"
                    osm_v = f"osm_{v}"
                    
                    # Ensure both nodes exist
                    if osm_u in G and osm_v in G:
                        # Determine edge weight based on path type
                        path_type = self.determine_path_type_from_osm(edge_data)
                        weight_factor = self.path_types.get(path_type, {'weight': 1.0})['weight']
                        
                        # Calculate distance
                        u_pos = G.nodes[osm_u]['pos']
                        v_pos = G.nodes[osm_v]['pos']
                        distance = self.calculate_distance(u_pos[0], u_pos[1], v_pos[0], v_pos[1])
                        
                        # Add edge with appropriate weight
                        G.add_edge(
                            osm_u,
                            osm_v,
                            weight=distance * weight_factor,
                            distance=distance,
                            path_type=path_type
                        )
                
                # Connect OSM nodes to existing nodes in our graph
                self.connect_osm_to_existing(G)
                
                print(f"Successfully integrated OSM data: added {len([n for n in G.nodes if n.startswith('osm_')])} OSM nodes")
                
            except Exception as e:
                print(f"Error integrating OSM data into graph: {e}")
                import traceback
                traceback.print_exc()
                # We'll continue with the basic graph without OSM data
                print("Continuing with basic campus graph without OSM data")
        else:
            print("No OSM graph available, using basic campus graph")
        
        self.graph = G
        return G
    
    def connect_osm_to_existing(self, G):
        """
        Connect OSM nodes to existing nodes to create a fully connected graph
        
        Args:
            G (networkx.Graph): Graph to connect
        """
        try:
            # Find OSM nodes and existing nodes
            osm_nodes = [n for n in G.nodes() if isinstance(n, str) and n.startswith('osm_')]
            existing_nodes = [n for n in G.nodes() if n not in osm_nodes]
            
            # If either list is empty, we can't make connections
            if not osm_nodes or not existing_nodes:
                print(f"Unable to connect OSM to existing: OSM nodes: {len(osm_nodes)}, Existing nodes: {len(existing_nodes)}")
                return
                
            # Maximum distance to connect nodes (50 meters)
            MAX_CONNECT_DISTANCE = 50
            
            # Create points list for nearest neighbors search
            existing_points = []
            existing_node_indices = []
            
            for i, node_id in enumerate(existing_nodes):
                if 'pos' in G.nodes[node_id]:
                    existing_points.append(G.nodes[node_id]['pos'])
                    existing_node_indices.append(i)
            
            if not existing_points:
                print("No existing nodes with position data, skipping OSM connection")
                return
                
            # Create the nearest neighbors model
            try:
                existing_points_array = np.array(existing_points)
                nbrs = NearestNeighbors(n_neighbors=min(3, len(existing_points)), algorithm='ball_tree').fit(existing_points_array)
                
                # Connect each OSM node to nearest existing nodes if close enough
                connections_made = 0
                
                for osm_node in osm_nodes:
                    if 'pos' in G.nodes[osm_node]:
                        osm_pos = G.nodes[osm_node]['pos']
                        
                        # Find nearest existing nodes
                        distances, indices = nbrs.kneighbors([[osm_pos[0], osm_pos[1]]])
                        
                        # Connect to nearby nodes
                        for i, idx in enumerate(indices[0]):
                            if idx < len(existing_points):
                                original_idx = existing_node_indices[idx]
                                if original_idx < len(existing_nodes):
                                    existing_node = existing_nodes[original_idx]
                                    distance = distances[0][i] * 1000  # Convert to meters
                                    
                                    if distance <= MAX_CONNECT_DISTANCE:
                                        # Add edge with weight based on distance
                                        G.add_edge(
                                            osm_node,
                                            existing_node,
                                            weight=distance * 0.9,  # Slightly favor connections to existing network
                                            distance=distance,
                                            path_type='connector'
                                        )
                                        connections_made += 1
                
                print(f"Connected {connections_made} edges between OSM and existing nodes")
                
            except Exception as e:
                print(f"Error in nearest neighbors calculation: {e}")
                # Fall back to a simpler, more manual connection method
                self._connect_osm_fallback(G, osm_nodes, existing_nodes)
        
        except Exception as e:
            print(f"Error connecting OSM to existing nodes: {e}")
            import traceback
            traceback.print_exc()
    
    def _connect_osm_fallback(self, G, osm_nodes, existing_nodes, max_distance=50):
        """Fallback method to connect OSM nodes to existing nodes"""
        print("Using fallback connection method")
        connections_made = 0
        
        # Use a simpler approach - check all nodes within a reasonable distance
        # Sample a limited number of OSM and existing nodes to avoid excessive calculations
        sample_osm = osm_nodes[:min(50, len(osm_nodes))]
        sample_existing = existing_nodes[:min(50, len(existing_nodes))]
        
        for osm_node in sample_osm:
            if 'pos' not in G.nodes[osm_node]:
                continue
                
            osm_pos = G.nodes[osm_node]['pos']
            
            for existing_node in sample_existing:
                if 'pos' not in G.nodes[existing_node]:
                    continue
                    
                existing_pos = G.nodes[existing_node]['pos']
                
                # Calculate distance
                distance = self.calculate_distance(
                    osm_pos[0], osm_pos[1], 
                    existing_pos[0], existing_pos[1]
                )
                
                if distance <= max_distance:
                    # Add edge
                    G.add_edge(
                        osm_node,
                        existing_node,
                        weight=distance * 0.9,
                        distance=distance,
                        path_type='connector'
                    )
                    connections_made += 1
        
        print(f"Fallback method connected {connections_made} edges") 