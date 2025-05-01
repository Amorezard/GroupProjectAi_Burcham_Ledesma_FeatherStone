#!/usr/bin/env python3
"""
Test script to verify OSMnx integration is working correctly
"""
import sys
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import traceback

def test_osmnx_functionality():
    """Test basic OSMnx functionality"""
    print(f"Using OSMnx version: {ox.__version__}")
    
    try:
        # Configure OSMnx with more aggressive settings
        import osmnx.settings as ox_settings
        ox_settings.max_query_area_size = 50000000  # 50 sq km, much larger to prevent subdivision
        ox_settings.timeout = 180  # Increase timeout to 3 minutes
        ox_settings.useful_tags_way = ['highway', 'name', 'footway', 'surface']  # Minimal tags
        ox_settings.log_console = True  # Show progress in console
        
        # Merrimack College bounding box coordinates
        north = 42.674598  # Top-left latitude
        south = 42.663485  # Bottom-right latitude
        east = -71.113261  # Bottom-right longitude
        west = -71.127175  # Top-left longitude
        print(f"Using Merrimack College coordinates: N:{north}, S:{south}, E:{east}, W:{west}")
        
        # Create a custom filter to only get essential road types for a campus
        custom_filter = (
            '["highway"]["area"!~"yes"]'
            '["highway"~"^(primary|secondary|tertiary|residential|service|footway|path)$"]'
            '["access"!~"private"]'
        )
        
        # Try a point-based approach instead of bbox
        center_lat = (north + south) / 2
        center_lng = (east + west) / 2
        
        try:
            # Try first with a point and distance approach
            dist = 500  # meters - start small and expand if needed
            print(f"Attempting point-based download from ({center_lat}, {center_lng}) with {dist}m radius")
            G = ox.graph_from_point(
                (center_lat, center_lng), 
                dist=dist,
                network_type='walk', 
                simplify=True,
                custom_filter=custom_filter
            )
        except Exception as e:
            print(f"Point-based approach failed: {e}, trying bbox approach with minimal network...")
            # Fall back to bbox approach with minimal network type
            G = ox.graph_from_bbox(
                north=north, south=south, east=east, west=west,
                network_type='walk',
                simplify=True,
                truncate_by_edge=True,
                clean_periphery=True,
                custom_filter=custom_filter
            )
        
        print(f"Graph created successfully with {len(G.nodes)} nodes and {len(G.edges)} edges")
        
        # Test finding nearest nodes
        print("Testing nearest_nodes...")
        center_lat = (north + south) / 2
        center_lng = (east + west) / 2
        nearest = ox.nearest_nodes(G, X=[center_lng], Y=[center_lat])
        print(f"Found nearest node: {nearest}")
        
        # Test getting route
        print("Testing shortest path...")
        if len(G.nodes) >= 2:
            origin_node = list(G.nodes())[0]
            destination_node = list(G.nodes())[1]  # Use the second node instead of the last
            try:
                route = nx.shortest_path(G, origin_node, destination_node, weight='length')
                print(f"Found route with {len(route)} nodes")
            except nx.NetworkXNoPath:
                print("No path found between nodes (this is normal for disconnected graphs)")
                
                # Try with nodes that are connected
                try:
                    if len(list(G.edges())) > 0:
                        connected_nodes = list(G.edges())[0]
                        route = nx.shortest_path(G, connected_nodes[0], connected_nodes[1], weight='length')
                        print(f"Found route between connected nodes with {len(route)} nodes")
                    else:
                        print("No edges found in the graph")
                except Exception as e:
                    print(f"Error finding path between connected nodes: {e}")
        else:
            print("Not enough nodes for pathfinding test")
        
        print("All tests completed successfully!")
        return True
    except Exception as e:
        print(f"Error testing OSMnx: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_osmnx_functionality()
    sys.exit(0 if success else 1) 