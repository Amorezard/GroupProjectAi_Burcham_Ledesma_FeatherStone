from flask import Flask, render_template, send_from_directory, jsonify, request, redirect, url_for
import os
import json
import math
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Updated data with actual Merrimack College coordinates and buildings
CAMPUS_DATA = {
    "origin": {
        "lat": 42.6686,  # Merrimack College central coordinates
        "lng": -71.1094
    },
    "buildings": [
        {
            "id": "austin",
            "name": "Austin Hall",
            "lat": 42.6703,
            "lng": -71.1087,
            "description": "Home to the Girard School of Business",
            "floors": 3,
            "image": "austin.jpg"
        },
        {
            "id": "mcquade",
            "name": "McQuade Library",
            "lat": 42.6694,
            "lng": -71.1080,
            "description": "Main campus library",
            "floors": 3,
            "image": "mcquade.jpg"
        },
        {
            "id": "mendel",
            "name": "Mendel Center",
            "lat": 42.6682,
            "lng": -71.1086,
            "description": "Science and Engineering Center",
            "floors": 2,
            "image": "mendel.jpg"
        },
        {
            "id": "oreilly",
            "name": "O'Reilly Hall",
            "lat": 42.6678,
            "lng": -71.1101,
            "description": "Administrative offices",
            "floors": 2,
            "image": "oreilly.jpg"
        },
        {
            "id": "rogers",
            "name": "Rogers Center",
            "lat": 42.6679,
            "lng": -71.1113,
            "description": "Arts and performance venue",
            "floors": 2,
            "image": "rogers.jpg"
        },
        {
            "id": "sakowich",
            "name": "Sakowich Campus Center",
            "lat": 42.6688,
            "lng": -71.1096,
            "description": "Student center with dining and activities",
            "floors": 3,
            "image": "sakowich.jpg"
        },
        {
            "id": "cushing",
            "name": "Cushing Hall",
            "lat": 42.6705,
            "lng": -71.1077,
            "description": "Liberal arts classrooms",
            "floors": 2,
            "image": "cushing.jpg"
        },
        {
            "id": "sullivanHall",
            "name": "Sullivan Hall",
            "lat": 42.6698, 
            "lng": -71.1071,
            "description": "Student residence",
            "floors": 3,
            "image": "sullivan.jpg"
        },
        {
            "id": "arcidi",
            "name": "Arcidi Center",
            "lat": 42.6677,
            "lng": -71.1123,
            "description": "Welcome Center",
            "floors": 2,
            "image": "arcidi.jpg"
        }
    ],
    "rooms": {
        "austin": [
            {"id": "a101", "name": "Austin 101", "floor": 1, "type": "classroom"},
            {"id": "a102", "name": "Austin 102", "floor": 1, "type": "classroom"},
            {"id": "a201", "name": "Austin 201", "floor": 2, "type": "classroom"},
            {"id": "a202", "name": "Austin 202", "floor": 2, "type": "classroom"}
        ],
        "mcquade": [
            {"id": "mq1", "name": "First Floor", "floor": 1, "type": "study"},
            {"id": "mq2", "name": "Second Floor", "floor": 2, "type": "study"},
            {"id": "mq3", "name": "Third Floor", "floor": 3, "type": "quiet"}
        ],
        "mendel": [
            {"id": "me101", "name": "Mendel 101", "floor": 1, "type": "lab"},
            {"id": "me102", "name": "Mendel 102", "floor": 1, "type": "classroom"},
            {"id": "me201", "name": "Mendel 201", "floor": 2, "type": "lab"}
        ],
        "oreilly": [
            {"id": "or101", "name": "O'Reilly 101", "floor": 1, "type": "office"},
            {"id": "or102", "name": "O'Reilly 102", "floor": 1, "type": "office"}
        ],
        "rogers": [
            {"id": "rg101", "name": "Main Theater", "floor": 1, "type": "theater"},
            {"id": "rg102", "name": "Gallery", "floor": 1, "type": "gallery"}
        ],
        "sakowich": [
            {"id": "s101", "name": "Dining Hall", "floor": 1, "type": "dining"},
            {"id": "s102", "name": "Bookstore", "floor": 1, "type": "shop"},
            {"id": "s201", "name": "MPR", "floor": 2, "type": "meeting"}
        ],
        "cushing": [
            {"id": "c101", "name": "Cushing 101", "floor": 1, "type": "classroom"},
            {"id": "c102", "name": "Cushing 102", "floor": 1, "type": "classroom"}
        ],
        "sullivanHall": [
            {"id": "sh101", "name": "Lobby", "floor": 1, "type": "common"},
            {"id": "sh201", "name": "Second Floor", "floor": 2, "type": "residence"}
        ],
        "arcidi": [
            {"id": "ar101", "name": "Welcome Center", "floor": 1, "type": "admin"},
            {"id": "ar102", "name": "Admission Office", "floor": 1, "type": "admin"},
            {"id": "ar201", "name": "Conference Room", "floor": 2, "type": "meeting"}
        ]
    },
    "paths": [
        # Main campus quad paths
        {"from": "mcquade", "to": "austin", "nodes": [
            {"id": "n1", "lat": 42.6694, "lng": -71.1080},
            {"id": "n2", "lat": 42.6693, "lng": -71.1083},
            {"id": "n3", "lat": 42.6692, "lng": -71.1087}
        ]},
        {"from": "austin", "to": "sakowich", "nodes": [
            {"id": "n4", "lat": 42.6692, "lng": -71.1087},
            {"id": "n5", "lat": 42.6690, "lng": -71.1088},
            {"id": "n6", "lat": 42.6688, "lng": -71.1088}
        ]},
        {"from": "sakowich", "to": "rogers", "nodes": [
            {"id": "n7", "lat": 42.6688, "lng": -71.1088},
            {"id": "n8", "lat": 42.6685, "lng": -71.1090},
            {"id": "n9", "lat": 42.6683, "lng": -71.1093}
        ]},
        {"from": "rogers", "to": "oreilly", "nodes": [
            {"id": "n10", "lat": 42.6683, "lng": -71.1093},
            {"id": "n11", "lat": 42.6682, "lng": -71.1089},
            {"id": "n12", "lat": 42.6682, "lng": -71.1085}
        ]},
        {"from": "oreilly", "to": "mendel", "nodes": [
            {"id": "n13", "lat": 42.6682, "lng": -71.1085},
            {"id": "n14", "lat": 42.6683, "lng": -71.1081},
            {"id": "n15", "lat": 42.6685, "lng": -71.1076}
        ]},
        {"from": "mendel", "to": "mcquade", "nodes": [
            {"id": "n16", "lat": 42.6685, "lng": -71.1076},
            {"id": "n17", "lat": 42.6689, "lng": -71.1077},
            {"id": "n18", "lat": 42.6694, "lng": -71.1080}
        ]},
        {"from": "mcquade", "to": "cushing", "nodes": [
            {"id": "n19", "lat": 42.6694, "lng": -71.1080},
            {"id": "n20", "lat": 42.6696, "lng": -71.1081},
            {"id": "n21", "lat": 42.6697, "lng": -71.1082}
        ]},
        {"from": "mendel", "to": "sullivanHall", "nodes": [
            {"id": "n22", "lat": 42.6685, "lng": -71.1076},
            {"id": "n23", "lat": 42.6687, "lng": -71.1075},
            {"id": "n24", "lat": 42.6689, "lng": -71.1073}
        ]},
        # Cross-campus paths (diagonal)
        {"from": "sakowich", "to": "mendel", "nodes": [
            {"id": "n25", "lat": 42.6688, "lng": -71.1088},
            {"id": "n26", "lat": 42.6687, "lng": -71.1083},
            {"id": "n27", "lat": 42.6685, "lng": -71.1076}
        ]},
        {"from": "austin", "to": "mcquade", "nodes": [
            {"id": "n28", "lat": 42.6692, "lng": -71.1087},
            {"id": "n29", "lat": 42.6693, "lng": -71.1084},
            {"id": "n30", "lat": 42.6694, "lng": -71.1080}
        ]},
        # Main entrance path from Arcidi Center (Welcome Center)
        {"from": "arcidi", "to": "sakowich", "nodes": [
            {"id": "n31", "lat": 42.6677, "lng": -71.1123},
            {"id": "n32", "lat": 42.6682, "lng": -71.1105},
            {"id": "n33", "lat": 42.6688, "lng": -71.1096}
        ]},
        # Route from Arcidi Center to Rogers Center
        {"from": "arcidi", "to": "rogers", "nodes": [
            {"id": "n34", "lat": 42.6677, "lng": -71.1123},
            {"id": "n35", "lat": 42.6678, "lng": -71.1118},
            {"id": "n36", "lat": 42.6679, "lng": -71.1113}
        ]},
        # Route along Turnpike St (main road)
        {"from": "campus_entrance", "to": "arcidi", "nodes": [
            {"id": "n37", "lat": 42.6670, "lng": -71.1133},
            {"id": "n38", "lat": 42.6674, "lng": -71.1128},
            {"id": "n39", "lat": 42.6677, "lng": -71.1123}
        ]},
        # Central athletic quad path
        {"from": "sakowich", "to": "cushing", "nodes": [
            {"id": "n40", "lat": 42.6688, "lng": -71.1096},
            {"id": "n41", "lat": 42.6697, "lng": -71.1087},
            {"id": "n42", "lat": 42.6705, "lng": -71.1077}
        ]},
        # Route from O'Reilly to Sakowich
        {"from": "oreilly", "to": "sakowich", "nodes": [
            {"id": "n43", "lat": 42.6678, "lng": -71.1101},
            {"id": "n44", "lat": 42.6683, "lng": -71.1098},
            {"id": "n45", "lat": 42.6688, "lng": -71.1096}
        ]}
    ],
    # Indoor nodes remain as before
    "indoor_nodes": {
        "mendel": {
            "m101": {"x": 10, "y": 20, "z": 0, "floor": 1, "connections": ["m102", "hallway1"]},
            "m102": {"x": 30, "y": 20, "z": 0, "floor": 1, "connections": ["m101", "hallway1"]},
            "hallway1": {"x": 20, "y": 10, "z": 0, "floor": 1, "connections": ["m101", "m102", "stairs1"]},
            "stairs1": {"x": 20, "y": 5, "z": 0, "floor": 1, "connections": ["hallway1", "hallway2"]},
            "hallway2": {"x": 20, "y": 10, "z": 0, "floor": 2, "connections": ["m201", "stairs1"]},
            "m201": {"x": 25, "y": 20, "z": 0, "floor": 2, "connections": ["hallway2"]}
        },
        "mcquade": {
            "mq1": {"x": 15, "y": 25, "z": 0, "floor": 1, "connections": ["entrance", "stairs1"]},
            "entrance": {"x": 5, "y": 25, "z": 0, "floor": 1, "connections": ["mq1"]},
            "stairs1": {"x": 25, "y": 25, "z": 0, "floor": 1, "connections": ["mq1", "mq2"]}
        }
    },
    "plaques": {
        "M101": {"building": "mendel", "roomId": "m101", "description": "Lecture Hall"},
        "M102": {"building": "mendel", "roomId": "m102", "description": "Chemistry Lab"},
        "M201": {"building": "mendel", "roomId": "m201", "description": "Physics Lab"},
        "MQ1": {"building": "mcquade", "roomId": "mq1", "description": "Library Main Floor"}
    }
}

@app.route('/')
def index():
    return redirect(url_for('wayfinding'))

@app.route('/hiro')
def hiro():
    return render_template('hiro.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/campus')
def campus():
    return render_template('campus_map.html')

@app.route('/plaque-nav')
def plaque_nav():
    return render_template('plaque_nav.html')

@app.route('/wayfinding')
def wayfinding():
    return render_template('wayfinding.html')

# API endpoints for campus data
@app.route('/api/buildings')
def get_buildings():
    return jsonify(CAMPUS_DATA['buildings'])

@app.route('/api/buildings/<building_id>')
def get_building(building_id):
    for building in CAMPUS_DATA['buildings']:
        if building['id'] == building_id:
            return jsonify(building)
    return jsonify({"error": "Building not found"}), 404

@app.route('/api/buildings/<building_id>/rooms')
def get_rooms(building_id):
    if building_id in CAMPUS_DATA['rooms']:
        return jsonify(CAMPUS_DATA['rooms'][building_id])
    return jsonify({"error": "Building not found"}), 404

@app.route('/api/navigate')
def navigate():
    # Get parameters
    building_id = request.args.get('buildingId')
    room_id = request.args.get('roomId')
    user_lat = float(request.args.get('lat', 0))
    user_lng = float(request.args.get('lng', 0))
    
    # Find building
    building = None
    for b in CAMPUS_DATA['buildings']:
        if b['id'] == building_id:
            building = b
            break
    
    if not building:
        return jsonify({"error": "Building not found"}), 404
    
    # Find room
    room = None
    if building_id in CAMPUS_DATA['rooms']:
        for r in CAMPUS_DATA['rooms'][building_id]:
            if r['id'] == room_id:
                room = r
                break
    
    if not room:
        return jsonify({"error": "Room not found"}), 404
    
    # Find path to building
    # Create a custom start node for the user's location
    user_node = f"user_location_{user_lat}_{user_lng}"
    
    # Add the user's location to the graph temporarily 
    custom_path = {
        "from": user_node,
        "to": building_id,
        "nodes": [
            {"id": user_node, "lat": user_lat, "lng": user_lng}
        ]
    }
    
    # Temporarily add this custom node to paths
    temp_paths = CAMPUS_DATA['paths'].copy()
    temp_paths.append(custom_path)
    
    # We need to find nearest node for user location
    nearest_building = find_nearest_node(user_lat, user_lng)
    
    # Add an edge between user location and nearest building
    custom_edge = {
        "from": user_node,
        "to": nearest_building,
        "nodes": [
            {"id": user_node, "lat": user_lat, "lng": user_lng},
            {"id": f"connector_{nearest_building}", "lat": (user_lat + CAMPUS_DATA['buildings'][0]['lat'])/2, 
             "lng": (user_lng + CAMPUS_DATA['buildings'][0]['lng'])/2}
        ]
    }
    temp_paths.append(custom_edge)
    
    # Temporarily replace paths
    original_paths = CAMPUS_DATA['paths']
    CAMPUS_DATA['paths'] = temp_paths
    
    # Calculate path
    path_points = calculate_path(user_node, building_id)
    
    # Restore original paths
    CAMPUS_DATA['paths'] = original_paths
    
    return jsonify({
        "path": path_points,
        "destination": {
            "lat": building['lat'],
            "lng": building['lng'],
            "name": f"{building['name']} - {room['name']}"
        },
        "info": {
            "building": building,
            "room": room
        }
    })

# New API endpoint for plaque data
@app.route('/api/plaques')
def get_plaques():
    return jsonify(CAMPUS_DATA['plaques'])

# New API endpoint for indoor navigation
@app.route('/api/navigate-indoor')
def navigate_indoor():
    from_building = request.args.get('fromBuilding')
    from_room = request.args.get('fromRoom')
    to_building = request.args.get('toBuilding')
    to_room = request.args.get('toRoom')
    
    if not all([from_building, from_room, to_building, to_room]):
        return jsonify({"error": "Missing parameters"}), 400
    
    # Check if buildings and rooms exist
    if from_building not in CAMPUS_DATA['indoor_nodes'] or to_building not in CAMPUS_DATA['indoor_nodes']:
        return jsonify({"error": "Building not found"}), 404
    
    if from_room not in CAMPUS_DATA['indoor_nodes'][from_building] or to_room not in CAMPUS_DATA['indoor_nodes'][to_building]:
        return jsonify({"error": "Room not found"}), 404
    
    # Same building navigation
    if from_building == to_building:
        path = find_indoor_path(from_building, from_room, to_room)
        
        if not path:
            return jsonify({"error": "No path found"}), 404
        
        # Format path for AR visualization
        path_points = []
        for node_id in path:
            node = CAMPUS_DATA['indoor_nodes'][from_building][node_id]
            path_points.append({
                "id": node_id,
                "x": node["x"],
                "y": node["y"],
                "z": node["z"],
                "floor": node["floor"]
            })
        
        # Get destination information
        to_room_data = next((r for r in CAMPUS_DATA['rooms'][to_building] if r['id'] == to_room), None)
        destination_name = to_room_data['name'] if to_room_data else to_room
        
        return jsonify({
            "path": path_points,
            "destination": {
                "id": to_room,
                "x": CAMPUS_DATA['indoor_nodes'][to_building][to_room]["x"],
                "y": CAMPUS_DATA['indoor_nodes'][to_building][to_room]["y"],
                "z": CAMPUS_DATA['indoor_nodes'][to_building][to_room]["z"],
                "name": destination_name
            }
        })
    else:
        # Cross-building navigation (simplified for demo)
        # In a real app, this would be more complex and consider building entrances/exits
        
        # First find path to exit current building
        path1 = find_indoor_path(from_building, from_room, "entrance")
        
        # Then find path from entrance of destination building to destination room
        path2 = find_indoor_path(to_building, "entrance", to_room)
        
        # Combine paths
        combined_path = []
        if path1:
            for node_id in path1:
                node = CAMPUS_DATA['indoor_nodes'][from_building][node_id]
                combined_path.append({
                    "id": node_id,
                    "building": from_building,
                    "x": node["x"],
                    "y": node["y"],
                    "z": node["z"],
                    "floor": node["floor"]
                })
        
        # Add outdoor path between buildings
        outdoor_path = next((p for p in CAMPUS_DATA['paths'] if p['from'] == from_building and p['to'] == to_building), None)
        if outdoor_path:
            for node in outdoor_path['nodes']:
                combined_path.append({
                    "id": node['id'],
                    "lat": node['lat'],
                    "lng": node['lng'],
                    "isOutdoor": True
                })
        
        # Add destination building path
        if path2:
            for node_id in path2:
                node = CAMPUS_DATA['indoor_nodes'][to_building][node_id]
                combined_path.append({
                    "id": node_id,
                    "building": to_building,
                    "x": node["x"],
                    "y": node["y"],
                    "z": node["z"],
                    "floor": node["floor"]
                })
        
        # Get destination information
        to_room_data = next((r for r in CAMPUS_DATA['rooms'][to_building] if r['id'] == to_room), None)
        destination_name = to_room_data['name'] if to_room_data else to_room
        
        return jsonify({
            "path": combined_path,
            "destination": {
                "id": to_room,
                "building": to_building,
                "x": CAMPUS_DATA['indoor_nodes'][to_building][to_room]["x"],
                "y": CAMPUS_DATA['indoor_nodes'][to_building][to_room]["y"],
                "z": CAMPUS_DATA['indoor_nodes'][to_building][to_room]["z"],
                "name": destination_name
            }
        })

@app.route('/api/find-path', methods=['POST'])
def find_path():
    data = request.json
    start_location = data.get('start')
    end_location = data.get('end')
    preferences = data.get('preferences', {})
    
    # Find nodes closest to start and end locations
    start_node = find_nearest_node(start_location['lat'], start_location['lng'])
    end_node = find_nearest_node(end_location['lat'], end_location['lng'])
    
    # Create a temporary graph with accessibility preferences
    G = create_graph_with_preferences(preferences)
    
    # Add start and end nodes to the graph
    G.add_node(
        'user_start',
        pos=(start_location['lat'], start_location['lng']),
        type='node'
    )
    
    # Connect start node to nearest building or path node
    nearest_to_start = find_nearest_node_for_preferences(start_location['lat'], start_location['lng'], preferences)
    nearest_node_pos = get_node_position(nearest_to_start)
    
    if nearest_node_pos:
        distance = calculate_distance(start_location['lat'], start_location['lng'], 
                                     nearest_node_pos[0], nearest_node_pos[1])
        G.add_edge('user_start', nearest_to_start, weight=distance)
    
    # Find path between nodes with preferences
    path = calculate_path_with_preferences('user_start', end_node, G, preferences)
    
    # Generate navigation instructions
    instructions = generate_instructions(path)
    
    return jsonify({
        'path': path,
        'instructions': instructions
    })

def create_graph_with_preferences(preferences):
    """Create a graph incorporating user preferences"""
    G = nx.Graph()
    
    # Add buildings as nodes
    for building in CAMPUS_DATA['buildings']:
        G.add_node(
            building['id'],
            pos=(building['lat'], building['lng']),
            type='building',
            name=building['name']
        )
    
    # Add path nodes and edges, considering preferences
    for path in CAMPUS_DATA['paths']:
        prev_node = None
        for node in path['nodes']:
            G.add_node(
                node['id'],
                pos=(node['lat'], node['lng']),
                type='node'
            )
            
            if prev_node:
                # Calculate actual distance between nodes
                lat1, lng1 = G.nodes[prev_node]['pos']
                lat2, lng2 = G.nodes[node['id']]['pos']
                distance = calculate_distance(lat1, lng1, lat2, lng2)
                
                # Apply preference weights
                weight = distance
                
                # If avoiding crowds and this is a crowded path, increase weight
                if preferences.get('avoidCrowds') and 'sakowich' in [prev_node, node['id']]:
                    weight *= 2  # Double the weight to make it less desirable
                
                # If preferring indoor paths and this is outdoor, increase weight
                if preferences.get('preferIndoor') and not any(b in [prev_node, node['id']] for b in ['austin', 'mendel', 'mcquade']):
                    weight *= 1.5
                
                # Add edge with modified weight
                G.add_edge(prev_node, node['id'], weight=weight)
                
            prev_node = node['id']
        
        # Connect buildings with path nodes
        if 'from' in path and 'to' in path:
            # Connect starting building
            if path['from'] in G.nodes and path['nodes']:
                first_node = path['nodes'][0]['id']
                from_building = path['from']
                lat1, lng1 = G.nodes[from_building]['pos']
                lat2, lng2 = G.nodes[first_node]['pos']
                distance = calculate_distance(lat1, lng1, lat2, lng2)
                
                # Apply accessibility preference
                if preferences.get('wheelchairAccessible') or preferences.get('avoidStairs'):
                    # Check if this building has accessibility issues
                    if from_building in ['sullivanHall', 'rogers']:  # Example buildings with stairs
                        distance *= 3  # Make much less desirable
                
                G.add_edge(from_building, first_node, weight=distance)
            
            # Connect ending building
            if path['to'] in G.nodes and path['nodes']:
                last_node = path['nodes'][-1]['id']
                to_building = path['to']
                lat1, lng1 = G.nodes[last_node]['pos']
                lat2, lng2 = G.nodes[to_building]['pos']
                distance = calculate_distance(lat1, lng1, lat2, lng2)
                
                # Apply accessibility preference
                if preferences.get('wheelchairAccessible') or preferences.get('avoidStairs'):
                    # Check if this building has accessibility issues
                    if to_building in ['sullivanHall', 'rogers']:  # Example buildings with stairs
                        distance *= 3  # Make much less desirable
                
                G.add_edge(last_node, to_building, weight=distance)
    
    return G

def find_nearest_node_for_preferences(lat, lng, preferences):
    """Find the node closest to the given coordinates, considering preferences"""
    # Extract nodes from CAMPUS_DATA, filtering by preferences
    nodes = []
    node_ids = []
    
    # Add building locations
    for building in CAMPUS_DATA['buildings']:
        # Skip buildings that don't meet accessibility requirements
        if (preferences.get('wheelchairAccessible') or preferences.get('avoidStairs')) and building['id'] in ['sullivanHall', 'rogers']:
            continue
        
        nodes.append([building['lat'], building['lng']])
        node_ids.append(building['id'])
    
    # Add path nodes
    for path in CAMPUS_DATA['paths']:
        for node in path['nodes']:
            nodes.append([node['lat'], node['lng']])
            node_ids.append(node['id'])
    
    # Find nearest node
    if not nodes:
        return None
    
    nodes_array = np.array(nodes)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(nodes_array)
    distances, indices = nbrs.kneighbors([[lat, lng]])
    
    return node_ids[indices[0][0]]

def get_node_position(node_id):
    """Get the position of a node by its ID"""
    # Check if it's a building
    for building in CAMPUS_DATA['buildings']:
        if building['id'] == node_id:
            return (building['lat'], building['lng'])
    
    # Check if it's a path node
    for path in CAMPUS_DATA['paths']:
        for node in path['nodes']:
            if node['id'] == node_id:
                return (node['lat'], node['lng'])
    
    return None

def calculate_path_with_preferences(start_node, end_node, G, preferences):
    """Calculate the path between two nodes using A* algorithm with preferences"""
    try:
        # Define heuristic function for A* (straight-line distance to goal)
        def heuristic(u, v):
            u_pos = G.nodes[u]['pos']
            v_pos = G.nodes[v]['pos']
            return calculate_distance(u_pos[0], u_pos[1], v_pos[0], v_pos[1])
        
        # Find shortest path using A*
        shortest_path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight='weight')
        
        # Calculate total distance of path
        total_distance = 0
        for i in range(len(shortest_path) - 1):
            u = shortest_path[i]
            v = shortest_path[i + 1]
            total_distance += G[u][v]['weight']
            
        print(f"A* path found with {len(shortest_path)} nodes and distance: {total_distance:.1f} meters")
        
        # Convert to lat/lng points
        path_points = []
        for node_id in shortest_path:
            node = G.nodes[node_id]
            if 'pos' in node:
                lat, lng = node['pos']
                name = node.get('name', '')
                path_points.append({
                    'id': node_id,
                    'lat': lat,
                    'lng': lng,
                    'name': name,
                    'type': node.get('type', 'node')
                })
        
        return path_points
    except (nx.NetworkXNoPath, KeyError) as e:
        print(f"Error finding path with preferences: {e}")
        return []

def find_nearest_node(lat, lng):
    """Find the node closest to the given coordinates"""
    # Extract nodes from CAMPUS_DATA
    nodes = []
    node_ids = []
    
    # Add building locations
    for building in CAMPUS_DATA['buildings']:
        nodes.append([building['lat'], building['lng']])
        node_ids.append(building['id'])
    
    # Add path nodes
    for path in CAMPUS_DATA['paths']:
        for node in path['nodes']:
            nodes.append([node['lat'], node['lng']])
            node_ids.append(node['id'])
    
    # Find nearest node
    if not nodes:
        return None
        
    nodes_array = np.array(nodes)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(nodes_array)
    distances, indices = nbrs.kneighbors([[lat, lng]])
    
    return node_ids[indices[0][0]]

def calculate_path(start_node, end_node):
    """Calculate the path between two nodes using A* algorithm"""
    # Create graph
    G = nx.Graph()
    
    # Add buildings as nodes
    for building in CAMPUS_DATA['buildings']:
        G.add_node(
            building['id'],
            pos=(building['lat'], building['lng']),
            type='building',
            name=building['name']
        )
    
    # Add path nodes and edges
    for path in CAMPUS_DATA['paths']:
        prev_node = None
        for node in path['nodes']:
            G.add_node(
                node['id'],
                pos=(node['lat'], node['lng']),
                type='node'
            )
            
            if prev_node:
                # Calculate actual distance between nodes (in meters)
                lat1, lng1 = G.nodes[prev_node]['pos']
                lat2, lng2 = G.nodes[node['id']]['pos']
                distance = calculate_distance(lat1, lng1, lat2, lng2)
                
                # Add edge with distance as weight
                G.add_edge(prev_node, node['id'], weight=distance)
                
            prev_node = node['id']
        
        # Connect first and last nodes to buildings if this is a building path
        if 'from' in path and 'to' in path:
            if path['from'] in G and path['nodes']:
                first_node = path['nodes'][0]['id']
                from_building = path['from']
                lat1, lng1 = G.nodes[from_building]['pos']
                lat2, lng2 = G.nodes[first_node]['pos']
                distance = calculate_distance(lat1, lng1, lat2, lng2)
                G.add_edge(from_building, first_node, weight=distance)
                
            if path['to'] in G and path['nodes']:
                last_node = path['nodes'][-1]['id']
                to_building = path['to']
                lat1, lng1 = G.nodes[last_node]['pos']
                lat2, lng2 = G.nodes[to_building]['pos']
                distance = calculate_distance(lat1, lng1, lat2, lng2)
                G.add_edge(last_node, to_building, weight=distance)
    
    # A* algorithm implementation
    try:
        # Define heuristic function for A* (straight-line distance to goal)
        def heuristic(u, v):
            u_pos = G.nodes[u]['pos']
            v_pos = G.nodes[v]['pos']
            return calculate_distance(u_pos[0], u_pos[1], v_pos[0], v_pos[1])
        
        # Find shortest path using A*
        shortest_path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight='weight')
        
        # Calculate total distance of path
        total_distance = 0
        for i in range(len(shortest_path) - 1):
            u = shortest_path[i]
            v = shortest_path[i + 1]
            total_distance += G[u][v]['weight']
            
        print(f"A* path found with {len(shortest_path)} nodes and distance: {total_distance:.1f} meters")
        
        # Convert to lat/lng points
        path_points = []
        for node_id in shortest_path:
            node = G.nodes[node_id]
            if 'pos' in node:
                lat, lng = node['pos']
                name = node.get('name', '')
                path_points.append({
                    'id': node_id,
                    'lat': lat,
                    'lng': lng,
                    'name': name,
                    'type': node.get('type', 'node')
                })
        
        return path_points
    except (nx.NetworkXNoPath, KeyError) as e:
        print(f"Error finding path: {e}")
        # Fallback to simple direct path if A* fails
        if start_node in G.nodes and end_node in G.nodes:
            start_pos = G.nodes[start_node]['pos']
            end_pos = G.nodes[end_node]['pos']
            return [
                {
                    'id': start_node,
                    'lat': start_pos[0],
                    'lng': start_pos[1],
                    'name': G.nodes[start_node].get('name', ''),
                    'type': G.nodes[start_node].get('type', 'node')
                },
                {
                    'id': end_node,
                    'lat': end_pos[0],
                    'lng': end_pos[1],
                    'name': G.nodes[end_node].get('name', ''),
                    'type': G.nodes[end_node].get('type', 'node')
                }
            ]
        return []

def generate_instructions(path):
    """Generate turn-by-turn instructions from path"""
    instructions = []
    
    if len(path) < 2:
        return instructions
    
    # First instruction is to start at the first node
    if path[0]['type'] == 'building':
        instructions.append(f"Start at {path[0]['name']}")
    else:
        instructions.append("Start at the marked location")
    
    # Generate instructions for each segment
    for i in range(1, len(path)-1):
        prev = path[i-1]
        curr = path[i]
        next_node = path[i+1]
        
        # Calculate direction change
        heading1 = calculate_bearing(prev['lat'], prev['lng'], curr['lat'], curr['lng'])
        heading2 = calculate_bearing(curr['lat'], curr['lng'], next_node['lat'], next_node['lng'])
        
        turn_angle = (heading2 - heading1 + 360) % 360
        
        if turn_angle < 45 or turn_angle > 315:
            direction = "Continue straight"
        elif turn_angle < 135:
            direction = "Turn right"
        elif turn_angle > 225:
            direction = "Turn left"
        else:
            direction = "Turn around"
        
        if curr['type'] == 'building':
            instructions.append(f"{direction} at {curr['name']}")
        else:
            # Calculate distance
            distance = calculate_distance(prev['lat'], prev['lng'], curr['lat'], curr['lng'])
            instructions.append(f"{direction} and walk {int(distance)} meters")
    
    # Last instruction is to arrive at destination
    if path[-1]['type'] == 'building':
        instructions.append(f"Arrive at {path[-1]['name']}")
    else:
        instructions.append("Arrive at your destination")
    
    return instructions

def calculate_bearing(lat1, lng1, lat2, lng2):
    """Calculate the bearing between two points in degrees"""
    lat1 = math.radians(lat1)
    lng1 = math.radians(lng1)
    lat2 = math.radians(lat2)
    lng2 = math.radians(lng2)
    
    y = math.sin(lng2 - lng1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lng2 - lng1)
    bearing = math.atan2(y, x)
    
    # Convert to degrees
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing

def calculate_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two points in meters using the Haversine formula"""
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

def find_indoor_path(building, start_node, end_node):
    """
    Simple implementation of BFS to find a path between two nodes in a building.
    In a real application, you would use A* or another pathfinding algorithm.
    
    Parameters:
    building (str): Building ID
    start_node (str): Starting node ID
    end_node (str): Ending node ID
    
    Returns:
    list: List of node IDs forming a path, or None if no path exists
    """
    indoor_nodes = CAMPUS_DATA['indoor_nodes'][building]
    
    # Simple BFS implementation
    queue = [[start_node]]
    visited = set()
    
    while queue:
        path = queue.pop(0)
        node = path[-1]
        
        if node == end_node:
            return path
        
        if node in visited:
            continue
            
        visited.add(node)
        
        # Check all connections of the current node
        for connection in indoor_nodes[node]['connections']:
            if connection not in visited:
                new_path = list(path)
                new_path.append(connection)
                queue.append(new_path)
    
    return None  # No path found

def generate_path(start_lat, start_lng, end_lat, end_lng):
    """
    Generate a simple path between two points.
    This is kept for backward compatibility.
    """
    # Create temporary nodes for the start and end points
    start_node_id = f"temp_start_{start_lat}_{start_lng}"
    end_node_id = f"temp_end_{end_lat}_{end_lng}"
    
    # Create a temporary graph
    G = nx.Graph()
    
    # Add the start and end nodes
    G.add_node(start_node_id, pos=(start_lat, start_lng), type='node')
    G.add_node(end_node_id, pos=(end_lat, end_lng), type='node')
    
    # Find the nearest nodes in the existing graph
    nearest_to_start = find_nearest_node(start_lat, start_lng)
    nearest_to_end = find_nearest_node(end_lat, end_lng)
    
    # Get the positions of the nearest nodes
    for building in CAMPUS_DATA['buildings']:
        if building['id'] == nearest_to_start:
            start_nearest_lat, start_nearest_lng = building['lat'], building['lng']
            break
            
    for building in CAMPUS_DATA['buildings']:
        if building['id'] == nearest_to_end:
            end_nearest_lat, end_nearest_lng = building['lat'], building['lng']
            break
    
    # Connect the start and end nodes to their nearest nodes
    distance_to_start = calculate_distance(start_lat, start_lng, start_nearest_lat, start_nearest_lng)
    distance_to_end = calculate_distance(end_lat, end_lng, end_nearest_lat, end_nearest_lng)
    
    # Create a simple path
    path = [{"lat": start_lat, "lng": start_lng}]
    
    # Calculate intermediate points (every ~10 meters)
    num_points = max(2, int(distance_to_start / 10))
    for i in range(1, num_points):
        fraction = i / num_points
        lat = start_lat + fraction * (start_nearest_lat - start_lat)
        lng = start_lng + fraction * (start_nearest_lng - start_lng)
        path.append({"lat": lat, "lng": lng})
    
    # Add a point for each building
    path.append({"lat": start_nearest_lat, "lng": start_nearest_lng})
    path.append({"lat": end_nearest_lat, "lng": end_nearest_lng})
    
    # Add points from the nearest end node to the end
    num_points = max(2, int(distance_to_end / 10))
    for i in range(1, num_points):
        fraction = i / num_points
        lat = end_nearest_lat + fraction * (end_lat - end_nearest_lat)
        lng = end_nearest_lng + fraction * (end_lng - end_nearest_lng)
        path.append({"lat": lat, "lng": lng})
    
    path.append({"lat": end_lat, "lng": end_lng})
    
    return path

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    import sys
    
    # Check if SSL should be used (default to no SSL for easier development)
    use_ssl = '--ssl' in sys.argv
    
    if use_ssl:
        # Use SSL
        app.run(debug=True, host='0.0.0.0', ssl_context=('ssl_cert/cert.pem', 'ssl_cert/key.pem'))
    else:
        # Run without SSL for easier development
        app.run(debug=True, host='0.0.0.0') 