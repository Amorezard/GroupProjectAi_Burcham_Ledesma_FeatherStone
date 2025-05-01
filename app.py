from flask import Flask, render_template, send_from_directory, jsonify, request, redirect, url_for
import os
import json
import math
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from path_processor import PathProcessor
import datetime

# Add a utility function to sanitize JSON data before sending it
def sanitize_json_data(data):
    """
    Clean JSON data to ensure it doesn't contain invalid values like NaN
    
    Args:
        data: Data structure to sanitize
        
    Returns:
        Sanitized data with NaN values converted to None
    """
    if isinstance(data, dict):
        return {k: sanitize_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json_data(item) for item in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    else:
        return data

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
            "lat":  42.670954,
            "lng":  -71.124795,
            "description": "Home to the Girard School of Business",
            "floors": 3,
            "image": "austin.jpg"
        },
        {
            "id": "mcquade",
            "name": "McQuade Library",
            "lat": 42.669513,
            "lng": -71.123195,
            "description": "Main campus library",
            "floors": 3,
            "image": "mcquade.jpg"
        },
        {
            "id": "mendel",
            "name": "Mendel Center",
            "lat":  42.669261,
            "lng":  -71.122032,
            "description": "Science and Engineering Center",
            "floors": 2,
            "image": "mendel.jpg"
        },
        {
            "id": "oreilly",
            "name": "O'Reilly Hall",
            "lat":  42.668862,
            "lng":  -71.123382,
            "description": "Health Sciences Center",
            "floors": 2,
            "image": "oreilly.jpg"
        },
        {
            "id": "rogers",
            "name": "Rogers Center",
            "lat":  42.668368,
            "lng":  -71.122026,
            "description": "Arts and performance venue",
            "floors": 2,
            "image": "rogers.jpg"
        },
        {
            "id": "sakowich",
            "name": "Sakowich Campus Center",
            "lat":  42.668641,
            "lng":  -71.124394,
            "description": "Student center with dining and activities",
            "floors": 3,
            "image": "sakowich.jpg"
        },
        {
            "id": "cushing",
            "name": "Cushing Hall",
            "lat":  42.669615,
            "lng":  -71.124125,
            "description": "Liberal arts classrooms",
            "floors": 2,
            "image": "cushing.jpg"
        },
        {
            "id": "sullivanHall",
            "name": "Sullivan Hall",
            "lat":  42.670211, 
            "lng":  -71.123153,
            "description": "General classrooms",
            "floors": 3,
            "image": "sullivan.jpg"
        },
        {
            "id": "arcidi",
            "name": "Arcidi Center",
            "lat":    42.670547,
            "lng":    -71.12617,
            "description": "Welcome Center",
            "floors": 2,
            "image": "arcidi.jpg"
        },
        {
            "id": "campus_entrance",
            "name": "Campus Entrance",
            "lat":  42.671528,
            "lng":  -71.126642,
            "description": "Main entrance",
            "floors": 1,
            "image": "campus_entrance.jpg"
        },
        {
            "id": "ash",
            "name": "Ash Hall",
            "lat":  42.666816,
            "lng":  -71.121648,
            "description": "Student residence",
            "floors": 3,
            "image": "ash.jpg"
        },
        {
            "id": "duane_stadium",
            "name": "Duane Stadium",
            "lat":   42.666414,
            "lng":   -71.120206,
            "description": "Athletic field",
            "floors": 1,
            "image": "duane_stadium.jpg"
        },
        {
            "id": "lawler_rink",
            "name": "Lawler Rink",
            "lat":   42.667775,
            "lng":   -71.120297,
            "description": "Ice rink",
            "floors": 1,
            "image": "lawler_rink.jpg"
        },
        {
            "id": "deegan_west",
            "name": "Deegan West",
            "lat":   42.667548,
            "lng":   -71.122179,
            "description": "Student residence",
            "floors": 3,
            "image": "deegan_west.jpg"
        },
        {
            "id": "deegan_east",
            "name": "Deegan East",
            "lat":    42.667524,
            "lng":    -71.121446,
            "description": "Student residence",
            "floors": 3,
            "image": "deegan_east.jpg"
        },
        {
            "id": "monican",
            "name": "Monican Center",
            "lat":     42.666417,
            "lng":     -71.126679,
            "description": "Student residence",
            "floors": 3,
            "image": "monican.jpg"
        },
        {
            "id": "obrien",
            "name": "O'Brien Hall",
            "lat":      42.666785,
            "lng":      -71.124803,
            "description": "Student residence",
            "floors": 3,
            "image": "obrien.jpg"
        },
        {
            "id": "cascia",
            "name": "Cascia Hall",
            "lat":       42.668298,
            "lng":       -71.123707,
            "description": "Church",
            "floors": 1,
            "image": "cascia.jpg"
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
            {"id": "n1", "lat": 42.666816, "lng": -71.121648},
            {"id": "n2", "lat": 42.667016, "lng": -71.121648},
            {"id": "n3", "lat": 42.667016, "lng": -71.121648}
        ]},
        {"from": "austin", "to": "sakowich", "nodes": [
            {"id": "n4", "lat": 42.667016, "lng": -71.121648},
            {"id": "n5", "lat": 42.667016, "lng": -71.121648},
            {"id": "n6", "lat": 42.667016, "lng": -71.121648}
        ]},
        {"from": "sakowich", "to": "rogers", "nodes": [
            {"id": "n7", "lat": 42.667016, "lng": -71.121648},
            {"id": "n8", "lat": 42.667016, "lng": -71.121648},
            {"id": "n9", "lat": 42.667016, "lng": -71.121648}
        ]},
        {"from": "rogers", "to": "oreilly", "nodes": [
            {"id": "n10", "lat": 42.667016, "lng": -71.121648},
            {"id": "n11", "lat": 42.667016, "lng": -71.121648},
            {"id": "n12", "lat": 42.667016, "lng": -71.121648}
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

@app.route('/wayfinding')
def wayfinding():
    return render_template('wayfinding.html')

@app.route('/path-editor')
def path_editor():
    return render_template('path_editor.html')

@app.route('/api/paths')
def get_paths():
    """API endpoint to get all campus paths"""
    return jsonify(sanitize_json_data(CAMPUS_DATA['paths']))

@app.route('/api/save-paths', methods=['POST'])
def save_paths():
    """API endpoint to save updated paths"""
    if request.method == 'POST':
        paths_data = request.json
        
        if not paths_data:
            return jsonify({"error": "No path data provided"}), 400
        
        # In a real app, you would validate the data here
        
        # Sanitize before saving
        clean_paths = sanitize_json_data(paths_data)
        
        # Save to an external JSON file for persistence
        with open('path.json', 'w') as f:
            json.dump(clean_paths, f, indent=2)
        
        # Update the current application data
        global CAMPUS_DATA
        CAMPUS_DATA['paths'] = clean_paths
        
        return jsonify({"success": True, "message": "Paths saved successfully"})
    
    return jsonify({"error": "Invalid request method"}), 405

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
    
    # Use the improved path algorithm
    path_data = calculate_improved_path(start_node, end_node, preferences)
    
    # Generate navigation instructions
    instructions = generate_instructions(path_data['path'])
    
    # Add stats to the response
    response = {
        'path': path_data['path'],
        'instructions': instructions,
        'stats': path_data['stats']
    }
    
    return jsonify(response)

def calculate_improved_path(start_id, end_id, preferences=None, time_of_day=None, weather=None):
    """Advanced A* pathfinding with dynamic heuristics"""
    # Default preferences if none provided
    if preferences is None:
        preferences = {}
    
    # Create a graph with user preferences
    G = create_graph_with_preferences(preferences)
    
    # Check if nodes exist in the graph
    if start_id not in G.nodes or end_id not in G.nodes:
        print(f"Error: Start ({start_id}) or end ({end_id}) node not found in graph")
        return {'path': [], 'segments': [], 'stats': {'error': 'Node not found'}}
    
    try:
        # Dynamic heuristic that considers multiple factors
        def advanced_heuristic(u, v):
            # Base distance heuristic (straight-line distance)
            u_pos = G.nodes[u]['pos']
            v_pos = G.nodes[v]['pos']
            base_distance = calculate_distance(u_pos[0], u_pos[1], v_pos[0], v_pos[1]) * 1000  # Convert to meters
            
            # Apply time-of-day factor (e.g., avoid crowded areas during class changes)
            time_factor = 1.0
            if time_of_day == "class_change" and G.nodes[u].get('type') == 'building':
                # Buildings are more crowded during class changes
                time_factor = 1.2
            
            # Apply weather factor (e.g., prefer covered paths in rain)
            weather_factor = 1.0
            if weather == "rain" and G.edges.get((u, v), {}).get('path_type') != 'covered':
                weather_factor = 1.3
            
            # Encourage visiting nodes closest to the destination
            return base_distance * time_factor * weather_factor
        
        # Find shortest path using A*
        shortest_path = nx.astar_path(G, start_id, end_id, heuristic=advanced_heuristic, weight='weight')
        
        # Calculate total distance and other stats
        total_distance = 0
        path_segments = []
        
        for i in range(len(shortest_path) - 1):
            u = shortest_path[i]
            v = shortest_path[i + 1]
            
            segment_data = {
                'from_id': u,
                'to_id': v,
                'from_type': G.nodes[u].get('type', 'node'),
                'to_type': G.nodes[v].get('type', 'node')
            }
            
            # Add segment properties
            if 'weight' in G[u][v]:
                segment_data['weight'] = G[u][v]['weight']
            
            if 'distance' in G[u][v]:
                segment_data['distance'] = G[u][v]['distance']
                total_distance += G[u][v]['distance']
            
            if 'path_type' in G[u][v]:
                segment_data['path_type'] = G[u][v]['path_type']
            
            path_segments.append(segment_data)
        
        # Convert to lat/lng points for mapping
        path_points = []
        for node_id in shortest_path:
            node = G.nodes[node_id]
            if 'pos' in node:
                lat, lng = node['pos']
                node_data = {
                    'id': node_id,
                    'lat': lat,
                    'lng': lng,
                    'type': node.get('type', 'node')
                }
                
                # Add name if it's a building
                if 'name' in node:
                    node_data['name'] = node['name']
                
                if 'path_type' in node:
                    node_data['path_type'] = node['path_type']
                
                path_points.append(node_data)
        
        # Generate path statistics
        stats = {
            'total_distance': total_distance,
            'num_segments': len(path_segments),
            'num_buildings': sum(1 for p in path_points if p.get('type') == 'building'),
            'num_nodes': len(path_points),
            'estimated_time': int(total_distance / 83.3),  # 5 km/h walking speed = 83.3 m/min
            'path_types': {}
        }
        
        # Count different path types used
        for seg in path_segments:
            path_type = seg.get('path_type', 'unknown')
            if path_type in stats['path_types']:
                stats['path_types'][path_type] += 1
            else:
                stats['path_types'][path_type] = 1
        
        print(f"A* improved path found with {len(path_points)} nodes and distance: {total_distance:.1f} meters")
        
        return {
            'path': path_points,
            'segments': path_segments,
            'stats': stats
        }
        
    except (nx.NetworkXNoPath, KeyError) as e:
        print(f"Error finding path with preferences: {e}")
        return {
            'path': [],
            'segments': [],
            'stats': {
                'error': str(e),
                'total_distance': 0
            }
        }

def create_graph_with_preferences(preferences):
    """Create a graph incorporating user preferences"""
    G = nx.Graph()
    
    # Add buildings as nodes
    for building in CAMPUS_DATA['buildings']:
        G.add_node(
            building['id'],
            pos=(building['lat'], building['lng']),
            type='building',
            name=building['name'],
            description=building.get('description', '')
        )
    
    # Keep track of all nodes to connect grid nodes to nearest buildings later
    all_nodes = {}
    for building in CAMPUS_DATA['buildings']:
        all_nodes[building['id']] = (building['lat'], building['lng'])
    
    # Add path nodes and edges, considering preferences
    grid_nodes = set()  # Track grid nodes for post-processing
    
    for path in CAMPUS_DATA['paths']:
        prev_node = None
        path_type = path.get('type', 'sidewalk')
        
        # Special handling for grid connections
        is_grid = path_type == 'grid_connection'
        
        for node in path['nodes']:
            node_id = node['id']
            is_grid_node = node_id.startswith('grid_')
            
            # Add to tracking set if it's a grid node
            if is_grid_node:
                grid_nodes.add(node_id)
            
            # Add node if it doesn't exist
            if node_id not in G:
                G.add_node(
                    node_id,
                    pos=(node['lat'], node['lng']),
                    type='grid_node' if is_grid_node else 'path_node',
                    path_type=path_type
                )
                all_nodes[node_id] = (node['lat'], node['lng'])
            
            if prev_node:
                # Calculate actual distance between nodes
                lat1, lng1 = G.nodes[prev_node]['pos']
                lat2, lng2 = G.nodes[node_id]['pos']
                distance = calculate_distance(lat1, lng1, lat2, lng2)
                
                # Apply preference weights
                weight = distance
                
                # Grid connections should have lower weight by default to encourage their use
                if is_grid:
                    weight *= 0.9  # 10% discount for grid connections
                
                # Apply path type modifications
                if path_type == 'stairs' and preferences.get('avoidStairs', False):
                    weight *= 5  # Make stairs much less desirable
                
                elif path_type == 'road' and preferences.get('preferSafeRoutes', False):
                    weight *= 2  # Prefer sidewalks over roads
                
                elif path_type == 'shortcut' and preferences.get('preferOfficial', False):
                    weight *= 3  # Avoid unofficial shortcuts
                
                # If needing wheelchair access, avoid non-accessible paths
                if preferences.get('wheelchairAccessible', False) and path_type not in ['accessible', 'sidewalk', 'grid_connection']:
                    weight *= 10
                
                # Add the edge with appropriate weight
                G.add_edge(prev_node, node_id, 
                          weight=weight, 
                          distance=distance,
                          path_type=path_type)
            
            prev_node = node_id
        
        # Connect starting and ending buildings if specified
        if path.get('from') and path.get('to') and not is_grid:  # Skip for grid paths
            # Connect starting building to first node
            if path['from'] in G.nodes and path['nodes']:
                first_node = path['nodes'][0]['id']
                from_building = path['from']
                if from_building in G.nodes and first_node in G.nodes:
                    lat1, lng1 = G.nodes[from_building]['pos']
                    lat2, lng2 = G.nodes[first_node]['pos']
                    distance = calculate_distance(lat1, lng1, lat2, lng2)
                    G.add_edge(from_building, first_node, 
                              weight=distance, 
                              distance=distance,
                              path_type=path_type)
            
            # Connect last node to destination building
            if path['to'] in G.nodes and path['nodes']:
                last_node = path['nodes'][-1]['id']
                to_building = path['to']
                if to_building in G.nodes and last_node in G.nodes:
                    lat1, lng1 = G.nodes[last_node]['pos']
                    lat2, lng2 = G.nodes[to_building]['pos']
                    distance = calculate_distance(lat1, lng1, lat2, lng2)
                    G.add_edge(last_node, to_building, 
                               weight=distance, 
                               distance=distance,
                               path_type=path_type)
    
    # Post-processing: Connect grid nodes to nearby buildings and path nodes
    # This ensures the grid is integrated with the rest of the navigation system
    connect_grid_to_buildings(G, grid_nodes, all_nodes, preferences)
    
    return G

def connect_grid_to_buildings(G, grid_nodes, all_nodes, preferences):
    """Connect grid nodes to nearby buildings to ensure the grid is usable for navigation"""
    MAX_CONNECTION_DISTANCE = 50  # Maximum distance in meters to connect a grid node to a building
    
    # Create spatial index for efficient nearest neighbor search
    node_points = []
    node_ids = []
    
    # Add all non-grid nodes to the spatial index
    for node_id, coords in all_nodes.items():
        if node_id not in grid_nodes and node_id in G:
            node_points.append([coords[0], coords[1]])
            node_ids.append(node_id)
    
    # If we have non-grid nodes to connect to
    if node_points:
        # Create nearest neighbor model
        node_points_array = np.array(node_points)
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(node_points_array)
        
        # For each grid node, find the nearest non-grid nodes
        for grid_node in grid_nodes:
            if grid_node in G:
                grid_pos = G.nodes[grid_node]['pos']
                
                # Find k-nearest neighbors
                distances, indices = nbrs.kneighbors([[grid_pos[0], grid_pos[1]]])
                
                # Connect to nearest neighbors within maximum distance
                for i, idx in enumerate(indices[0]):
                    neighbor_id = node_ids[idx]
                    distance = distances[0][i] * 1000  # Convert to meters
                    
                    if distance <= MAX_CONNECTION_DISTANCE:
                        # Is this a building or a path node?
                        is_building = G.nodes[neighbor_id].get('type') == 'building'
                        
                        # Set path type based on the node types
                        path_type = 'grid_to_building' if is_building else 'grid_connection'
                        
                        # Add edge with appropriate weight
                        G.add_edge(grid_node, neighbor_id,
                                  weight=distance * (0.8 if is_building else 0.9),  # Incentivize building connections
                                  distance=distance,
                                  path_type=path_type)

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

@app.route('/api/grid-nodes')
def get_grid_nodes():
    """API endpoint to get all grid nodes for visualization"""
    grid_nodes = []
    grid_connections = []
    
    # Extract grid nodes from paths
    for path in CAMPUS_DATA['paths']:
        if path.get('type') == 'grid_connection':
            for node in path['nodes']:
                if node['id'].startswith('grid_'):
                    grid_nodes.append(node)
            
            # Add connection info if there are exactly 2 nodes (typical grid connection)
            if len(path['nodes']) == 2 and path['nodes'][0]['id'].startswith('grid_') and path['nodes'][1]['id'].startswith('grid_'):
                grid_connections.append({
                    'from': path['nodes'][0]['id'],
                    'to': path['nodes'][1]['id'],
                    'from_pos': {'lat': path['nodes'][0]['lat'], 'lng': path['nodes'][0]['lng']},
                    'to_pos': {'lat': path['nodes'][1]['lat'], 'lng': path['nodes'][1]['lng']}
                })
    
    return jsonify({
        'nodes': grid_nodes,
        'connections': grid_connections
    })

@app.route('/api/smart-path-generate', methods=['POST'])
def generate_smart_path():
    """Generate an intelligent path between two buildings or points"""
    data = request.json
    from_id = data.get('from')
    to_id = data.get('to')
    resolution = int(data.get('resolution', 3))  # Number of intermediate points (min 1, max 10)
    path_type = data.get('path_type', 'sidewalk')
    
    # Validate resolution
    resolution = max(1, min(10, resolution))
    
    # Get coordinates for the endpoints
    from_pos = None
    to_pos = None
    
    # Check if these are building IDs
    for building in CAMPUS_DATA['buildings']:
        if building['id'] == from_id:
            from_pos = (building['lat'], building['lng'])
        if building['id'] == to_id:
            to_pos = (building['lat'], building['lng'])
    
    # Check if these are custom coordinates
    if not from_pos and 'lat' in data.get('fromPos', {}):
        from_pos = (data['fromPos']['lat'], data['fromPos']['lng'])
    if not to_pos and 'lat' in data.get('toPos', {}):
        to_pos = (data['toPos']['lat'], data['toPos']['lng'])
    
    if not from_pos or not to_pos:
        return jsonify({"error": "Invalid start or end location"}), 400
    
    # Generate smart path based on existing pathways
    smart_path = generate_intelligent_path(from_id, to_id, from_pos, to_pos, resolution, path_type)
    
    return jsonify(smart_path)

def generate_intelligent_path(from_id, to_id, from_pos, to_pos, resolution, path_type):
    """
    Generate an intelligent path between two points incorporating:
    1. Terrain awareness (using existing paths where possible)
    2. Building entrance detection
    3. Natural path curvature
    4. Obstacle avoidance
    """
    # Begin with A* search for preferred route using existing paths
    G = create_graph_with_preferences({})
    
    # Find nearest nodes to starting and ending points
    start_node = find_nearest_node(from_pos[0], from_pos[1])
    end_node = find_nearest_node(to_pos[0], to_pos[1])
    
    # Attempt to find an existing path
    intermediate_points = []
    
    try:
        # Try to find an existing path using A*
        if start_node and end_node and start_node in G.nodes and end_node in G.nodes:
            def heuristic(u, v):
                u_pos = G.nodes[u]['pos']
                v_pos = G.nodes[v]['pos']
                return calculate_distance(u_pos[0], u_pos[1], v_pos[0], v_pos[1])
            
            # Get shortest path
            shortest_path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight='weight')
            
            # Extract points from shortest path
            path_points = []
            for node_id in shortest_path:
                if node_id in G.nodes and 'pos' in G.nodes[node_id]:
                    lat, lng = G.nodes[node_id]['pos']
                    path_points.append((lat, lng))
            
            # Use path points as guide for intelligent path
            if len(path_points) >= 2:
                # Simplify path if too complex (get key points)
                if len(path_points) > resolution + 2:
                    step = len(path_points) // (resolution + 1)
                    intermediate_points = [path_points[i] for i in range(step, len(path_points) - 1, step)]
                    # Ensure we don't have too many points
                    intermediate_points = intermediate_points[:resolution]
                else:
                    # Remove first and last points (these are our endpoints)
                    intermediate_points = path_points[1:-1]
    except Exception as e:
        print(f"Error finding existing path: {e}")
        # We'll generate a new path if this fails
    
    # If no path found or not enough intermediate points, generate a natural path
    if len(intermediate_points) < resolution:
        # Generate more points using natural path algorithm with bezier curves
        additional_points_needed = resolution - len(intermediate_points)
        
        if additional_points_needed > 0:
            # Use bezier curve to create a natural curved path
            if len(intermediate_points) == 0:
                # Create a natural curve with a slight deviation
                # Calculate midpoint with slight offset for natural curve
                mid_lat = (from_pos[0] + to_pos[0]) / 2
                mid_lng = (from_pos[1] + to_pos[1]) / 2
                
                # Calculate perpendicular vector for offset
                dx = to_pos[0] - from_pos[0]
                dy = to_pos[1] - from_pos[1]
                distance = ((dx**2) + (dy**2))**0.5
                
                # Perpendicular offset scaled by distance
                perpendicular_scale = 0.15 * distance  # 15% of path length
                offset_lat = -dy/distance * perpendicular_scale
                offset_lng = dx/distance * perpendicular_scale
                
                # Add control point
                control_point = (mid_lat + offset_lat, mid_lng + offset_lng)
                control_points = [control_point]
                
                # For longer paths, add more control points
                if resolution >= 3:
                    # Add another control point on the other side for S-curve
                    quarter_point = (from_pos[0]*0.75 + to_pos[0]*0.25, from_pos[1]*0.75 + to_pos[1]*0.25)
                    three_quarter_point = (from_pos[0]*0.25 + to_pos[0]*0.75, from_pos[1]*0.25 + to_pos[1]*0.75)
                    
                    control_points = [
                        (quarter_point[0] - offset_lat*0.7, quarter_point[1] - offset_lng*0.7),
                        (three_quarter_point[0] + offset_lat*0.7, three_quarter_point[1] + offset_lng*0.7)
                    ]
            else:
                # Use existing points as control points and add additional ones
                control_points = intermediate_points
            
            # Generate points using bezier curve
            new_points = generate_bezier_path(from_pos, to_pos, control_points, additional_points_needed + 2)
            
            # Skip first and last point (they're our endpoints)
            new_intermediate_points = new_points[1:-1]
            
            # Combine with existing intermediate points
            intermediate_points.extend(new_intermediate_points)
    
    # Create node IDs and full path structure
    nodes = []
    
    # Start with first node
    start_node_id = f"auto_{from_id}_to_{to_id}_0"
    nodes.append({
        "id": start_node_id,
        "lat": from_pos[0],
        "lng": from_pos[1]
    })
    
    # Add intermediate nodes
    for i, point in enumerate(intermediate_points):
        node_id = f"auto_{from_id}_to_{to_id}_{i+1}"
        nodes.append({
            "id": node_id,
            "lat": point[0],
            "lng": point[1]
        })
    
    # Add final node
    end_node_id = f"auto_{from_id}_to_{to_id}_{len(intermediate_points)+1}"
    nodes.append({
        "id": end_node_id,
        "lat": to_pos[0],
        "lng": to_pos[1]
    })
    
    # Create complete path
    path = {
        "from": from_id,
        "to": to_id,
        "type": path_type,
        "nodes": nodes
    }
    
    return path

def generate_bezier_path(start, end, control_points, num_points):
    """
    Generate a smooth path using bezier curve algorithm
    
    Parameters:
    start: (lat, lng) for start point
    end: (lat, lng) for end point
    control_points: List of (lat, lng) tuples for control points
    num_points: Number of points to generate on the curve
    
    Returns:
    List of (lat, lng) tuples representing points on the bezier curve
    """
    # Create a list of all points including start, control points, and end
    all_points = [start] + control_points + [end]
    
    # Generate bezier curve points
    result = []
    
    for i in range(num_points):
        t = i / (num_points - 1)
        point = bezier_point(all_points, t)
        result.append(point)
    
    return result

def bezier_point(points, t):
    """Calculate point on a bezier curve at parameter t"""
    if len(points) == 1:
        return points[0]
    
    new_points = []
    for i in range(len(points) - 1):
        lat = (1 - t) * points[i][0] + t * points[i + 1][0]
        lng = (1 - t) * points[i][1] + t * points[i + 1][1]
        new_points.append((lat, lng))
    
    return bezier_point(new_points, t)

@app.route('/api/analyze-paths', methods=['POST'])
def analyze_paths():
    """
    Analyze existing paths and detect their types automatically
    """
    global CAMPUS_DATA
    
    # Initialize path processor with campus data
    processor = PathProcessor(CAMPUS_DATA)
    
    # Analyze paths to detect types
    updated_paths = processor.analyze_path_types()
    
    # Update the current application data
    CAMPUS_DATA['paths'] = updated_paths
    
    # Save to file for persistence
    with open('path.json', 'w') as f:
        json.dump(updated_paths, f, indent=2)
    
    return jsonify({
        "success": True, 
        "message": "Paths analyzed and updated",
        "paths": updated_paths
    })

@app.route('/api/prioritize-paths', methods=['POST'])
def prioritize_paths():
    """
    Prioritize paths for wayfinding
    """
    global CAMPUS_DATA
    
    # Initialize path processor with campus data
    processor = PathProcessor(CAMPUS_DATA)
    
    # Prioritize paths
    prioritized_paths = processor.prioritize_paths(CAMPUS_DATA['paths'])
    
    # Update the current application data with priority values
    CAMPUS_DATA['paths'] = prioritized_paths
    
    # Save to file for persistence
    with open('path.json', 'w') as f:
        json.dump(prioritized_paths, f, indent=2)
    
    return jsonify({
        "success": True, 
        "message": "Paths prioritized",
        "paths": prioritized_paths
    })

@app.route('/api/suggest-connections', methods=['GET'])
def suggest_connections():
    """
    Identify missing connections between buildings
    """
    # Initialize path processor with campus data
    processor = PathProcessor(CAMPUS_DATA)
    
    # Find missing connections
    suggested_paths = processor.identify_missing_connections()
    
    return jsonify({
        "success": True,
        "suggested_paths": suggested_paths,
        "count": len(suggested_paths)
    })

@app.route('/api/smart-paths', methods=['POST'])
def generate_smart_paths():
    """Generate smart paths with terrain following"""
    if request.method == 'POST':
        data = request.json
        include_terrain = data.get('include_terrain', True)
        
        # Create path processor
        processor = PathProcessor(CAMPUS_DATA)
        
        # Load OSM data
        try:
            processor.load_osm_data()
        except Exception as e:
            print(f"Error loading OSM data: {e}")
            # Continue without OSM data if it fails
        
        # Generate grid paths
        grid_paths = processor.generate_grid_paths(resolution=10)
        
        # Combine with existing paths and analyze
        all_paths = CAMPUS_DATA['paths'] + grid_paths
        
        # Update campus data temporarily
        CAMPUS_DATA_COPY = CAMPUS_DATA.copy()
        CAMPUS_DATA_COPY['paths'] = all_paths
        
        # Create new processor with updated data
        advanced_processor = PathProcessor(CAMPUS_DATA_COPY)
        
        # Create graph with preferences
        advanced_processor.create_path_graph_with_osm(include_existing=True, include_terrain=include_terrain)
        
        # Analyze path types
        updated_paths = advanced_processor.analyze_path_types()
        
        # Prioritize paths
        prioritized_paths = advanced_processor.prioritize_paths(updated_paths)
        
        # Find suggested connections
        suggested_paths = advanced_processor.identify_missing_connections()
        
        # Sanitize data to remove any NaN values
        clean_prioritized_paths = sanitize_json_data(prioritized_paths)
        clean_suggested_paths = sanitize_json_data(suggested_paths)
        
        # Record stats
        stats = {
            'prioritized_count': len(prioritized_paths),
            'suggested_count': len(suggested_paths),
            'total_paths': len(prioritized_paths) + len(suggested_paths)
        }
        
        # Return the updated paths
        return jsonify({
            'success': True,
            'prioritized_paths': clean_prioritized_paths,
            'suggested_paths': clean_suggested_paths,
            'stats': stats
        })
    
    return jsonify({'success': False, 'message': 'Invalid request method'}), 405

@app.route('/api/integrate-osm', methods=['POST'])
def integrate_osm_data():
    """
    Integrate OpenStreetMap data with existing campus paths
    """
    global CAMPUS_DATA
    
    # Initialize path processor
    processor = PathProcessor(CAMPUS_DATA)
    
    # Load OpenStreetMap data
    processor.load_osm_data()
    
    # Extract and integrate OSM paths
    combined_paths = processor.integrate_osm_paths()
    
    # Update campus data
    CAMPUS_DATA['paths'] = combined_paths
    
    # Save to file for persistence
    with open('path.json', 'w') as f:
        json.dump(combined_paths, f, indent=2)
    
    # Return path statistics
    path_types = {}
    for path in combined_paths:
        path_type = path.get('type', 'unknown')
        path_types[path_type] = path_types.get(path_type, 0) + 1
    
    return jsonify({
        "success": True,
        "message": "OpenStreetMap data integrated successfully",
        "stats": {
            "total_paths": len(combined_paths),
            "osm_paths": len([p for p in combined_paths if 'osm_id' in p]),
            "path_types": path_types
        }
    })

@app.route('/api/find-natural-path', methods=['POST'])
def find_natural_path():
    """Find a natural path between two points using OSM data"""
    if request.method == 'POST':
        data = request.json
        
        # Extract start and end coordinates
        start = data.get('start', {})
        end = data.get('end', {})
        preferences = data.get('preferences', {})
        
        if not all(k in start for k in ['lat', 'lng']) or not all(k in end for k in ['lat', 'lng']):
            return jsonify({'success': False, 'message': 'Invalid coordinates'}), 400
        
        # Create path processor
        processor = PathProcessor(CAMPUS_DATA)
        
        # Try to load OSM data if available
        osm_available = False
        if preferences.get('useOSM', True):
            try:
                processor.load_osm_data()
                osm_available = processor.osm_graph is not None
            except Exception as e:
                print(f"Error loading OSM data: {e}")
        
        # Find path using OpenStreetMap data if available
        if osm_available:
            path_points = processor.find_path_on_osm(
                start['lat'], start['lng'],
                end['lat'], end['lng']
            )
            
            # If OSM path was found
            if path_points and len(path_points) >= 2:
                # Generate navigation instructions
                instructions = generate_instructions(path_points)
                
                # Calculate stats
                total_distance = 0
                for i in range(len(path_points) - 1):
                    p1 = path_points[i]
                    p2 = path_points[i+1]
                    total_distance += calculate_distance(p1['lat'], p1['lng'], p2['lat'], p2['lng'])
                
                stats = {
                    'total_distance': total_distance,
                    'num_segments': len(path_points) - 1,
                    'estimated_time': int(total_distance / 83.3),  # 5 km/h walking speed
                    'path_source': 'openstreetmap'
                }
                
                # Sanitize the data before returning
                clean_path = sanitize_json_data(path_points)
                clean_instructions = sanitize_json_data(instructions)
                clean_stats = sanitize_json_data(stats)
                
                return jsonify({
                    'success': True,
                    'path': clean_path,
                    'instructions': clean_instructions,
                    'stats': clean_stats
                })
        
        # If OSM path wasn't found or OSM is not enabled, fall back to custom path
        print("Falling back to campus-based path")
        
        # Try to find path with campus data
        try:
            # Create graph with terrain consideration if requested
            processor.create_path_graph(include_terrain=preferences.get('followTerrain', True))
            
            # Find nearest nodes to start and end
            start_node = processor.find_nearest_node(start['lat'], start['lng'])
            end_node = processor.find_nearest_node(end['lat'], end['lng'])
            
            if start_node and end_node:
                # Get path based on graph
                path_points = []
                
                try:
                    path = nx.shortest_path(processor.graph, start_node, end_node, weight='weight')
                    
                    for node_id in path:
                        node = processor.graph.nodes[node_id]
                        
                        if 'pos' in node:
                            lat, lng = node['pos']
                            path_points.append({
                                'id': node_id,
                                'lat': lat,
                                'lng': lng,
                                'type': node.get('type', 'path_node')
                            })
                except nx.NetworkXNoPath:
                    print(f"No path found between {start_node} and {end_node}")
                    # Create a simple direct path instead
                    path_points = [
                        {'id': 'start', 'lat': start['lat'], 'lng': start['lng'], 'type': 'custom'},
                        {'id': 'end', 'lat': end['lat'], 'lng': end['lng'], 'type': 'custom'}
                    ]
                
                # Generate instructions
                instructions = generate_instructions(path_points)
                
                # Calculate stats
                total_distance = 0
                for i in range(len(path_points) - 1):
                    p1 = path_points[i]
                    p2 = path_points[i+1]
                    total_distance += calculate_distance(p1['lat'], p1['lng'], p2['lat'], p2['lng'])
                
                stats = {
                    'total_distance': total_distance,
                    'num_segments': len(path_points) - 1,
                    'estimated_time': int(total_distance / 83.3),
                    'path_source': 'campus_data'
                }
                
                # Sanitize the data before returning
                clean_path = sanitize_json_data(path_points)
                clean_instructions = sanitize_json_data(instructions)
                clean_stats = sanitize_json_data(stats)
                
                return jsonify({
                    'success': True,
                    'path': clean_path,
                    'instructions': clean_instructions,
                    'stats': clean_stats
                })
        except Exception as e:
            print(f"Error finding campus path: {e}")
        
        # If all else fails, create a simple direct path
        path_points = [
            {'id': 'start', 'lat': start['lat'], 'lng': start['lng'], 'type': 'custom'},
            {'id': 'end', 'lat': end['lat'], 'lng': end['lng'], 'type': 'custom'}
        ]
        
        instructions = ["Start at your location", "Go directly to destination"]
        
        stats = {
            'total_distance': calculate_distance(start['lat'], start['lng'], end['lat'], end['lng']),
            'num_segments': 1,
            'estimated_time': int(calculate_distance(start['lat'], start['lng'], end['lat'], end['lng']) / 83.3),
            'path_source': 'direct'
        }
        
        # Sanitize the data before returning
        clean_path = sanitize_json_data(path_points)
        clean_instructions = sanitize_json_data(instructions)
        clean_stats = sanitize_json_data(stats)
        
        return jsonify({
            'success': True,
            'path': clean_path,
            'instructions': clean_instructions,
            'stats': clean_stats
        })
    
    return jsonify({'success': False, 'message': 'Invalid request method'}), 405

@app.route('/api/test-osm-path', methods=['POST'])
def test_osm_path():
    """
    Test endpoint to find and visualize a path between two points using OSM data
    This is a debugging endpoint that returns both path points and a visualization URL
    """
    data = request.json
    
    if not data or 'start' not in data or 'end' not in data:
        return jsonify({"error": "Missing start or end location"}), 400
        
    start = data['start']
    end = data['end']
    
    # Validate coordinates
    if 'lat' not in start or 'lng' not in start or 'lat' not in end or 'lng' not in end:
        return jsonify({"error": "Invalid coordinates format"}), 400
    
    # Initialize PathProcessor
    processor = PathProcessor(CAMPUS_DATA)
    
    # Ensure static/debug directory exists
    os.makedirs('static/debug', exist_ok=True)
    
    # Generate a unique filename for this visualization
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    visualization_path = f"static/debug/osm_path_{timestamp}.png"
    
    # Find and visualize the path
    path_points, vis_path = processor.find_path_on_osm_with_visualization(
        start['lat'], start['lng'],
        end['lat'], end['lng'],
        save_path=visualization_path
    )
    
    # Generate a full URL to the visualization if it exists
    vis_url = None
    if vis_path:
        vis_url = request.host_url.rstrip('/') + '/' + vis_path
    
    return jsonify({
        "success": bool(path_points),
        "path": path_points,
        "visualization_url": vis_url,
        "message": f"Found path with {len(path_points)} points" if path_points else "No path found"
    })

@app.route('/api/visualize-osm-graph', methods=['GET'])
def visualize_osm_graph():
    """
    Generate and return a visualization of the current OSM graph
    """
    # Initialize PathProcessor
    processor = PathProcessor(CAMPUS_DATA)
    
    # Ensure static/debug directory exists
    os.makedirs('static/debug', exist_ok=True)
    
    # Generate a unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    visualization_path = f"static/debug/osm_graph_{timestamp}.png"
    
    # Load OSM data if not already loaded
    if processor.osm_graph is None:
        processor.load_osm_data()
    
    # Generate visualization
    processor.visualize_osm_graph(save_path=visualization_path)
    
    # Generate a full URL to the visualization
    vis_url = request.host_url.rstrip('/') + '/' + visualization_path
    
    return jsonify({
        "success": True,
        "visualization_url": vis_url,
        "message": "OSM graph visualization generated"
    })

@app.route('/osm-test')
def osm_test():
    """
    Simple page to test OSM path visualization
    """
    return render_template('osm_test.html', title="OSM Path Test")

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