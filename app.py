from flask import Flask, render_template, send_from_directory, jsonify, request
import os
import json
import math

app = Flask(__name__)

# Sample data - in a real app, you would load this from a database
CAMPUS_DATA = {
    "origin": {
        "lat": 42.6686, # Replace with your campus central coordinates
        "lng": -71.1094
    },
    "buildings": [
        {
            "id": "mendel",
            "name": "Mendel Science Center",
            "lat": 42.6680,
            "lng": -71.1090,
            "description": "Home to the science departments",
            "floors": 3,
            "image": "mendel.jpg"
        },
        {
            "id": "mcquade",
            "name": "McQuade Library",
            "lat": 42.6690,
            "lng": -71.1085,
            "description": "Main campus library",
            "floors": 4,
            "image": "mcquade.jpg"
        },
        {
            "id": "sakowich",
            "name": "Sakowich Campus Center",
            "lat": 42.6685,
            "lng": -71.1089,
            "description": "Student center with dining and activities",
            "floors": 2,
            "image": "sakowich.jpg"
        }
    ],
    "rooms": {
        "mendel": [
            {"id": "m101", "name": "Mendel 101", "floor": 1, "type": "classroom"},
            {"id": "m102", "name": "Mendel 102", "floor": 1, "type": "lab"},
            {"id": "m201", "name": "Mendel 201", "floor": 2, "type": "classroom"},
            {"id": "m301", "name": "Mendel 301", "floor": 3, "type": "faculty"}
        ],
        "mcquade": [
            {"id": "mq1", "name": "First Floor", "floor": 1, "type": "study"},
            {"id": "mq2", "name": "Second Floor", "floor": 2, "type": "study"},
            {"id": "mq3", "name": "Third Floor", "floor": 3, "type": "quiet"},
            {"id": "mq4", "name": "Fourth Floor", "floor": 4, "type": "research"}
        ],
        "sakowich": [
            {"id": "s101", "name": "Dining Hall", "floor": 1, "type": "dining"},
            {"id": "s102", "name": "Bookstore", "floor": 1, "type": "shop"},
            {"id": "s201", "name": "MPR", "floor": 2, "type": "meeting"}
        ]
    },
    "paths": [
        # Simplified outdoor paths between buildings
        {"from": "campus_entrance", "to": "mendel", "nodes": [
            {"id": "n1", "lat": 42.6675, "lng": -71.1100},
            {"id": "n2", "lat": 42.6678, "lng": -71.1095},
            {"id": "n3", "lat": 42.6680, "lng": -71.1090}
        ]},
        {"from": "mendel", "to": "mcquade", "nodes": [
            {"id": "n4", "lat": 42.6680, "lng": -71.1090},
            {"id": "n5", "lat": 42.6685, "lng": -71.1088},
            {"id": "n6", "lat": 42.6690, "lng": -71.1085}
        ]},
        {"from": "mendel", "to": "sakowich", "nodes": [
            {"id": "n7", "lat": 42.6680, "lng": -71.1090},
            {"id": "n8", "lat": 42.6683, "lng": -71.1089},
            {"id": "n9", "lat": 42.6685, "lng": -71.1089}
        ]},
        {"from": "mcquade", "to": "sakowich", "nodes": [
            {"id": "n10", "lat": 42.6690, "lng": -71.1085},
            {"id": "n11", "lat": 42.6688, "lng": -71.1087},
            {"id": "n12", "lat": 42.6685, "lng": -71.1089}
        ]}
    ],
    # New: Indoor nodes for plaque-based navigation
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
    # New: Plaque data for recognition
    "plaques": {
        "M101": {"building": "mendel", "roomId": "m101", "description": "Lecture Hall"},
        "M102": {"building": "mendel", "roomId": "m102", "description": "Chemistry Lab"},
        "M201": {"building": "mendel", "roomId": "m201", "description": "Physics Lab"},
        "MQ1": {"building": "mcquade", "roomId": "mq1", "description": "Library Main Floor"}
    }
}

@app.route('/')
def index():
    return render_template('index.html')

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
    # In a real app, this would use a proper pathfinding algorithm
    path = generate_path(user_lat, user_lng, building['lat'], building['lng'])
    
    return jsonify({
        "path": path,
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
    In a real app, this would use proper pathfinding on a graph.
    """
    # For demo purposes, just create a simple straight line path
    # with points every ~10 meters
    path = []
    
    # Calculate distance
    earth_radius = 6371000  # meters
    lat1 = math.radians(start_lat)
    lng1 = math.radians(start_lng)
    lat2 = math.radians(end_lat)
    lng2 = math.radians(end_lng)
    
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = earth_radius * c
    
    # Number of points to generate
    num_points = max(2, int(distance / 10))
    
    # Generate points
    for i in range(num_points + 1):
        fraction = i / num_points
        lat = start_lat + fraction * (end_lat - start_lat)
        lng = start_lng + fraction * (end_lng - start_lng)
        path.append({"lat": lat, "lng": lng})
    
    return path

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', ssl_context=('ssl_cert/cert.pem', 'ssl_cert/key.pem')) 