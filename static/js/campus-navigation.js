/**
 * Campus AR Navigation System
 * Core functionality for AR pathfinding
 */

// A* Pathfinding algorithm for calculating optimal paths
class PathFinder {
    constructor(graph) {
        this.graph = graph;
    }
    
    // Calculate distance between two points
    distance(pointA, pointB) {
        const dx = pointA.x - pointB.x;
        const dy = pointA.y - pointB.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    // Find the shortest path using A* algorithm
    findPath(startId, endId) {
        const graph = this.graph;
        const start = graph.nodes[startId];
        const end = graph.nodes[endId];
        
        if (!start || !end) {
            console.error('Invalid start or end node');
            return null;
        }
        
        // Initialize data structures
        const openSet = [start];
        const closedSet = new Set();
        const cameFrom = {};
        
        // Cost from start to current node
        const gScore = {};
        graph.nodeIds.forEach(id => gScore[id] = Infinity);
        gScore[startId] = 0;
        
        // Estimated total cost from start to goal through current node
        const fScore = {};
        graph.nodeIds.forEach(id => fScore[id] = Infinity);
        fScore[startId] = this.distance(start, end);
        
        while (openSet.length > 0) {
            // Find node with lowest fScore
            let current = openSet[0];
            let lowestIndex = 0;
            
            for (let i = 0; i < openSet.length; i++) {
                if (fScore[openSet[i].id] < fScore[current.id]) {
                    current = openSet[i];
                    lowestIndex = i;
                }
            }
            
            // If we've reached the goal, reconstruct and return the path
            if (current.id === endId) {
                let path = [current];
                while (cameFrom[current.id]) {
                    current = cameFrom[current.id];
                    path.unshift(current);
                }
                return path;
            }
            
            // Remove current from openSet and add to closedSet
            openSet.splice(lowestIndex, 1);
            closedSet.add(current.id);
            
            // Check all neighbors
            for (const neighborId of graph.edges[current.id] || []) {
                const neighbor = graph.nodes[neighborId];
                
                // Skip if already evaluated
                if (closedSet.has(neighborId)) continue;
                
                // Calculate tentative gScore
                const tentativeGScore = gScore[current.id] + this.distance(current, neighbor);
                
                // Add neighbor to openSet if not there
                if (!openSet.some(node => node.id === neighborId)) {
                    openSet.push(neighbor);
                } else if (tentativeGScore >= gScore[neighborId]) {
                    // Not a better path
                    continue;
                }
                
                // This is the best path so far
                cameFrom[neighborId] = current;
                gScore[neighborId] = tentativeGScore;
                fScore[neighborId] = gScore[neighborId] + this.distance(neighbor, end);
            }
        }
        
        // No path found
        return null;
    }
}

// Navigation Graph representing the campus
class NavigationGraph {
    constructor() {
        this.nodes = {}; // Map of node ID to node object
        this.edges = {}; // Map of node ID to array of connected node IDs
        this.nodeIds = []; // Array of all node IDs
    }
    
    // Add a node to the graph
    addNode(id, x, y, z = 0, metadata = {}) {
        this.nodes[id] = { id, x, y, z, ...metadata };
        this.edges[id] = [];
        this.nodeIds.push(id);
        return this;
    }
    
    // Add an edge between two nodes
    addEdge(fromId, toId, bidirectional = true) {
        if (!this.nodes[fromId] || !this.nodes[toId]) {
            console.error('Cannot add edge for non-existent nodes');
            return this;
        }
        
        if (!this.edges[fromId].includes(toId)) {
            this.edges[fromId].push(toId);
        }
        
        if (bidirectional && !this.edges[toId].includes(fromId)) {
            this.edges[toId].push(fromId);
        }
        
        return this;
    }
    
    // Find the nearest node to given coordinates
    findNearestNode(x, y, z = 0) {
        let nearestNode = null;
        let minDistance = Infinity;
        
        for (const id of this.nodeIds) {
            const node = this.nodes[id];
            const dx = node.x - x;
            const dy = node.y - y;
            const dz = node.z - z;
            const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
            
            if (distance < minDistance) {
                minDistance = distance;
                nearestNode = node;
            }
        }
        
        return nearestNode;
    }
    
    // Convert latitude/longitude to x,y coordinates
    latLngToXY(lat, lng, originLat, originLng) {
        // Simple approximation, for more accuracy use proper projection
        const earthRadius = 6371000; // meters
        const dLat = (lat - originLat) * (Math.PI / 180);
        const dLng = (lng - originLng) * (Math.PI / 180);
        
        const x = earthRadius * dLng * Math.cos(originLat * (Math.PI / 180));
        const y = earthRadius * dLat;
        
        return { x, y };
    }
    
    // Convert x,y coordinates to latitude/longitude
    xyToLatLng(x, y, originLat, originLng) {
        // Simple approximation, for more accuracy use proper projection
        const earthRadius = 6371000; // meters
        
        const latChange = y / earthRadius;
        const lngChange = x / (earthRadius * Math.cos(originLat * (Math.PI / 180)));
        
        const lat = originLat + (latChange * (180 / Math.PI));
        const lng = originLng + (lngChange * (180 / Math.PI));
        
        return { lat, lng };
    }
}

// AR marker management for indoor navigation points
class ARMarkerManager {
    constructor(scene) {
        this.scene = scene;
        this.markers = {};
    }
    
    // Create a new AR marker
    createMarker(id, position, type = 'waypoint', metadata = {}) {
        // Remove any existing marker with this ID
        if (this.markers[id]) {
            this.removeMarker(id);
        }
        
        // Create marker based on type
        let markerEntity;
        
        switch (type) {
            case 'waypoint':
                markerEntity = document.createElement('a-sphere');
                markerEntity.setAttribute('color', '#0066cc');
                markerEntity.setAttribute('radius', '0.3');
                markerEntity.setAttribute('opacity', '0.7');
                break;
                
            case 'destination':
                markerEntity = document.createElement('a-box');
                markerEntity.setAttribute('color', '#ff0000');
                markerEntity.setAttribute('depth', '0.5');
                markerEntity.setAttribute('height', '0.5');
                markerEntity.setAttribute('width', '0.5');
                
                // Add text label if name is provided
                if (metadata.name) {
                    const text = document.createElement('a-text');
                    text.setAttribute('value', metadata.name);
                    text.setAttribute('look-at', '[gps-new-camera]');
                    text.setAttribute('scale', '2 2 2');
                    text.setAttribute('align', 'center');
                    text.setAttribute('color', '#ffffff');
                    text.setAttribute('position', '0 1 0');
                    markerEntity.appendChild(text);
                }
                break;
                
            case 'info':
                markerEntity = document.createElement('a-cylinder');
                markerEntity.setAttribute('color', '#00cc00');
                markerEntity.setAttribute('radius', '0.4');
                markerEntity.setAttribute('height', '0.8');
                
                // Add info icon
                const info = document.createElement('a-text');
                info.setAttribute('value', 'i');
                info.setAttribute('look-at', '[gps-new-camera]');
                info.setAttribute('scale', '3 3 3');
                info.setAttribute('align', 'center');
                info.setAttribute('color', '#ffffff');
                info.setAttribute('position', '0 0 0.3');
                markerEntity.appendChild(info);
                
                // Make clickable
                if (metadata.content) {
                    markerEntity.setAttribute('class', 'clickable');
                    markerEntity.addEventListener('click', () => {
                        alert(metadata.content); // Replace with better UI
                    });
                }
                break;
        }
        
        // Set common attributes
        markerEntity.setAttribute('id', `marker-${id}`);
        markerEntity.setAttribute('position', position);
        
        // Add GPS data if available
        if (metadata.lat && metadata.lng) {
            markerEntity.setAttribute('gps-new-entity-place', `latitude: ${metadata.lat}; longitude: ${metadata.lng}`);
        }
        
        // Store reference to marker
        this.markers[id] = {
            entity: markerEntity,
            type,
            metadata
        };
        
        // Add to scene
        this.scene.appendChild(markerEntity);
        
        return markerEntity;
    }
    
    // Remove a marker
    removeMarker(id) {
        if (this.markers[id]) {
            const markerEntity = this.markers[id].entity;
            markerEntity.parentNode.removeChild(markerEntity);
            delete this.markers[id];
        }
    }
    
    // Clear all markers
    clearMarkers() {
        Object.keys(this.markers).forEach(id => {
            this.removeMarker(id);
        });
    }
    
    // Update marker position
    updateMarkerPosition(id, position) {
        if (this.markers[id]) {
            this.markers[id].entity.setAttribute('position', position);
        }
    }
    
    // Create markers for a path
    createPathMarkers(path, options = {}) {
        path.forEach((point, index) => {
            const id = `path-${index}`;
            this.createMarker(id, `${point.x} ${point.y} ${point.z || 0}`, 'waypoint', {
                lat: point.lat,
                lng: point.lng
            });
        });
    }
} 