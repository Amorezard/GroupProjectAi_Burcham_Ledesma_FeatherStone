/**
 * Campus Path Processor - Frontend functionality for path analysis and processing
 */

class CampusPathProcessor {
    constructor(mapInstance) {
        this.map = mapInstance;
        this.currentPaths = [];
        this.suggestedPaths = [];
        this.prioritizedPaths = [];
        this.pathColors = {
            'main_road': '#0066cc',
            'sidewalk': '#3388ff',
            'shortcut': '#ff9900',
            'stairs': '#cc3300',
            'accessible': '#00cc66',
            'suggested_connection': '#cc00cc',
            'suggested_shortcut': '#ff66ff',
            'detected': '#666666',
            'default': '#3388ff'
        };
        this.pathWidths = {
            'main_road': 6,
            'sidewalk': 4,
            'shortcut': 3,
            'stairs': 3,
            'accessible': 5,
            'suggested_connection': 3,
            'suggested_shortcut': 3,
            'detected': 2,
            'default': 4
        };
        this.pathLayers = {};
    }

    /**
     * Initialize the path processor
     */
    init() {
        // Setup UI elements and event handlers
        this.setupUI();
        
        // Load initial paths
        this.loadPaths();
    }

    /**
     * Setup UI elements and event listeners
     */
    setupUI() {
        // Add UI elements for path controls if not already present
        const controlsDiv = document.getElementById('path-controls');
        if (!controlsDiv) {
            const container = document.createElement('div');
            container.id = 'path-controls';
            container.className = 'path-controls';
            container.innerHTML = `
                <h3>Path Controls</h3>
                <div class="control-buttons">
                    <button id="analyze-paths-btn" class="action-btn">Analyze Paths</button>
                    <button id="prioritize-paths-btn" class="action-btn">Prioritize Paths</button>
                    <button id="suggest-paths-btn" class="action-btn">Suggest Connections</button>
                    <button id="smart-paths-btn" class="action-btn">Generate Smart Paths</button>
                    <button id="integrate-osm-btn" class="action-btn">Import OpenStreetMap</button>
                </div>
                <div class="path-options">
                    <div class="option">
                        <label for="follow-terrain">
                            <input type="checkbox" id="follow-terrain" checked> 
                            Follow Terrain
                        </label>
                    </div>
                    <div class="option">
                        <label for="use-osm">
                            <input type="checkbox" id="use-osm" checked> 
                            Use OpenStreetMap
                        </label>
                    </div>
                    <div class="option">
                        <label for="show-suggested">
                            <input type="checkbox" id="show-suggested" checked> 
                            Show Suggested Paths
                        </label>
                    </div>
                    <div class="option">
                        <label for="path-visibility">Path Visibility</label>
                        <select id="path-visibility">
                            <option value="all">All Paths</option>
                            <option value="priority">By Priority</option>
                            <option value="type">By Type</option>
                        </select>
                    </div>
                </div>
                <div class="path-legend">
                    <h4>Path Types</h4>
                    <div class="legend-item">
                        <span class="color-box" style="background-color: #0066cc;"></span>
                        <span>Main Road</span>
                    </div>
                    <div class="legend-item">
                        <span class="color-box" style="background-color: #3388ff;"></span>
                        <span>Sidewalk</span>
                    </div>
                    <div class="legend-item">
                        <span class="color-box" style="background-color: #ff9900;"></span>
                        <span>Shortcut</span>
                    </div>
                    <div class="legend-item">
                        <span class="color-box" style="background-color: #cc3300;"></span>
                        <span>Stairs</span>
                    </div>
                    <div class="legend-item">
                        <span class="color-box" style="background-color: #00cc66;"></span>
                        <span>Accessible</span>
                    </div>
                    <div class="legend-item">
                        <span class="color-box" style="background-color: #cc00cc;"></span>
                        <span>Suggested Connection</span>
                    </div>
                    <div class="legend-item">
                        <span class="color-box" style="background-color: #444444;"></span>
                        <span>OpenStreetMap Road</span>
                    </div>
                </div>
                <div id="path-stats" class="path-stats"></div>
            `;
            
            document.body.appendChild(container);
            
            // Add event listeners
            document.getElementById('analyze-paths-btn').addEventListener('click', () => this.analyzePaths());
            document.getElementById('prioritize-paths-btn').addEventListener('click', () => this.prioritizePaths());
            document.getElementById('suggest-paths-btn').addEventListener('click', () => this.suggestConnections());
            document.getElementById('smart-paths-btn').addEventListener('click', () => this.generateSmartPaths());
            document.getElementById('integrate-osm-btn').addEventListener('click', () => this.integrateOSM());
            
            document.getElementById('follow-terrain').addEventListener('change', (e) => {
                this.followTerrain = e.target.checked;
            });
            
            document.getElementById('use-osm').addEventListener('change', (e) => {
                this.useOSM = e.target.checked;
            });
            
            document.getElementById('show-suggested').addEventListener('change', (e) => {
                this.toggleSuggestedPaths(e.target.checked);
            });
            
            document.getElementById('path-visibility').addEventListener('change', (e) => {
                this.updatePathVisibility(e.target.value);
            });
        }
        
        // Add CSS for the controls
        if (!document.getElementById('path-controls-css')) {
            const style = document.createElement('style');
            style.id = 'path-controls-css';
            style.textContent = `
                .path-controls {
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background-color: white;
                    border-radius: 5px;
                    padding: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.2);
                    z-index: 1000;
                    max-width: 300px;
                }
                
                .path-controls h3 {
                    margin-top: 0;
                    margin-bottom: 10px;
                }
                
                .control-buttons {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 5px;
                    margin-bottom: 10px;
                }
                
                .action-btn {
                    padding: 5px 10px;
                    border: none;
                    background-color: #0066cc;
                    color: white;
                    border-radius: 3px;
                    cursor: pointer;
                }
                
                .action-btn:hover {
                    background-color: #0055aa;
                }
                
                .path-options {
                    margin-bottom: 10px;
                }
                
                .option {
                    margin-bottom: 5px;
                }
                
                .path-legend {
                    margin-top: 10px;
                    border-top: 1px solid #eee;
                    padding-top: 10px;
                }
                
                .path-legend h4 {
                    margin-top: 0;
                    margin-bottom: 5px;
                }
                
                .legend-item {
                    display: flex;
                    align-items: center;
                    margin-bottom: 3px;
                }
                
                .color-box {
                    width: 15px;
                    height: 15px;
                    display: inline-block;
                    margin-right: 5px;
                }
                
                .path-stats {
                    margin-top: 10px;
                    font-size: 12px;
                }
            `;
            document.head.appendChild(style);
        }
    }

    /**
     * Load existing paths from the server
     */
    loadPaths() {
        fetch('/api/paths')
            .then(response => response.json())
            .then(paths => {
                this.currentPaths = paths;
                this.displayPaths(paths);
            })
            .catch(error => {
                console.error('Error loading paths:', error);
            });
    }

    /**
     * Display paths on the map
     * @param {Array} paths Array of path objects
     */
    displayPaths(paths) {
        // Clear existing path layers
        this.clearPathLayers();
        
        // Create a layer group for each path type
        const layerGroups = {};
        
        paths.forEach(path => {
            const pathType = path.type || 'default';
            
            if (!layerGroups[pathType]) {
                layerGroups[pathType] = [];
            }
            
            const points = path.nodes.map(node => [node.lat, node.lng]);
            
            if (points.length >= 2) {
                const color = this.pathColors[pathType] || this.pathColors['default'];
                const width = this.pathWidths[pathType] || this.pathWidths['default'];
                
                const polyline = L.polyline(points, {
                    color: color,
                    weight: width,
                    opacity: 0.8,
                    lineJoin: 'round'
                });
                
                // Add popup with path information
                polyline.bindPopup(`
                    <h3>Path Information</h3>
                    <p><strong>Type:</strong> ${pathType}</p>
                    <p><strong>From:</strong> ${path.from || 'N/A'}</p>
                    <p><strong>To:</strong> ${path.to || 'N/A'}</p>
                    ${path.priority ? `<p><strong>Priority:</strong> ${path.priority}</p>` : ''}
                `);
                
                layerGroups[pathType].push(polyline);
            }
        });
        
        // Add layer groups to the map
        for (const type in layerGroups) {
            this.pathLayers[type] = L.layerGroup(layerGroups[type]).addTo(this.map);
        }
        
        // Update stats
        this.updateStats(paths);
    }

    /**
     * Clear all path layers from the map
     */
    clearPathLayers() {
        for (const type in this.pathLayers) {
            this.map.removeLayer(this.pathLayers[type]);
        }
        this.pathLayers = {};
    }

    /**
     * Toggle the visibility of suggested paths
     * @param {boolean} visible Whether to show suggested paths
     */
    toggleSuggestedPaths(visible) {
        const suggestedTypes = ['suggested_connection', 'suggested_shortcut'];
        
        suggestedTypes.forEach(type => {
            if (this.pathLayers[type]) {
                if (visible) {
                    this.map.addLayer(this.pathLayers[type]);
                } else {
                    this.map.removeLayer(this.pathLayers[type]);
                }
            }
        });
    }

    /**
     * Update path visibility based on selected filter
     * @param {string} filterType Type of filter to apply ('all', 'priority', 'type')
     */
    updatePathVisibility(filterType) {
        switch (filterType) {
            case 'all':
                // Show all paths
                for (const type in this.pathLayers) {
                    this.map.addLayer(this.pathLayers[type]);
                }
                break;
                
            case 'priority':
                // Only show high priority paths (priority <= 2)
                this.clearPathLayers();
                
                const priorityPaths = this.prioritizedPaths.filter(path => 
                    path.priority && path.priority <= 2);
                
                this.displayPaths(priorityPaths);
                break;
                
            case 'type':
                // Show a dialog to select which types to display
                const types = Object.keys(this.pathLayers);
                
                // Simple implementation - toggle visibility based on checkboxes
                const dialog = document.createElement('div');
                dialog.className = 'path-type-dialog';
                dialog.innerHTML = `
                    <div class="dialog-content">
                        <h3>Select Path Types</h3>
                        <div class="path-type-options">
                            ${types.map(type => `
                                <div class="type-option">
                                    <label>
                                        <input type="checkbox" data-type="${type}" checked> 
                                        ${type.replace('_', ' ')}
                                    </label>
                                </div>
                            `).join('')}
                        </div>
                        <div class="dialog-buttons">
                            <button id="apply-types-btn">Apply</button>
                            <button id="cancel-types-btn">Cancel</button>
                        </div>
                    </div>
                `;
                
                document.body.appendChild(dialog);
                
                // Add CSS for dialog
                const dialogStyle = document.createElement('style');
                dialogStyle.textContent = `
                    .path-type-dialog {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background-color: rgba(0,0,0,0.5);
                        z-index: 2000;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    
                    .dialog-content {
                        background-color: white;
                        padding: 20px;
                        border-radius: 5px;
                        min-width: 300px;
                    }
                    
                    .path-type-options {
                        margin: 10px 0;
                        max-height: 300px;
                        overflow-y: auto;
                    }
                    
                    .dialog-buttons {
                        display: flex;
                        justify-content: flex-end;
                        gap: 10px;
                        margin-top: 10px;
                    }
                    
                    .dialog-buttons button {
                        padding: 5px 10px;
                        border: none;
                        background-color: #0066cc;
                        color: white;
                        border-radius: 3px;
                        cursor: pointer;
                    }
                    
                    .dialog-buttons button:hover {
                        background-color: #0055aa;
                    }
                    
                    #cancel-types-btn {
                        background-color: #666;
                    }
                `;
                document.head.appendChild(dialogStyle);
                
                // Add event listeners
                document.getElementById('apply-types-btn').addEventListener('click', () => {
                    const selectedTypes = [];
                    document.querySelectorAll('.type-option input:checked').forEach(checkbox => {
                        selectedTypes.push(checkbox.dataset.type);
                    });
                    
                    // Hide all types first
                    for (const type in this.pathLayers) {
                        this.map.removeLayer(this.pathLayers[type]);
                    }
                    
                    // Show selected types
                    selectedTypes.forEach(type => {
                        if (this.pathLayers[type]) {
                            this.map.addLayer(this.pathLayers[type]);
                        }
                    });
                    
                    // Remove dialog
                    document.body.removeChild(dialog);
                });
                
                document.getElementById('cancel-types-btn').addEventListener('click', () => {
                    document.body.removeChild(dialog);
                });
                
                break;
        }
    }

    /**
     * Update statistics display
     * @param {Array} paths Array of path objects
     */
    updateStats(paths) {
        const statsDiv = document.getElementById('path-stats');
        if (!statsDiv) return;
        
        // Count paths by type
        const typeCounts = {};
        paths.forEach(path => {
            const type = path.type || 'unknown';
            typeCounts[type] = (typeCounts[type] || 0) + 1;
        });
        
        // Calculate total path length
        let totalLength = 0;
        paths.forEach(path => {
            const nodes = path.nodes;
            for (let i = 1; i < nodes.length; i++) {
                const prev = nodes[i-1];
                const curr = nodes[i];
                totalLength += this.calculateDistance(prev.lat, prev.lng, curr.lat, curr.lng);
            }
        });
        
        // Build stats HTML
        let statsHtml = `
            <h4>Path Statistics</h4>
            <p><strong>Total Paths:</strong> ${paths.length}</p>
            <p><strong>Total Length:</strong> ${Math.round(totalLength)} meters</p>
            <p><strong>Path Types:</strong></p>
            <ul>
                ${Object.entries(typeCounts).map(([type, count]) => 
                    `<li>${type}: ${count}</li>`).join('')}
            </ul>
        `;
        
        statsDiv.innerHTML = statsHtml;
    }

    /**
     * Calculate distance between two points in meters
     * @param {number} lat1 Latitude of first point
     * @param {number} lng1 Longitude of first point
     * @param {number} lat2 Latitude of second point
     * @param {number} lng2 Longitude of second point
     * @returns {number} Distance in meters
     */
    calculateDistance(lat1, lng1, lat2, lng2) {
        // Earth's radius in meters
        const R = 6371000;
        
        const dLat = this.toRadians(lat2 - lat1);
        const dLng = this.toRadians(lng2 - lng1);
        
        const a = 
            Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(this.toRadians(lat1)) * Math.cos(this.toRadians(lat2)) * 
            Math.sin(dLng/2) * Math.sin(dLng/2);
            
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        const distance = R * c;
        
        return distance;
    }

    /**
     * Convert degrees to radians
     * @param {number} degrees Angle in degrees
     * @returns {number} Angle in radians
     */
    toRadians(degrees) {
        return degrees * Math.PI / 180;
    }

    /**
     * Analyze paths to detect their types
     */
    analyzePaths() {
        this.showLoader('Analyzing paths...');
        
        fetch('/api/analyze-paths', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(response => response.json())
            .then(data => {
                this.hideLoader();
                
                if (data.success) {
                    // Update and display paths
                    this.currentPaths = data.paths;
                    this.displayPaths(data.paths);
                    
                    // Show success message
                    this.showMessage('Paths analyzed successfully!', 'success');
                } else {
                    this.showMessage('Error analyzing paths', 'error');
                }
            })
            .catch(error => {
                this.hideLoader();
                console.error('Error analyzing paths:', error);
                this.showMessage('Error analyzing paths', 'error');
            });
    }

    /**
     * Prioritize paths for wayfinding
     */
    prioritizePaths() {
        this.showLoader('Prioritizing paths...');
        
        fetch('/api/prioritize-paths', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(response => response.json())
            .then(data => {
                this.hideLoader();
                
                if (data.success) {
                    // Update and display paths
                    this.currentPaths = data.paths;
                    this.prioritizedPaths = data.paths;
                    this.displayPaths(data.paths);
                    
                    // Show success message
                    this.showMessage('Paths prioritized successfully!', 'success');
                } else {
                    this.showMessage('Error prioritizing paths', 'error');
                }
            })
            .catch(error => {
                this.hideLoader();
                console.error('Error prioritizing paths:', error);
                this.showMessage('Error prioritizing paths', 'error');
            });
    }

    /**
     * Suggest connections between buildings
     */
    suggestConnections() {
        this.showLoader('Finding missing connections...');
        
        fetch('/api/suggest-connections')
            .then(response => response.json())
            .then(data => {
                this.hideLoader();
                
                if (data.success) {
                    this.suggestedPaths = data.suggested_paths;
                    
                    // Display all paths including suggested ones
                    const allPaths = [...this.currentPaths, ...this.suggestedPaths];
                    this.displayPaths(allPaths);
                    
                    // Show success message
                    this.showMessage(`Found ${data.count} suggested connections!`, 'success');
                } else {
                    this.showMessage('Error finding connections', 'error');
                }
            })
            .catch(error => {
                this.hideLoader();
                console.error('Error suggesting connections:', error);
                this.showMessage('Error suggesting connections', 'error');
            });
    }

    /**
     * Generate smart paths with terrain following
     */
    generateSmartPaths() {
        this.showLoader('Generating smart paths...');
        
        const includeTerrainCheckbox = document.getElementById('follow-terrain');
        const includeTerrain = includeTerrainCheckbox ? includeTerrainCheckbox.checked : true;
        
        fetch('/api/smart-paths', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                include_terrain: includeTerrain
            })
        })
            .then(response => response.json())
            .then(data => {
                this.hideLoader();
                
                if (data.success) {
                    // Update paths
                    this.prioritizedPaths = data.prioritized_paths;
                    this.suggestedPaths = data.suggested_paths;
                    
                    // Display all paths
                    const allPaths = [...this.prioritizedPaths, ...this.suggestedPaths];
                    this.displayPaths(allPaths);
                    
                    // Show success message
                    this.showMessage(`Generated smart paths with ${data.stats.suggested_count} suggestions!`, 'success');
                } else {
                    this.showMessage('Error generating smart paths', 'error');
                }
            })
            .catch(error => {
                this.hideLoader();
                console.error('Error generating smart paths:', error);
                this.showMessage('Error generating smart paths', 'error');
            });
    }

    /**
     * Integrate OpenStreetMap data with existing paths
     */
    integrateOSM() {
        this.showLoader('Importing OpenStreetMap data...');
        
        fetch('/api/integrate-osm', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        })
            .then(response => response.json())
            .then(data => {
                this.hideLoader();
                
                if (data.success) {
                    // Reload paths
                    this.loadPaths();
                    
                    // Show success message with statistics
                    const statsMessage = `
                        Imported ${data.stats.osm_paths} paths from OpenStreetMap.
                        Total paths: ${data.stats.total_paths}
                    `;
                    this.showMessage(statsMessage, 'success', 5000);
                } else {
                    this.showMessage('Error importing OpenStreetMap data', 'error');
                }
            })
            .catch(error => {
                this.hideLoader();
                console.error('Error integrating OpenStreetMap:', error);
                this.showMessage('Error importing OpenStreetMap data', 'error');
            });
    }

    /**
     * Find a natural path between two points
     * @param {Object} startLocation Start location with lat/lng
     * @param {Object} endLocation End location with lat/lng
     * @param {Object} preferences Path preferences
     */
    findNaturalPath(startLocation, endLocation, preferences = {}) {
        this.showLoader('Finding natural path...');
        
        // Get terrain and OSM preferences
        const includeTerrainCheckbox = document.getElementById('follow-terrain');
        const useOSMCheckbox = document.getElementById('use-osm');
        
        preferences.followTerrain = includeTerrainCheckbox ? includeTerrainCheckbox.checked : true;
        preferences.useOSM = useOSMCheckbox ? useOSMCheckbox.checked : true;
        
        fetch('/api/find-natural-path', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                start: startLocation,
                end: endLocation,
                preferences: preferences
            })
        })
            .then(response => response.json())
            .then(data => {
                this.hideLoader();
                
                // Display the path on the map
                if (data.path && data.path.length > 0) {
                    // Clear any existing path
                    if (this.currentPathLayer) {
                        this.map.removeLayer(this.currentPathLayer);
                    }
                    
                    // Create points array for the polyline
                    const points = data.path.map(point => [point.lat, point.lng]);
                    
                    // Determine path color based on source
                    let pathColor = '#ff0000';
                    if (data.stats && data.stats.path_source === 'openstreetmap') {
                        pathColor = '#444444';
                    }
                    
                    // Create and display the polyline
                    this.currentPathLayer = L.polyline(points, {
                        color: pathColor,
                        weight: 5,
                        opacity: 0.8,
                        lineJoin: 'round'
                    }).addTo(this.map);
                    
                    // Add start and end markers
                    this.startMarker = L.marker([startLocation.lat, startLocation.lng], {
                        icon: L.icon({
                            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
                            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
                            iconSize: [25, 41],
                            iconAnchor: [12, 41],
                            popupAnchor: [1, -34],
                            shadowSize: [41, 41]
                        })
                    }).addTo(this.map);
                    
                    this.endMarker = L.marker([endLocation.lat, endLocation.lng], {
                        icon: L.icon({
                            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
                            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
                            iconSize: [25, 41],
                            iconAnchor: [12, 41],
                            popupAnchor: [1, -34],
                            shadowSize: [41, 41]
                        })
                    }).addTo(this.map);
                    
                    // Zoom to fit the path
                    this.map.fitBounds(this.currentPathLayer.getBounds());
                    
                    // Display instructions
                    if (data.instructions && data.instructions.length > 0) {
                        this.displayInstructions(data.instructions);
                    }
                    
                    // Show success message with path source info
                    let sourceMessage = 'Path found!';
                    if (data.stats && data.stats.path_source) {
                        sourceMessage = `Path found using ${data.stats.path_source}!`;
                    }
                    this.showMessage(sourceMessage, 'success');
                    
                    // Display path statistics in the main UI
                    this.displayPathStats(data.stats);
                } else {
                    this.showMessage('No path found', 'error');
                }
            })
            .catch(error => {
                this.hideLoader();
                console.error('Error finding natural path:', error);
                this.showMessage('Error finding path', 'error');
            });
    }

    /**
     * Display navigation instructions
     * @param {Array} instructions Array of instruction strings
     */
    displayInstructions(instructions) {
        // Create or get instructions container
        let instructionsDiv = document.getElementById('path-instructions');
        
        if (!instructionsDiv) {
            instructionsDiv = document.createElement('div');
            instructionsDiv.id = 'path-instructions';
            instructionsDiv.className = 'path-instructions';
            document.body.appendChild(instructionsDiv);
            
            // Add CSS
            const style = document.createElement('style');
            style.textContent = `
                .path-instructions {
                    position: absolute;
                    bottom: 20px;
                    left: 20px;
                    background-color: white;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.2);
                    z-index: 1000;
                    max-width: 300px;
                    max-height: 300px;
                    overflow-y: auto;
                }
                
                .path-instructions h3 {
                    margin-top: 0;
                    margin-bottom: 10px;
                }
                
                .instructions-list {
                    padding-left: 20px;
                }
                
                .instructions-list li {
                    margin-bottom: 5px;
                }
                
                .close-instructions {
                    position: absolute;
                    top: 5px;
                    right: 5px;
                    cursor: pointer;
                    font-size: 16px;
                    color: #666;
                }
                
                .close-instructions:hover {
                    color: #000;
                }
            `;
            document.head.appendChild(style);
        }
        
        // Generate instructions HTML
        instructionsDiv.innerHTML = `
            <div class="close-instructions" onclick="document.getElementById('path-instructions').remove()">×</div>
            <h3>Navigation Instructions</h3>
            <ol class="instructions-list">
                ${instructions.map(instruction => `<li>${instruction}</li>`).join('')}
            </ol>
        `;
    }

    /**
     * Show a loading indicator
     * @param {string} message Loading message to display
     */
    showLoader(message = 'Loading...') {
        // Create loader if it doesn't exist
        let loader = document.getElementById('path-loader');
        
        if (!loader) {
            loader = document.createElement('div');
            loader.id = 'path-loader';
            loader.className = 'path-loader';
            document.body.appendChild(loader);
            
            // Add CSS
            const style = document.createElement('style');
            style.textContent = `
                .path-loader {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(0,0,0,0.5);
                    z-index: 3000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                    color: white;
                }
                
                .loader-spinner {
                    border: 5px solid #f3f3f3;
                    border-top: 5px solid #0066cc;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                    margin-bottom: 10px;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `;
            document.head.appendChild(style);
        }
        
        // Set loader content
        loader.innerHTML = `
            <div class="loader-spinner"></div>
            <div class="loader-message">${message}</div>
        `;
        
        // Show loader
        loader.style.display = 'flex';
    }

    /**
     * Hide the loading indicator
     */
    hideLoader() {
        const loader = document.getElementById('path-loader');
        if (loader) {
            loader.style.display = 'none';
        }
    }

    /**
     * Show a message toast
     * @param {string} message Message to display
     * @param {string} type Message type ('success', 'error', 'info')
     * @param {number} duration Duration in milliseconds
     */
    showMessage(message, type = 'info', duration = 3000) {
        // Create toast container if it doesn't exist
        let toastContainer = document.getElementById('path-toasts');
        
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'path-toasts';
            toastContainer.className = 'path-toasts';
            document.body.appendChild(toastContainer);
            
            // Add CSS
            const style = document.createElement('style');
            style.textContent = `
                .path-toasts {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 2000;
                    display: flex;
                    flex-direction: column;
                    align-items: flex-end;
                }
                
                .toast-message {
                    background-color: white;
                    border-radius: 5px;
                    padding: 10px 15px;
                    margin-bottom: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.2);
                    animation: fadeIn 0.3s ease;
                    max-width: 300px;
                }
                
                .toast-message.success {
                    border-left: 4px solid #00cc66;
                }
                
                .toast-message.error {
                    border-left: 4px solid #cc3300;
                }
                
                .toast-message.info {
                    border-left: 4px solid #0066cc;
                }
                
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(-10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                
                @keyframes fadeOut {
                    from { opacity: 1; transform: translateY(0); }
                    to { opacity: 0; transform: translateY(-10px); }
                }
            `;
            document.head.appendChild(style);
        }
        
        // Create toast
        const toast = document.createElement('div');
        toast.className = `toast-message ${type}`;
        toast.textContent = message;
        
        // Add to container
        toastContainer.appendChild(toast);
        
        // Remove after duration
        setTimeout(() => {
            toast.style.animation = 'fadeOut 0.3s ease';
            
            // Remove after animation
            setTimeout(() => {
                toastContainer.removeChild(toast);
            }, 300);
        }, duration);
    }

    /**
     * Display path statistics in the UI
     * @param {Object} stats Path statistics
     */
    displayPathStats(stats) {
        const statsDiv = document.getElementById('path-stats');
        if (!statsDiv) return;
        
        let statsHtml = `<h4>Path Statistics</h4>`;
        
        if (stats) {
            statsHtml += `
                <p><strong>Distance:</strong> ${Math.round(stats.total_distance)} meters</p>
                <p><strong>Est. Time:</strong> ${stats.estimated_time} min</p>
                <p><strong>Path Source:</strong> ${stats.path_source || 'Campus data'}</p>
                <p><strong>Segments:</strong> ${stats.num_segments || 0}</p>
            `;
            
            if (stats.path_types) {
                statsHtml += `<p><strong>Path Types:</strong></p><ul>`;
                for (const [type, count] of Object.entries(stats.path_types)) {
                    statsHtml += `<li>${type}: ${count}</li>`;
                }
                statsHtml += `</ul>`;
            }
        }
        
        statsDiv.innerHTML = statsHtml;
    }
} 