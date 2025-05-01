/**
 * Custom UI Enhancement for Merrimack College Wayfinding Application
 */

class CampusUIEnhancer {
    constructor(map) {
        this.map = map;
        this.buildingsLayer = L.layerGroup().addTo(map);
        this.buildingsData = [];
        this.activeInfoWindow = null;
        this.categories = {
            'academic': { icon: 'graduation-cap', color: '#0066cc' },
            'residential': { icon: 'home', color: '#cc6600' },
            'dining': { icon: 'utensils', color: '#00a651' },
            'athletics': { icon: 'basketball-ball', color: '#cc0000' },
            'admin': { icon: 'university', color: '#9933cc' },
            'parking': { icon: 'parking', color: '#666666' },
            'other': { icon: 'building', color: '#003366' }
        };
        
        // Create legend
        this.createLegend();
    }

    /**
     * Initialize buildings with advanced markers and info panels
     */
    initializeBuildings(buildings) {
        this.buildingsLayer.clearLayers();
        this.buildingsData = buildings;
        
        buildings.forEach(building => {
            const marker = this.createBuildingMarker(building);
            this.buildingsLayer.addLayer(marker);
        });
        
        return this;
    }
    
    /**
     * Determine the category of a building based on its name or ID
     */
    getBuildingCategory(building) {
        const name = building.name.toLowerCase();
        const id = building.id.toLowerCase();
        
        if (id.includes('hall') || id.includes('center') || name.includes('school') || name.includes('college') || name.includes('academic')) {
            return 'academic';
        } else if (id.includes('lot') || id.includes('parking')) {
            return 'parking';
        } else if (id.includes('stadium') || id.includes('rink') || id.includes('field') || id.includes('gym') || id.includes('athletic')) {
            return 'athletics';
        } else if (id.includes('residence') || id.includes('dorm') || id.includes('apartment') || id.includes('house') || name.includes('residence')) {
            return 'residential';
        } else if (id.includes('dining') || id.includes('cafe') || id.includes('restaurant') || id.includes('food')) {
            return 'dining';
        } else if (id.includes('office') || id.includes('admin') || id.includes('department')) {
            return 'admin';
        }
        
        return 'other';
    }
    
    /**
     * Create a custom marker for building with icon
     */
    createBuildingMarker(building) {
        const category = this.getBuildingCategory(building);
        const { icon, color } = this.categories[category] || this.categories.other;
        
        // Create custom div icon with Font Awesome icon
        const customIcon = L.divIcon({
            className: `building-marker ${category}`,
            html: `<i class="fas fa-${icon}"></i>`,
            iconSize: [36, 36],
            iconAnchor: [18, 18]
        });
        
        // Create marker with custom icon
        const marker = L.marker([building.lat, building.lng], {
            icon: customIcon,
            title: building.name,
            riseOnHover: true
        });
        
        // Add popup on click
        marker.on('click', () => {
            this.openInfoPanel(building, marker);
        });
        
        // Add tooltip on hover
        marker.bindTooltip(building.name, {
            direction: 'top',
            offset: [0, -10],
            className: 'custom-tooltip'
        });
        
        return marker;
    }
    
    /**
     * Open an enhanced info panel for a building
     */
    openInfoPanel(building, marker) {
        // Close any existing info panel
        if (this.activeInfoWindow) {
            this.map.closePopup(this.activeInfoWindow);
        }
        
        const category = this.getBuildingCategory(building);
        const { icon, color } = this.categories[category] || this.categories.other;
        
        // Generate amenities based on building type
        const amenities = this.generateAmenities(building, category);
        
        // Create info panel content
        const content = `
            <div class="info-panel">
                <div class="info-panel-header" style="background-color: ${color}">
                    <h3>${building.name}</h3>
                    <div class="info-panel-header-icon">
                        <i class="fas fa-${icon}"></i>
                    </div>
                </div>
                
                <div class="info-panel-body">
                    ${building.image ? 
                        `<img src="/static/images/buildings/${building.image}" class="info-panel-image" alt="${building.name}">` : 
                        ''}
                    
                    <div class="info-panel-section">
                        <span class="info-panel-label">Building Type</span>
                        <span class="info-panel-value">${this.formatCategory(category)}</span>
                        
                        <span class="info-panel-label">Location</span>
                        <span class="info-panel-value">${building.lat.toFixed(6)}, ${building.lng.toFixed(6)}</span>
                        
                        ${building.floors ? 
                            `<span class="info-panel-label">Floors</span>
                             <span class="info-panel-value">${building.floors}</span>` : 
                            ''}
                    </div>
                    
                    ${building.description ? 
                        `<div class="info-panel-section">
                            <div class="info-panel-description">${building.description}</div>
                         </div>` : 
                        ''}
                    
                    ${amenities.length > 0 ? 
                        `<div class="info-panel-section">
                            <span class="info-panel-label">Amenities</span>
                            <div class="info-panel-amenities">
                                ${amenities.map(a => 
                                    `<div class="amenity-badge">
                                        <i class="fas fa-${a.icon}"></i> ${a.name}
                                     </div>`
                                ).join('')}
                            </div>
                         </div>` : 
                        ''}
                </div>
                
                <div class="info-panel-footer">
                    <button class="info-panel-btn" onclick="campusUI.startRoute('${building.id}')">
                        <i class="fas fa-route"></i> Navigate Here
                    </button>
                    
                    <button class="info-panel-btn" onclick="campusUI.routeFromHere('${building.id}')">
                        <i class="fas fa-walking"></i> Route From Here
                    </button>
                </div>
            </div>
        `;
        
        // Create and open popup
        this.activeInfoWindow = L.popup({
            closeButton: true,
            autoClose: false,
            closeOnEscapeKey: true,
            closeOnClick: false,
            className: 'custom-popup',
            maxWidth: 350
        })
            .setLatLng([building.lat, building.lng])
            .setContent(content)
            .openOn(this.map);
    }
    
    /**
     * Format category string for display
     */
    formatCategory(category) {
        return category.charAt(0).toUpperCase() + category.slice(1);
    }
    
    /**
     * Generate amenities based on building category and other properties
     */
    generateAmenities(building, category) {
        const amenities = [];
        
        // Add amenities based on building category
        if (category === 'academic') {
            amenities.push({ name: 'Wi-Fi', icon: 'wifi' });
            amenities.push({ name: 'Classrooms', icon: 'chalkboard-teacher' });
            
            if (Math.random() > 0.5) {
                amenities.push({ name: 'Study Spaces', icon: 'book' });
            }
            
            if (Math.random() > 0.7) {
                amenities.push({ name: 'Labs', icon: 'flask' });
            }
        } else if (category === 'residential') {
            amenities.push({ name: 'Wi-Fi', icon: 'wifi' });
            amenities.push({ name: 'Laundry', icon: 'tshirt' });
            
            if (Math.random() > 0.5) {
                amenities.push({ name: 'Common Room', icon: 'couch' });
            }
            
            if (Math.random() > 0.7) {
                amenities.push({ name: 'Kitchen', icon: 'utensils' });
            }
        } else if (category === 'athletics') {
            amenities.push({ name: 'Changing Rooms', icon: 'door-open' });
            
            if (Math.random() > 0.5) {
                amenities.push({ name: 'Spectator Area', icon: 'users' });
            }
            
            if (Math.random() > 0.7) {
                amenities.push({ name: 'Equipment', icon: 'dumbbell' });
            }
        } else if (category === 'dining') {
            amenities.push({ name: 'Food Service', icon: 'utensils' });
            amenities.push({ name: 'Seating', icon: 'chair' });
            
            if (Math.random() > 0.5) {
                amenities.push({ name: 'Coffee', icon: 'coffee' });
            }
        } else if (category === 'parking') {
            amenities.push({ name: 'Parking', icon: 'parking' });
            
            if (building.id.includes('lot_')) {
                const lotType = building.description?.includes('Resident') ? 'Resident' : 'Commuter';
                amenities.push({ name: `${lotType} Parking`, icon: 'car' });
            }
        } else if (category === 'admin') {
            amenities.push({ name: 'Reception', icon: 'concierge-bell' });
            amenities.push({ name: 'Offices', icon: 'briefcase' });
        }
        
        // Add accessibility amenity
        if (Math.random() > 0.7) {
            amenities.push({ name: 'Accessible', icon: 'wheelchair' });
        }
        
        return amenities;
    }
    
    /**
     * Create a legend showing building types
     */
    createLegend() {
        const legend = L.control({ position: 'bottomleft' });
        
        legend.onAdd = () => {
            const div = L.DomUtil.create('div', 'map-legend');
            div.innerHTML = `
                <div class="legend-title">Building Types</div>
                ${Object.entries(this.categories).map(([category, details]) => `
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: ${details.color}; height: 12px;"></div>
                        <div class="legend-text">${this.formatCategory(category)}</div>
                    </div>
                `).join('')}
                
                <div class="legend-title" style="margin-top: 10px;">Path Types</div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #3388ff;"></div>
                    <div class="legend-text">Sidewalk</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #0066cc;"></div>
                    <div class="legend-text">Main Road</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #ff9900;"></div>
                    <div class="legend-text">Shortcut</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #cc3300;"></div>
                    <div class="legend-text">Stairs</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #00cc66;"></div>
                    <div class="legend-text">Accessible</div>
                </div>
            `;
            return div;
        };
        
        legend.addTo(this.map);
    }
    
    /**
     * Start a route to this building
     */
    startRoute(buildingId) {
        // Find destination in dropdown
        const endSelect = document.getElementById('end-location');
        const endOptions = Array.from(endSelect.options);
        
        const targetOption = endOptions.find(option => {
            try {
                const data = JSON.parse(option.value);
                return data.id === buildingId;
            } catch (e) {
                return false;
            }
        });
        
        if (targetOption) {
            endSelect.value = targetOption.value;
            
            // If start is not selected, try to get user location or use a default
            const startSelect = document.getElementById('start-location');
            if (!startSelect.value) {
                startSelect.selectedIndex = 1; // Use first building as starting point
            }
            
            // Close popup
            this.map.closePopup(this.activeInfoWindow);
            
            // Click find path button
            document.getElementById('find-path-btn').click();
        }
    }
    
    /**
     * Set this building as the starting point for a route
     */
    routeFromHere(buildingId) {
        // Find start in dropdown
        const startSelect = document.getElementById('start-location');
        const startOptions = Array.from(startSelect.options);
        
        const targetOption = startOptions.find(option => {
            try {
                const data = JSON.parse(option.value);
                return data.id === buildingId;
            } catch (e) {
                return false;
            }
        });
        
        if (targetOption) {
            startSelect.value = targetOption.value;
            
            // Close popup
            this.map.closePopup(this.activeInfoWindow);
            
            // Focus on destination dropdown
            document.getElementById('end-location').focus();
        }
    }
} 