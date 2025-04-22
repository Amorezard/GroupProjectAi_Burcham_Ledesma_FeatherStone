document.addEventListener('DOMContentLoaded', () => {
    // Debug logs
    console.log('AR.js script loaded');
    
    // Hide loader when AR is ready
    const loader = document.querySelector('.arjs-loader');
    const scene = document.querySelector('a-scene');
    
    // Add error handling for AR scene
    scene.addEventListener('loaded', () => {
        console.log('AR scene loaded successfully');
        loader.style.display = 'none';
        
        // Show mode switch button
        document.getElementById('mode-switch').style.display = 'block';
        
        // Show initial information
        showInfo('AR Navigation', 'Use a Hiro marker for testing. Find and print a Hiro marker from the web.');
    });
    
    scene.addEventListener('arjs-nft-loaded', () => {
        console.log('AR NFT content loaded');
    });
    
    // Error handling
    scene.addEventListener('renderstart', () => {
        console.log('AR rendering started');
    });
    
    // Catch any errors in the AR initialization
    window.addEventListener('camera-error', (error) => {
        console.error('Camera error:', error);
        showErrorMessage('Camera access error. Please check permissions.');
    });
    
    // Set up marker-based AR by default since it's more reliable for testing
    setupMarkerBasedAR();
});

function updateNearestWaypoint(userPosition) {
    // Calculate distance to each waypoint and update UI
    const ui = document.querySelector('#ui');
    ui.innerHTML = `
        <a-text value="You are here" 
                position="0 0 -1" 
                color="white" 
                scale="2 2 2">
        </a-text>
    `;
}

function setupMarkerBasedAR() {
    console.log('Setting up marker-based AR');
    
    // Get scene reference
    const scene = document.querySelector('a-scene');
    
    // Get loader reference
    const loader = document.querySelector('.arjs-loader');
    
    // Remove GPS-based entities
    const navigationPoints = document.querySelector('#navigation-points');
    navigationPoints.innerHTML = '';
    
    // Create a hiro marker for testing
    const marker = document.createElement('a-marker');
    marker.setAttribute('preset', 'hiro');
    marker.setAttribute('id', 'hiro-marker');
    
    // Add content to the marker
    const box = document.createElement('a-box');
    box.setAttribute('position', '0 0.5 0');
    box.setAttribute('material', 'color: red;');
    box.setAttribute('animation', 'property: rotation; to: 0 360 0; loop: true; dur: 5000; easing: linear');
    marker.appendChild(box);
    
    // Add campus locations as clickable objects
    const locations = [
        { name: 'Library', position: '-1 0.5 0', color: 'blue' },
        { name: 'Student Center', position: '1 0.5 0', color: 'green' },
        { name: 'Academic Building', position: '0 0.5 -1', color: 'yellow' }
    ];
    
    locations.forEach(location => {
        const sphere = document.createElement('a-sphere');
        sphere.setAttribute('position', location.position);
        sphere.setAttribute('radius', '0.3');
        sphere.setAttribute('material', `color: ${location.color}`);
        sphere.setAttribute('class', 'clickable');
        
        // Add event listener for click
        sphere.addEventListener('click', () => {
            console.log(`${location.name} clicked`);
            showInfo(location.name, `This is the ${location.name}. Additional information would be displayed here.`);
        });
        
        // Add label
        const text = document.createElement('a-text');
        text.setAttribute('value', location.name);
        text.setAttribute('position', location.position.replace(/^([-\d.]+) ([-\d.]+) ([-\d.]+)$/, '$1 $2 $3'));
        text.setAttribute('position', (pos => {
            const [x, y, z] = pos.split(' ');
            return `${x} ${parseFloat(y) + 0.5} ${z}`;
        })(location.position));
        text.setAttribute('align', 'center');
        text.setAttribute('color', 'white');
        text.setAttribute('scale', '0.5 0.5 0.5');
        
        marker.appendChild(sphere);
        marker.appendChild(text);
    });
    
    // Add a simple path between locations
    const path = document.createElement('a-entity');
    path.setAttribute('line', 'start: -1 0.5 0; end: 1 0.5 0; color: white');
    marker.appendChild(path);
    
    const path2 = document.createElement('a-entity');
    path2.setAttribute('line', 'start: 1 0.5 0; end: 0 0.5 -1; color: white');
    marker.appendChild(path2);
    
    const path3 = document.createElement('a-entity');
    path3.setAttribute('line', 'start: 0 0.5 -1; end: -1 0.5 0; color: white');
    marker.appendChild(path3);
    
    // Add title
    const title = document.createElement('a-text');
    title.setAttribute('value', 'Merrimack College AR Map');
    title.setAttribute('position', '0 1.5 0');
    title.setAttribute('align', 'center');
    title.setAttribute('color', 'white');
    title.setAttribute('scale', '1 1 1');
    marker.appendChild(title);
    
    // Add the marker to the scene
    scene.appendChild(marker);
    
    // Update user instructions
    loader.innerHTML = '<div>Point your camera at a Hiro marker to see the campus map. You can print a Hiro marker from the web or display it on another device.</div>';
    
    // Hide loader after a short delay
    setTimeout(() => {
        loader.style.display = 'none';
    }, 5000);
}

function setupGpsBasedAR() {
    console.log('Setting up GPS-based AR');
    
    // Get scene reference
    const scene = document.querySelector('a-scene');
    
    // Show loading while switching modes
    const loader = document.querySelector('.arjs-loader');
    loader.innerHTML = '<div>Switching to GPS mode. Please allow location access when prompted.</div>';
    loader.style.display = 'flex';
    
    // Remove existing markers
    const existingMarker = document.querySelector('a-marker');
    if (existingMarker) {
        existingMarker.parentNode.removeChild(existingMarker);
    }
    
    // Add GPS camera
    const camera = document.querySelector('a-entity[camera]');
    camera.setAttribute('gps-camera', '');
    camera.setAttribute('rotation-reader', '');
    
    // Define campus waypoints (example coordinates - would need to be replaced with actual coordinates)
    const waypoints = [
        { name: "Library", lat: 42.6700, lon: -71.1234, description: "Main Library" },
        { name: "Student Center", lat: 42.6701, lon: -71.1235, description: "Student Center" },
        { name: "Academic Building", lat: 42.6702, lon: -71.1236, description: "Academic Building" }
    ];
    
    // Create navigation points
    const navigationPoints = document.querySelector('#navigation-points');
    navigationPoints.innerHTML = '';
    
    waypoints.forEach(waypoint => {
        const entity = document.createElement('a-entity');
        entity.setAttribute('gps-entity-place', {
            latitude: waypoint.lat,
            longitude: waypoint.lon
        });
        
        // Create a 3D marker
        entity.setAttribute('geometry', {
            primitive: 'sphere',
            radius: 0.5
        });
        
        entity.setAttribute('material', {
            color: '#4CC3D9'
        });
        
        // Add text label
        const text = document.createElement('a-text');
        text.setAttribute('value', waypoint.name);
        text.setAttribute('align', 'center');
        text.setAttribute('position', '0 1.5 0');
        text.setAttribute('color', 'white');
        text.setAttribute('scale', '2 2 2');
        
        entity.appendChild(text);
        navigationPoints.appendChild(entity);
    });
    
    // Hide loader after GPS is initialized
    setTimeout(() => {
        loader.style.display = 'none';
    }, 3000);
}

function showErrorMessage(message) {
    const loader = document.querySelector('.arjs-loader');
    loader.innerHTML = `<div>${message}</div>`;
    loader.style.backgroundColor = 'rgba(255, 0, 0, 0.7)';
    loader.style.display = 'flex';
}

function showInfo(title, content) {
    const infoPanel = document.getElementById('info-panel');
    document.getElementById('info-title').textContent = title;
    document.getElementById('info-content').textContent = content;
    infoPanel.style.display = 'block';
    
    // Hide after 5 seconds
    setTimeout(() => {
        infoPanel.style.display = 'none';
    }, 5000);
} 