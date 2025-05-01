# Merrimack College AR Campus Navigation

A simple web-based AR application for navigating Merrimack College campus using marker-based or GPS-based AR.

# Merrimack College Wayfinding AI

This application provides AI-powered navigation for Merrimack College campus, helping students, faculty, and visitors find their way around efficiently.

## Features

- **Interactive Campus Map**: Explore the campus with an interactive map showing all buildings and points of interest
- **Intelligent Pathfinding**: Find the optimal path between any two locations on campus
- **Turn-by-Turn Directions**: Receive detailed directions for navigating around campus
- **AI-Enhanced Navigation**: Get intelligent insights about your route based on real-time conditions
- **OpenStreetMap Integration**: Uses real-world path data from OpenStreetMap for accurate navigation
- **Mobile-Friendly Design**: Access from any device with a responsive interface

## Setup and Installation

### Automatic Setup (Recommended)

1. For Windows users:
   ```
   setup.bat
   ```

2. For Linux/macOS users:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

### Manual Setup

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate.bat  # Windows
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Test the OSMnx installation:
   ```
   python test_osmnx.py
   ```

4. Run the Flask application:
   ```
   python app.py
   ```

5. Access the application through a web browser:
   ```
   http://localhost:5000/wayfinding
   ```

## System Requirements

- Python 3.8 or higher
- Modern web browser with JavaScript enabled
- Internet connection (for downloading OpenStreetMap data)
- At least 2GB of RAM (recommended 4GB+) for processing path data

## Using the Application

### Marker-Based AR Mode (Default)

1. Print a Hiro marker from here: https://raw.githubusercontent.com/AR-js-org/AR.js/master/data/images/HIRO.jpg
   - Alternatively, display the marker on another device.

2. Allow camera access when prompted.

3. Point your device's camera at the Hiro marker to see the campus map.

4. Interact with the 3D objects on the map to see building information.

### GPS-Based AR Mode

1. Click the "Switch AR Mode" button to switch to GPS-based navigation.

2. Allow location access when prompted.

3. The application will display campus buildings as 3D objects in their real-world locations.

### Using the Wayfinding Interface

1. Select your starting location from the dropdown or by clicking on the map
2. Select your destination
3. Configure route options if needed (accessibility, terrain following, etc.)
4. Click "Find Path" to generate the optimal route
5. Follow the turn-by-turn directions displayed in the sidebar

## AI Features

The wayfinding application leverages several AI technologies:

- **Pathfinding Algorithms**: Uses A* search algorithm to find optimal paths between locations
- **Spatial Intelligence**: Analyzes campus layout to provide the most efficient routes
- **Context-Aware Routing**: Considers factors like accessibility and building hours
- **Natural Language Directions**: Translates complex paths into easy-to-follow instructions

## Path Processing Feature

The wayfinding AI includes an intelligent path processing system that can automatically identify, analyze, and prioritize campus paths. This feature uses advanced pathfinding algorithms with terrain following capabilities to create natural-looking paths that follow existing campus walkways and roads.

### Key Features

- **Automatic Path Type Detection**: Analyzes path properties to determine if they are sidewalks, roads, shortcuts, stairs, or accessible routes
- **Path Prioritization**: Assigns priorities to paths based on their importance, connectivity, and properties
- **Terrain Following**: Adjusts paths to follow the terrain naturally, avoiding unnecessarily steep sections
- **Missing Connection Detection**: Identifies where new paths could be added to improve campus connectivity
- **Natural Path Generation**: Creates smooth, natural-looking paths between any two points
- **OpenStreetMap Integration**: Incorporates real-world path data from OpenStreetMap for accurate navigation

### How to Use

1. In the wayfinding interface, check "Enable AI Path Processing" to activate the advanced path processing features
2. Use the path control panel to:
   - Analyze existing paths to detect their types
   - Prioritize paths for wayfinding
   - Suggest new connections between buildings
   - Generate smart paths with terrain following
   - Import OpenStreetMap data for your campus area

3. When finding a route, enable "Follow Terrain" to create paths that naturally follow the campus topography

## OpenStreetMap Integration

This application uses OpenStreetMap data through the OSMnx library (version 2.0.2) to provide accurate path data for navigation. The integration allows:

- Downloading and processing real-world street network data
- Identifying sidewalks, roads, paths, and other transportation infrastructure
- Creating route visualizations based on actual geographic data
- Combining custom campus paths with public OSM data for complete coverage

## Technical Details

The application uses:
- Flask for the web server (backend)
- Leaflet.js for interactive maps (frontend)
- OSMnx 2.0.2 for OpenStreetMap data processing
- NetworkX for graph-based pathfinding
- Python 3.8+ with numpy, scikit-learn and other scientific libraries

## API Endpoints

- `/api/analyze-paths`: Detects path types based on their properties
- `/api/prioritize-paths`: Assigns priority values to paths
- `/api/suggest-connections`: Identifies missing connections between buildings
- `/api/smart-paths`: Generates comprehensive path data with terrain following
- `/api/find-natural-path`: Finds a natural path between two points
- `/api/integrate-osm`: Imports and integrates OpenStreetMap data

## Future Enhancements

- Indoor navigation with floor-to-floor directions
- Integration with class schedules for personalized navigation
- Real-time updates for construction zones and detours
- Voice-guided navigation for accessibility
- 3D terrain visualization for better path understanding

## Troubleshooting

- **Loading screen doesn't disappear**: Make sure your camera permissions are granted and that you're pointing at a Hiro marker in marker mode.
  
- **No AR content appears**: Check that your device supports AR (most modern smartphones do).

- **GPS mode doesn't work**: Ensure you have GPS enabled on your device and that you're outdoors for better GPS accuracy.

## Development Notes

- The current implementation uses sample coordinates. For a production version, you'll need to replace these with actual GPS coordinates of campus buildings.
- For better performance, consider optimizing 3D models and using compressed textures.

## How to Use

1. Select your starting location from the dropdown or by clicking on the map
2. Select your destination
3. Click "Find Path" to generate the optimal route
4. Follow the turn-by-turn directions displayed in the sidebar

## AI Features

The wayfinding application leverages several AI technologies:

- **Pathfinding Algorithms**: Uses A* search algorithm to find optimal paths between locations
- **Spatial Intelligence**: Analyzes campus layout to provide the most efficient routes
- **Context-Aware Routing**: Considers factors like accessibility and building hours
- **Natural Language Directions**: Translates complex paths into easy-to-follow instructions

## Future Enhancements

- Indoor navigation with floor-to-floor directions
- Integration with class schedules for personalized navigation
- Real-time updates for construction zones and detours
- Voice-guided navigation for accessibility

## Path Processing Feature

The wayfinding AI includes an intelligent path processing system that can automatically identify, analyze, and prioritize campus paths. This feature uses advanced pathfinding algorithms with terrain following capabilities to create natural-looking paths that follow existing campus walkways and roads.

### Key Features

- **Automatic Path Type Detection**: Analyzes path properties to determine if they are sidewalks, roads, shortcuts, stairs, or accessible routes
- **Path Prioritization**: Assigns priorities to paths based on their importance, connectivity, and properties
- **Terrain Following**: Adjusts paths to follow the terrain naturally, avoiding unnecessarily steep sections
- **Missing Connection Detection**: Identifies where new paths could be added to improve campus connectivity
- **Natural Path Generation**: Creates smooth, natural-looking paths between any two points

### How to Use

1. In the wayfinding interface, check "Enable AI Path Processing" to activate the advanced path processing features
2. Use the path control panel to:
   - Analyze existing paths to detect their types
   - Prioritize paths for wayfinding
   - Suggest new connections between buildings
   - Generate smart paths with terrain following

3. When finding a route, enable "Follow Terrain" to create paths that naturally follow the campus topography

### Technical Details

The path processing system uses:
- Non-linear cost functions to penalize steep slopes
- Graph-based pathfinding with weighted edges for different path types
- Computer vision for path detection from aerial imagery (when available)
- Spatial analysis to identify optimal connections

### API Endpoints

- `/api/analyze-paths`: Detects path types based on their properties
- `/api/prioritize-paths`: Assigns priority values to paths
- `/api/suggest-connections`: Identifies missing connections between buildings
- `/api/smart-paths`: Generates comprehensive path data with terrain following
- `/api/find-natural-path`: Finds a natural path between two points