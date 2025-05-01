# Merrimack College AR Campus Navigation

A simple web-based AR application for navigating Merrimack College campus using marker-based or GPS-based AR.

# Merrimack College Wayfinding AI

This application provides AI-powered navigation for Merrimack College campus, helping students, faculty, and visitors find their way around efficiently.

## Features

- **Interactive Campus Map**: Explore the campus with an interactive map showing all buildings and points of interest
- **Intelligent Pathfinding**: Find the optimal path between any two locations on campus
- **Turn-by-Turn Directions**: Receive detailed directions for navigating around campus
- **AI-Enhanced Navigation**: Get intelligent insights about your route based on real-time conditions
- **Mobile-Friendly Design**: Access from any device with a responsive interface

## Setup and Installation

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask application:
   ```
   python app.py
   ```

3. Access the application through a web browser on your mobile device by navigating to:
   ```
   http://[your-local-ip]:5000
   ```
   
   Note: Your device must be on the same network as the server.

3. Navigate to https://localhost:5000/wayfinding in your browser

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

## Troubleshooting

- **Loading screen doesn't disappear**: Make sure your camera permissions are granted and that you're pointing at a Hiro marker in marker mode.
  
- **No AR content appears**: Check that your device supports AR (most modern smartphones do).

- **GPS mode doesn't work**: Ensure you have GPS enabled on your device and that you're outdoors for better GPS accuracy.

## Technical Information

This application uses:
- AR.js for augmented reality functionality
- A-Frame for 3D content rendering
- Flask for the web server

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