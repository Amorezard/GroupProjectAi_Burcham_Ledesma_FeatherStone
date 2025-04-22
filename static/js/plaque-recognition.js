/**
 * Room Plaque Recognition System
 * 
 * This module integrates computer vision-based room plaque detection
 * with AR pathfinding for enhanced indoor navigation. It uses the
 * device camera to recognize room plaques and overlay navigation
 * information accordingly.
 */

class PlaqueRecognitionSystem {
    /**
     * Initialize the plaque recognition system
     * 
     * @param {Object} options - Configuration options
     * @param {HTMLElement} options.videoElement - Video element for camera feed
     * @param {HTMLElement} options.canvasElement - Canvas for processing
     * @param {Function} options.onPlaqueRecognized - Callback when plaque is recognized
     * @param {String} options.modelPath - Path to the trained model (if using a pre-trained model)
     */
    constructor(options) {
        this.videoElement = options.videoElement;
        this.canvasElement = options.canvasElement;
        this.canvasContext = this.canvasElement.getContext('2d');
        this.onPlaqueRecognized = options.onPlaqueRecognized || (() => {});
        this.modelPath = options.modelPath;
        this.isProcessing = false;
        this.isActive = false;
        this.plaqueData = {};
        this.recognitionConfidence = 0.7; // Minimum confidence threshold
        this.scanInterval = null;
        this.lastRecognizedPlaque = null;
        this.cooldownPeriod = 3000; // Time in ms before the same plaque can trigger recognition again
        
        // Initialize the plaque database
        this.initializePlaqueDatabase();
    }
    
    /**
     * Initialize the database of known plaques
     * This would typically be populated from the server
     */
    async initializePlaqueDatabase() {
        try {
            // Fetch plaque data from the server
            const response = await fetch('/api/plaques');
            if (response.ok) {
                this.plaqueData = await response.json();
                console.log('Plaque database initialized with', Object.keys(this.plaqueData).length, 'entries');
            } else {
                console.error('Failed to fetch plaque data');
                // Use fallback data if server request fails
                this.plaqueData = this.getFallbackPlaqueData();
            }
        } catch (error) {
            console.error('Error initializing plaque database:', error);
            this.plaqueData = this.getFallbackPlaqueData();
        }
    }
    
    /**
     * Provides fallback plaque data if server request fails
     * In a production system, this would be replaced by server data
     */
    getFallbackPlaqueData() {
        return {
            "M101": {
                building: "mendel",
                roomId: "m101",
                description: "Lecture Hall"
            },
            "M102": {
                building: "mendel",
                roomId: "m102",
                description: "Chemistry Lab"
            },
            "MQ1": {
                building: "mcquade",
                roomId: "mq1",
                description: "Library Main Floor"
            }
        };
    }
    
    /**
     * Start the plaque recognition system
     */
    start() {
        if (this.isActive) return;
        
        this.isActive = true;
        
        // Set up canvas size to match video
        this.canvasElement.width = this.videoElement.videoWidth || 640;
        this.canvasElement.height = this.videoElement.videoHeight || 480;
        
        // Start periodic scanning of video frames
        this.scanInterval = setInterval(() => {
            if (!this.isProcessing && this.videoElement.readyState === this.videoElement.HAVE_ENOUGH_DATA) {
                this.processCurrentFrame();
            }
        }, 500); // Process every 500ms
        
        console.log('Plaque recognition system started');
    }
    
    /**
     * Stop the plaque recognition system
     */
    stop() {
        this.isActive = false;
        if (this.scanInterval) {
            clearInterval(this.scanInterval);
            this.scanInterval = null;
        }
        console.log('Plaque recognition system stopped');
    }
    
    /**
     * Process the current video frame for plaque detection
     */
    async processCurrentFrame() {
        if (!this.isActive || this.isProcessing) return;
        
        this.isProcessing = true;
        
        try {
            // Draw the current video frame to the canvas
            this.canvasContext.drawImage(
                this.videoElement, 
                0, 0, 
                this.canvasElement.width, 
                this.canvasElement.height
            );
            
            // Get image data from canvas for processing
            const imageData = this.canvasContext.getImageData(
                0, 0, 
                this.canvasElement.width, 
                this.canvasElement.height
            );
            
            // Process the image to detect plaques
            const results = await this.detectPlaques(imageData);
            
            // If plaques were detected, handle them
            if (results && results.length > 0) {
                this.handlePlaqueDetection(results);
            }
        } catch (error) {
            console.error('Error processing video frame:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    /**
     * Detect plaques in the provided image data
     * This is where integration with OpenCV.js or a similar library would occur
     * 
     * @param {ImageData} imageData - The image data to process
     * @returns {Array} Array of detected plaques with text and position
     */
    async detectPlaques(imageData) {
        // This is a placeholder for actual CV processing
        // In a real implementation, this would use OpenCV.js or TensorFlow.js
        
        // Mock implementation for demonstration
        const processedResults = await this.processImageWithOpenCV(imageData);
        
        return processedResults;
    }
    
    /**
     * Process image using OpenCV.js for plaque detection
     * 
     * @param {ImageData} imageData - The image data to process
     * @returns {Array} Detected plaque information
     */
    async processImageWithOpenCV(imageData) {
        // This is where we would implement the actual CV processing
        // For demonstration, this is a placeholder that would be replaced with actual OpenCV code
        
        console.log('Processing image for plaque detection...');
        
        // Placeholder for actual CV processing
        // In a real implementation, we would:
        // 1. Convert to grayscale
        // 2. Apply filters (bilateral filter)
        // 3. Detect edges (Canny)
        // 4. Find contours
        // 5. Filter for rectangular shapes
        // 6. Extract potential plaques
        // 7. Apply OCR to extract text
        
        // For demonstration purposes, we'll simulate finding a plaque
        // In a real implementation, this would be the result of actual image processing
        const simulatedDetection = this.simulatePlaqueDetection();
        
        return simulatedDetection;
    }
    
    /**
     * Simulates plaque detection for demonstration purposes
     * In a real implementation, this would be replaced by actual CV processing
     */
    simulatePlaqueDetection() {
        // This is just for demonstration
        // In a real implementation, detection would be based on actual image analysis
        
        // Simulate a 10% chance of detecting a plaque on each frame
        if (Math.random() < 0.1) {
            // Randomly select one of the mock plaques
            const plaqueIds = Object.keys(this.plaqueData);
            const randomPlaqueId = plaqueIds[Math.floor(Math.random() * plaqueIds.length)];
            
            return [{
                text: randomPlaqueId,
                confidence: 0.85 + (Math.random() * 0.1),
                boundingBox: {
                    x1: 100,
                    y1: 100,
                    x2: 200,
                    y2: 150
                }
            }];
        }
        
        return [];
    }
    
    /**
     * Handle detected plaques
     * 
     * @param {Array} detections - Array of detected plaques
     */
    handlePlaqueDetection(detections) {
        // Process each detected plaque
        for (const detection of detections) {
            // Only process if confidence is above threshold
            if (detection.confidence >= this.recognitionConfidence) {
                const plaqueText = detection.text.trim().toUpperCase();
                
                // Check if this plaque exists in our database
                if (this.plaqueData[plaqueText]) {
                    // Check if this is the same plaque we just recognized
                    if (this.lastRecognizedPlaque === plaqueText) {
                        // If we recognized this plaque recently, skip it
                        const timeSinceLastRecognition = Date.now() - this.lastRecognitionTime;
                        if (timeSinceLastRecognition < this.cooldownPeriod) {
                            continue;
                        }
                    }
                    
                    // Update last recognized plaque and time
                    this.lastRecognizedPlaque = plaqueText;
                    this.lastRecognitionTime = Date.now();
                    
                    // Draw a rectangle around the detected plaque
                    this.highlightDetectedPlaque(detection.boundingBox);
                    
                    // Call the recognition callback with the plaque data
                    const plaqueInfo = {
                        ...this.plaqueData[plaqueText],
                        detectedText: plaqueText,
                        confidence: detection.confidence,
                        position: detection.boundingBox
                    };
                    
                    console.log(`Recognized plaque: ${plaqueText} (${detection.confidence.toFixed(2)})`);
                    this.onPlaqueRecognized(plaqueInfo);
                }
            }
        }
    }
    
    /**
     * Highlight detected plaque on the canvas
     * 
     * @param {Object} boundingBox - The bounding box of the detected plaque
     */
    highlightDetectedPlaque(boundingBox) {
        // Draw rectangle on canvas around detected plaque
        this.canvasContext.strokeStyle = '#00FF00';
        this.canvasContext.lineWidth = 3;
        this.canvasContext.strokeRect(
            boundingBox.x1,
            boundingBox.y1,
            boundingBox.x2 - boundingBox.x1,
            boundingBox.y2 - boundingBox.y1
        );
        
        // Add a label
        this.canvasContext.fillStyle = 'rgba(0, 255, 0, 0.5)';
        this.canvasContext.fillRect(
            boundingBox.x1,
            boundingBox.y1 - 20,
            100,
            20
        );
        
        this.canvasContext.fillStyle = '#000000';
        this.canvasContext.font = '16px Arial';
        this.canvasContext.fillText(
            'Room Plaque',
            boundingBox.x1 + 5,
            boundingBox.y1 - 5
        );
    }
}

/**
 * Room Plaque-Based Navigation
 * 
 * This class handles navigation based on recognized room plaques
 */
class PlaqueBasedNavigation {
    /**
     * Initialize plaque-based navigation system
     * 
     * @param {Object} options - Configuration options
     * @param {HTMLElement} options.sceneElement - The A-Frame scene element
     * @param {NavigationGraph} options.navigationGraph - The navigation graph
     */
    constructor(options) {
        this.sceneElement = options.sceneElement;
        this.navigationGraph = options.navigationGraph;
        this.currentLocation = null;
        this.destinationLocation = null;
        this.pathFinder = new PathFinder(this.navigationGraph);
        this.markerManager = new ARMarkerManager(this.sceneElement);
        this.currentPath = [];
    }
    
    /**
     * Handle recognized plaque information
     * 
     * @param {Object} plaqueInfo - Information about the recognized plaque
     */
    handlePlaqueRecognition(plaqueInfo) {
        console.log('Navigation system received plaque recognition:', plaqueInfo);
        
        // Update current location based on recognized plaque
        this.currentLocation = {
            building: plaqueInfo.building,
            roomId: plaqueInfo.roomId
        };
        
        // If we have a destination set, update navigation path
        if (this.destinationLocation) {
            this.updateNavigationPath();
        } else {
            // Just show information about the current location
            this.showLocationInfo(plaqueInfo);
        }
    }
    
    /**
     * Set the destination for navigation
     * 
     * @param {String} buildingId - Building ID
     * @param {String} roomId - Room ID
     */
    setDestination(buildingId, roomId) {
        this.destinationLocation = {
            building: buildingId,
            roomId: roomId
        };
        
        console.log(`Destination set to ${buildingId} - ${roomId}`);
        
        // If we know our current location, update the path
        if (this.currentLocation) {
            this.updateNavigationPath();
        } else {
            console.log('Current location unknown. Scan a room plaque to start navigation.');
        }
    }
    
    /**
     * Update the navigation path based on current and destination locations
     */
    async updateNavigationPath() {
        if (!this.currentLocation || !this.destinationLocation) {
            console.error('Cannot update navigation path: missing current or destination location');
            return;
        }
        
        console.log(`Calculating path from ${this.currentLocation.building}/${this.currentLocation.roomId} to ${this.destinationLocation.building}/${this.destinationLocation.roomId}`);
        
        try {
            // Fetch the path from the server
            const response = await fetch(`/api/navigate-indoor?fromBuilding=${this.currentLocation.building}&fromRoom=${this.currentLocation.roomId}&toBuilding=${this.destinationLocation.building}&toRoom=${this.destinationLocation.roomId}`);
            
            if (response.ok) {
                const navigationData = await response.json();
                this.displayNavigationPath(navigationData);
            } else {
                console.error('Failed to fetch indoor navigation path');
            }
        } catch (error) {
            console.error('Error updating navigation path:', error);
        }
    }
    
    /**
     * Display the navigation path in AR
     * 
     * @param {Object} navigationData - Path and destination information
     */
    displayNavigationPath(navigationData) {
        // Clear any existing path
        this.markerManager.clearMarkers();
        this.currentPath = navigationData.path;
        
        // Create markers for the path
        navigationData.path.forEach((point, index) => {
            this.markerManager.createMarker(
                `path-${index}`,
                `${point.x} ${point.y} ${point.z || 0}`,
                'waypoint',
                {
                    lat: point.lat,
                    lng: point.lng
                }
            );
        });
        
        // Create destination marker
        const destination = navigationData.destination;
        this.markerManager.createMarker(
            'destination',
            `${destination.x} ${destination.y} ${destination.z || 0}`,
            'destination',
            {
                lat: destination.lat,
                lng: destination.lng,
                name: destination.name
            }
        );
        
        console.log(`Navigation path displayed with ${navigationData.path.length} points`);
    }
    
    /**
     * Show information about the current location
     * 
     * @param {Object} plaqueInfo - Information about the recognized plaque
     */
    showLocationInfo(plaqueInfo) {
        // Create an information marker at the current location
        this.markerManager.createMarker(
            'current-location',
            '0 0 0',
            'info',
            {
                content: `You are at: ${plaqueInfo.building} - ${plaqueInfo.roomId} (${plaqueInfo.description})`
            }
        );
    }
}

/**
 * Integration with A-Frame for AR plaque recognition
 */
AFRAME.registerComponent('plaque-recognizer', {
    schema: {
        active: {type: 'boolean', default: true},
        showDebugCanvas: {type: 'boolean', default: false}
    },
    
    init: function() {
        // Create video and canvas elements
        this.video = document.createElement('video');
        this.video.setAttribute('autoplay', '');
        this.video.setAttribute('playsinline', '');
        this.canvas = document.createElement('canvas');
        
        // Set up the canvas for debugging
        if (this.data.showDebugCanvas) {
            this.canvas.style.position = 'absolute';
            this.canvas.style.top = '0';
            this.canvas.style.left = '0';
            this.canvas.style.zIndex = '1000';
            this.canvas.style.width = '160px';
            this.canvas.style.height = '120px';
            this.canvas.style.border = '2px solid red';
            document.body.appendChild(this.canvas);
        }
        
        // Get scene element and navigation system
        const sceneEl = document.querySelector('a-scene');
        const navigationSystem = new PlaqueBasedNavigation({
            sceneElement: sceneEl,
            navigationGraph: window.campusNavigationGraph || new NavigationGraph()
        });
        
        // Initialize the plaque recognition system
        this.recognitionSystem = new PlaqueRecognitionSystem({
            videoElement: this.video,
            canvasElement: this.canvas,
            onPlaqueRecognized: (plaqueInfo) => {
                navigationSystem.handlePlaqueRecognition(plaqueInfo);
            }
        });
        
        // Expose navigation API to global scope for UI interaction
        window.plaqueNavigation = navigationSystem;
        
        // Start camera stream
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' }
            }).then((stream) => {
                this.video.srcObject = stream;
                console.log('Camera stream started');
            }).catch((error) => {
                console.error('Error accessing camera:', error);
            });
        } else {
            console.error('getUserMedia not supported');
        }
    },
    
    update: function() {
        // Toggle recognition system based on active property
        if (this.data.active) {
            this.recognitionSystem.start();
        } else {
            this.recognitionSystem.stop();
        }
        
        // Toggle debug canvas visibility
        if (this.canvas.parentNode) {
            this.canvas.style.display = this.data.showDebugCanvas ? 'block' : 'none';
        }
    },
    
    remove: function() {
        // Clean up when component is removed
        this.recognitionSystem.stop();
        
        // Stop video stream
        if (this.video.srcObject) {
            const tracks = this.video.srcObject.getTracks();
            tracks.forEach(track => track.stop());
        }
        
        // Remove debug canvas if it was added to the DOM
        if (this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
    }
}); 