/**
 * OpenCV Room Plaque Detector
 * 
 * This module implements computer vision techniques for detecting and recognizing
 * room plaques using OpenCV.js. It handles the image processing pipeline from
 * capturing frames to extracting text from plaques.
 * 
 * Based on techniques found in the OpenCV object recognition literature.
 * @see https://medium.com/@amit25173/opencv-object-recognition-642c8cf8379b
 */

// OpenCV modules to use when the library is loaded
const CV_MODULES = ['core', 'imgproc', 'features2d', 'calib3d', 'text', 'objdetect'];

class OpenCVPlaqueDetector {
    /**
     * Initialize the plaque detector
     * 
     * @param {Object} options - Configuration options
     * @param {Function} options.onPlaqueDetected - Callback when a plaque is detected
     * @param {Function} options.onTextRecognized - Callback when text is recognized
     * @param {HTMLElement} options.debugCanvas - Optional canvas for debug visualization
     */
    constructor(options = {}) {
        this.onPlaqueDetected = options.onPlaqueDetected || (() => {});
        this.onTextRecognized = options.onTextRecognized || (() => {});
        this.debugCanvas = options.debugCanvas || null;
        this.isOpenCVLoaded = false;
        this.isProcessing = false;
        this.confidence = options.confidence || 0.7;
        
        // Initialize OpenCV.js
        this.initOpenCV();
    }
    
    /**
     * Initialize OpenCV.js by loading the library
     */
    initOpenCV() {
        // Check if OpenCV is already loaded
        if (window.cv) {
            this.isOpenCVLoaded = true;
            console.log('OpenCV.js is already loaded');
            return;
        }
        
        console.log('Loading OpenCV.js...');
        
        // Add OpenCV.js script to the page
        const script = document.createElement('script');
        script.setAttribute('async', '');
        script.setAttribute('type', 'text/javascript');
        script.addEventListener('load', () => {
            console.log('OpenCV.js script loaded');
            
            // Initialize OpenCV modules when ready
            window.Module = {
                onRuntimeInitialized: () => {
                    this.isOpenCVLoaded = true;
                    console.log('OpenCV.js runtime initialized');
                }
            };
        });
        script.addEventListener('error', () => {
            console.error('Failed to load OpenCV.js');
        });
        
        // Load from CDN
        script.src = 'https://docs.opencv.org/4.7.0/opencv.js';
        document.body.appendChild(script);
    }
    
    /**
     * Process an image to detect plaques
     * 
     * @param {ImageData|HTMLImageElement|HTMLCanvasElement} image - The image to process
     * @returns {Promise<Array>} Array of detected plaques with bounding boxes and text
     */
    async detectPlaques(image) {
        if (!this.isOpenCVLoaded || this.isProcessing) {
            return [];
        }
        
        this.isProcessing = true;
        let results = [];
        
        try {
            // Convert input to OpenCV Mat
            const mat = this.imageToMat(image);
            
            // Apply processing pipeline
            const gray = new cv.Mat();
            cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
            
            // Apply bilateral filter to reduce noise while preserving edges
            const filtered = new cv.Mat();
            cv.bilateralFilter(gray, filtered, 11, 17, 17);
            
            // Apply Canny edge detection
            const edges = new cv.Mat();
            cv.Canny(filtered, edges, 30, 200);
            
            // Find contours
            const contours = new cv.MatVector();
            const hierarchy = new cv.Mat();
            cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            
            // Draw debug visualization if canvas provided
            if (this.debugCanvas) {
                cv.cvtColor(edges, mat, cv.COLOR_GRAY2RGBA);
                cv.drawContours(mat, contours, -1, [0, 255, 0, 255], 2);
                this.showDebugImage(mat);
            }
            
            // Process each contour to find potential plaques
            for (let i = 0; i < contours.size(); i++) {
                const contour = contours.get(i);
                const perimeter = cv.arcLength(contour, true);
                const approx = new cv.Mat();
                cv.approxPolyDP(contour, approx, 0.04 * perimeter, true);
                
                // Plaques are typically rectangular, so we look for quadrilaterals
                if (approx.rows === 4) {
                    // Check aspect ratio and size
                    const rect = cv.boundingRect(contour);
                    const aspectRatio = rect.width / rect.height;
                    
                    // Room plaques typically have aspect ratios between 1.5 and 4
                    if (aspectRatio > 1.5 && aspectRatio < 4 && rect.width > 50 && rect.height > 20) {
                        // Extract the plaque region
                        const roi = gray.roi(rect);
                        
                        // Prepare for OCR (increase contrast)
                        const processedRoi = new cv.Mat();
                        cv.threshold(roi, processedRoi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
                        
                        // For debug visualization
                        if (this.debugCanvas) {
                            const debugRoi = new cv.Mat();
                            cv.cvtColor(processedRoi, debugRoi, cv.COLOR_GRAY2RGBA);
                            this.showDebugImage(debugRoi, false);
                            debugRoi.delete();
                        }
                        
                        // In a real implementation, we would perform OCR here
                        // For this demonstration, we'll simulate text recognition
                        const recognizedText = await this.simulateTextRecognition(processedRoi);
                        
                        if (recognizedText) {
                            results.push({
                                text: recognizedText,
                                confidence: 0.85,
                                boundingBox: {
                                    x1: rect.x,
                                    y1: rect.y,
                                    x2: rect.x + rect.width,
                                    y2: rect.y + rect.height
                                }
                            });
                            
                            // Call the detection callback
                            this.onPlaqueDetected({
                                text: recognizedText,
                                confidence: 0.85,
                                rect: rect
                            });
                        }
                        
                        // Clean up
                        roi.delete();
                        processedRoi.delete();
                    }
                }
                
                approx.delete();
                contour.delete();
            }
            
            // Clean up
            mat.delete();
            gray.delete();
            filtered.delete();
            edges.delete();
            contours.delete();
            hierarchy.delete();
            
        } catch (error) {
            console.error('Error in OpenCV processing:', error);
        } finally {
            this.isProcessing = false;
        }
        
        return results;
    }
    
    /**
     * Convert an image to an OpenCV Mat
     * 
     * @param {ImageData|HTMLImageElement|HTMLCanvasElement} image - The input image
     * @returns {cv.Mat} OpenCV Mat object
     */
    imageToMat(image) {
        if (image instanceof ImageData) {
            return cv.matFromImageData(image);
        } else if (image instanceof HTMLImageElement || image instanceof HTMLCanvasElement) {
            const mat = cv.imread(image);
            return mat;
        } else {
            throw new Error('Unsupported image format');
        }
    }
    
    /**
     * Show debug visualization on the canvas
     * 
     * @param {cv.Mat} mat - The matrix to display
     * @param {boolean} clearCanvas - Whether to clear the canvas first
     */
    showDebugImage(mat, clearCanvas = true) {
        if (!this.debugCanvas) return;
        
        if (clearCanvas) {
            const ctx = this.debugCanvas.getContext('2d');
            ctx.clearRect(0, 0, this.debugCanvas.width, this.debugCanvas.height);
        }
        
        cv.imshow(this.debugCanvas, mat);
    }
    
    /**
     * Simulate text recognition from an image
     * In a real implementation, this would use a proper OCR library
     * 
     * @param {cv.Mat} image - The preprocessed image
     * @returns {Promise<string>} The recognized text
     */
    async simulateTextRecognition(image) {
        // This is just a placeholder for demonstration
        // In a real implementation, we would use Tesseract.js or EasyOCR.js
        return new Promise(resolve => {
            // Simulate processing time
            setTimeout(() => {
                // Generate a random room number for demonstration
                const buildings = ['M', 'S', 'L'];
                const building = buildings[Math.floor(Math.random() * buildings.length)];
                const number = Math.floor(Math.random() * 400) + 100;
                resolve(`${building}${number}`);
            }, 100);
        });
    }
    
    /**
     * Process a video frame for plaque detection
     * 
     * @param {HTMLVideoElement} videoElement - The video element with the camera feed
     * @param {HTMLCanvasElement} canvasElement - Canvas for processing
     * @returns {Promise<Array>} Array of detected plaques
     */
    async processVideoFrame(videoElement, canvasElement) {
        if (!this.isOpenCVLoaded || this.isProcessing) {
            return [];
        }
        
        // Draw the current video frame to the canvas
        const ctx = canvasElement.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        
        // Get the image data from the canvas
        const imageData = ctx.getImageData(0, 0, canvasElement.width, canvasElement.height);
        
        // Process the image to detect plaques
        return this.detectPlaques(imageData);
    }
}

// Export the class
if (typeof module !== 'undefined' && module.exports) {
    module.exports = OpenCVPlaqueDetector;
} else {
    window.OpenCVPlaqueDetector = OpenCVPlaqueDetector;
} 