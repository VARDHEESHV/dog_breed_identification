import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, render_template_string
from PIL import Image
import tensorflow as tf
import json
import h5py
import base64
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
MODEL_PATH = 'zero.h5'  # Default model path
TEMPLATE_FOLDER = 'templates'
STATIC_FOLDER = 'static'

# Create directories if they don't exist
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Dog breed labels
DOG_BREEDS = [
    'Affenpinscher', 'Afghan Hound', 'African Hunting Dog', 'Airedale Terrier',
    'American Staffordshire Terrier', 'Appenzeller Mountain Dog', 'Australian Terrier',
    'Basenji', 'Basset Hound', 'Beagle', 'Bedlington Terrier', 'Bernese Mountain Dog',
    'Black and Tan Coonhound', 'Blenheim Spaniel', 'Bloodhound', 'Bluetick Coonhound',
    'Border Collie', 'Border Terrier', 'Borzoi', 'Boston Terrier', 'Bouvier des Flandres',
    'Boxer', 'Brabancon Griffon', 'Briard', 'Brittany Spaniel', 'Bull Mastiff',
    'Cairn Terrier', 'Cardigan Welsh Corgi', 'Chesapeake Bay Retriever', 'Chihuahua',
    'Chow Chow', 'Clumber Spaniel', 'Cocker Spaniel', 'Collie', 'Curly-Coated Retriever',
    'Dandie Dinmont Terrier', 'Dhole', 'Dingo', 'Doberman Pinscher', 'English Foxhound',
    'English Setter', 'English Springer Spaniel', 'Entlebucher Mountain Dog', 'Eskimo Dog',
    'Flat-Coated Retriever', 'French Bulldog', 'German Shepherd', 'German Short-Haired Pointer',
    'Giant Schnauzer', 'Golden Retriever', 'Gordon Setter', 'Great Dane', 'Great Pyrenees',
    'Greater Swiss Mountain Dog', 'Groenendael', 'Ibizan Hound', 'Irish Setter',
    'Irish Terrier', 'Irish Water Spaniel', 'Irish Wolfhound', 'Italian Greyhound',
    'Japanese Spaniel', 'Keeshond', 'Australian Kelpie', 'Kerry Blue Terrier',
    'Komondor', 'Kuvasz', 'Labrador Retriever', 'Lakeland Terrier', 'Leonberger',
    'Lhasa Apso', 'Alaskan Malamute', 'Belgian Malinois', 'Maltese', 'Mexican Hairless Dog',
    'Miniature Pinscher', 'Miniature Poodle', 'Miniature Schnauzer', 'Newfoundland',
    'Norfolk Terrier', 'Norwegian Elkhound', 'Norwich Terrier', 'Old English Sheepdog',
    'Otterhound', 'Papillon', 'Pekingese', 'Pembroke Welsh Corgi', 'Pomeranian',
    'Pug', 'Redbone Coonhound', 'Rhodesian Ridgeback', 'Rottweiler', 'Saint Bernard',
    'Saluki', 'Samoyed', 'Schipperke', 'Scottish Terrier', 'Scottish Deerhound',
    'Sealyham Terrier', 'Shetland Sheepdog', 'Shih Tzu', 'Siberian Husky',
    'Silky Terrier', 'Soft-Coated Wheaten Terrier', 'Staffordshire Bull Terrier',
    'Standard Poodle', 'Standard Schnauzer', 'Sussex Spaniel', 'Tibetan Mastiff',
    'Tibetan Terrier', 'Toy Poodle', 'Toy Terrier', 'Vizsla', 'Walker Hound',
    'Weimaraner', 'Welsh Springer Spaniel', 'West Highland White Terrier', 'Whippet',
    'Wire-Haired Fox Terrier', 'Yorkshire Terrier'
]

# Global model variable
model = None
model_loaded = False

def recreate_vgg19_model(num_classes=120):
    """Recreate a VGG19-based model architecture"""
    base_model = tf.keras.applications.VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
        base_model,
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    
    return model

def extract_weights_from_h5(model_path):
    """Extract weights from H5 file manually"""
    weights = {}
    try:
        with h5py.File(model_path, 'r') as f:
            def extract_weights_recursive(name, obj):
                if isinstance(obj, h5py.Dataset):
                    weights[name] = obj[...]
            
            if 'model_weights' in f:
                f['model_weights'].visititems(extract_weights_recursive)
            else:
                f.visititems(extract_weights_recursive)
    except Exception as e:
        logger.error(f"Error extracting weights: {e}")
        return None
    
    return weights

def load_weights_to_model(model, weights_dict):
    """Load weights dictionary to model"""
    try:
        for layer in model.layers:
            layer_name = layer.name
            
            # Try different weight naming conventions
            possible_names = [
                f'model_weights/{layer_name}',
                f'{layer_name}',
                f'sequential/{layer_name}',
                f'model/{layer_name}'
            ]
            
            for possible_name in possible_names:
                # Look for kernel/bias weights
                kernel_key = f'{possible_name}/kernel:0'
                bias_key = f'{possible_name}/bias:0'
                
                if kernel_key in weights_dict or bias_key in weights_dict:
                    layer_weights = []
                    
                    if kernel_key in weights_dict:
                        layer_weights.append(weights_dict[kernel_key])
                    if bias_key in weights_dict:
                        layer_weights.append(weights_dict[bias_key])
                    
                    if layer_weights:
                        try:
                            layer.set_weights(layer_weights)
                            logger.info(f"Loaded weights for layer: {layer_name}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to set weights for {layer_name}: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading weights to model: {e}")
        return False

def load_model_with_enhanced_compatibility(model_path):
    """Enhanced model loading with multiple fallback strategies"""
    logger.info(f"Attempting to load model: {model_path}")
    
    # Method 1: Standard loading
    try:
        model = tf.keras.models.load_model(model_path)
        return model, "Standard loading successful"
    except Exception as e1:
        logger.warning(f"Standard loading failed: {str(e1)[:100]}...")
    
    # Method 2: Load with compile=False
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, "Loaded without compilation"
    except Exception as e2:
        logger.warning(f"Loading without compilation failed: {str(e2)[:100]}...")
    
    # Method 3: Recreate model and load weights
    try:
        logger.info("Attempting to recreate model architecture and load weights...")
        
        # Extract weights from H5 file
        weights_dict = extract_weights_from_h5(model_path)
        
        if weights_dict is None:
            raise Exception("Could not extract weights from H5 file")
        
        # Recreate model architecture
        model = recreate_vgg19_model(num_classes=len(DOG_BREEDS))
        
        # Load weights
        success = load_weights_to_model(model, weights_dict)
        
        if success:
            return model, "Recreated model with loaded weights"
        else:
            raise Exception("Failed to load weights to recreated model")
    
    except Exception as e3:
        logger.warning(f"Model recreation failed: {str(e3)[:100]}...")
    
    return None, "All loading methods failed"

def initialize_model():
    """Initialize the model at startup"""
    global model, model_loaded
    
    if os.path.exists(MODEL_PATH):
        try:
            model, load_method = load_model_with_enhanced_compatibility(MODEL_PATH)
            if model is not None:
                # Recompile model if needed
                try:
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    logger.info("Model recompiled successfully")
                except Exception as e:
                    logger.warning(f"Could not recompile model: {e}")
                
                model_loaded = True
                logger.info(f"Model loaded successfully: {load_method}")
            else:
                logger.error("Failed to load model")
                model_loaded = False
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            model_loaded = False
    else:
        logger.error(f"Model file not found: {MODEL_PATH}")
        model_loaded = False

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Routes
@app.route('/')
def index():
    """Serve the index page"""
    try:
        # Read the index HTML file
        index_path = r'D:\project\project\index_html\index_html.html'
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return html_content
        else:
            # Fallback if file doesn't exist
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dog Breed Classifier</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                    .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    .header { text-align: center; margin-bottom: 30px; }
                    .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; border-radius: 10px; margin-bottom: 20px; }
                    .upload-area:hover { border-color: #007bff; }
                    .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                    .btn:hover { background: #0056b3; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🐕 Dog Breed Classifier</h1>
                        <p>Upload an image to identify the dog breed!</p>
                    </div>
                    <form id="uploadForm" enctype="multipart/form-data">
                        <div class="upload-area">
                            <input type="file" id="imageFile" name="image" accept="image/*" required>
                            <p>Choose an image file or drag and drop here</p>
                        </div>
                        <button type="submit" class="btn">Classify Breed</button>
                    </form>
                </div>
                
                <script>
                document.getElementById('uploadForm').addEventListener('submit', function(e) {
                    e.preventDefault();
                    
                    const formData = new FormData();
                    const imageFile = document.getElementById('imageFile').files[0];
                    
                    if (!imageFile) {
                        alert('Please select an image file');
                        return;
                    }
                    
                    formData.append('image', imageFile);
                    
                    fetch('/predict', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            // Store results and redirect to results page
                            sessionStorage.setItem('predictionResults', JSON.stringify(data));
                            window.location.href = '/results';
                        } else {
                            alert('Error: ' + data.error);
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('An error occurred while processing the image');
                    });
                });
                </script>
            </body>
            </html>
            ''')
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        return f"Error loading index page: {e}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction"""
    try:
        if not model_loaded:
            return jsonify({'success': False, 'error': 'Model not loaded properly'})
        
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected'})
        
        # Process the image
        try:
            image = Image.open(file.stream)
            processed_image = preprocess_image(image)
            
            if processed_image is None:
                return jsonify({'success': False, 'error': 'Failed to process image'})
            
            # Make prediction
            predictions = model.predict(processed_image, verbose=0)
            
            # Get top 5 predictions
            top_indices = np.argsort(predictions[0])[-5:][::-1]
            top_breeds = [DOG_BREEDS[idx] if idx < len(DOG_BREEDS) else f"Class_{idx}" for idx in top_indices]
            top_confidences = [float(predictions[0][idx]) * 100 for idx in top_indices]
            
            # Convert image to base64 for display
            image_base64 = image_to_base64(image)
            
            return jsonify({
                'success': True,
                'predictions': [
                    {'breed': breed, 'confidence': conf} 
                    for breed, conf in zip(top_breeds, top_confidences)
                ],
                'image': image_base64,
                'top_prediction': {
                    'breed': top_breeds[0],
                    'confidence': top_confidences[0]
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({'success': False, 'error': f'Error processing image: {str(e)}'})
    
    except Exception as e:
        logger.error(f"Error in predict route: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/results')
def results():
    """Serve the results page"""
    try:
        # Read the predict HTML file
        predict_path = r'D:\project\project\predict_html\predict_html.html'
        if os.path.exists(predict_path):
            with open(predict_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return html_content
        else:
            # Fallback if file doesn't exist
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dog Breed Classifier - Results</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                    .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    .header { text-align: center; margin-bottom: 30px; }
                    .results-container { display: flex; gap: 20px; flex-wrap: wrap; }
                    .image-section { flex: 1; min-width: 300px; }
                    .results-section { flex: 1; min-width: 300px; }
                    .prediction-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; }
                    .prediction-list { margin-top: 20px; }
                    .prediction-item { display: flex; justify-content: space-between; align-items: center; padding: 10px; border-bottom: 1px solid #eee; }
                    .progress-bar { background: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden; margin-top: 5px; }
                    .progress-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.3s ease; }
                    .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; margin-top: 20px; }
                    .btn:hover { background: #0056b3; }
                    .uploaded-image { max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🐕 Dog Breed Classification Results</h1>
                    </div>
                    
                    <div class="results-container">
                        <div class="image-section">
                            <h3>📸 Uploaded Image</h3>
                            <img id="uploadedImage" class="uploaded-image" alt="Uploaded dog image">
                        </div>
                        
                        <div class="results-section">
                            <div class="prediction-card">
                                <h3>🏆 Top Prediction</h3>
                                <h2 id="topBreed"></h2>
                                <h3 id="topConfidence"></h3>
                            </div>
                            
                            <h3>📊 All Predictions</h3>
                            <div class="prediction-list" id="predictionList"></div>
                        </div>
                    </div>
                    
                    <div style="text-align: center;">
                        <a href="/" class="btn">🔄 Classify Another Image</a>
                    </div>
                </div>
                
                <script>
                // Load results from session storage
                const results = JSON.parse(sessionStorage.getItem('predictionResults') || '{}');
                
                if (results.success) {
                    // Display uploaded image
                    document.getElementById('uploadedImage').src = results.image;
                    
                    // Display top prediction
                    document.getElementById('topBreed').textContent = results.top_prediction.breed;
                    document.getElementById('topConfidence').textContent = results.top_prediction.confidence.toFixed(2) + '% confidence';
                    
                    // Display all predictions
                    const predictionList = document.getElementById('predictionList');
                    results.predictions.forEach((pred, index) => {
                        const item = document.createElement('div');
                        item.className = 'prediction-item';
                        item.innerHTML = `
                            <div>
                                <strong>${index + 1}. ${pred.breed}</strong>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${pred.confidence}%"></div>
                                </div>
                            </div>
                            <span>${pred.confidence.toFixed(2)}%</span>
                        `;
                        predictionList.appendChild(item);
                    });
                } else {
                    document.querySelector('.container').innerHTML = `
                        <div class="header">
                            <h1>❌ Error</h1>
                            <p>No prediction results found. Please try again.</p>
                            <a href="/" class="btn">Go Back</a>
                        </div>
                    `;
                }
                </script>
            </body>
            </html>
            ''')
    except Exception as e:
        logger.error(f"Error serving results page: {e}")
        return f"Error loading results page: {e}", 500

@app.route('/api/model-info')
def model_info():
    """Get model information"""
    try:
        if not model_loaded:
            return jsonify({'success': False, 'error': 'Model not loaded'})
        
        info = {
            'success': True,
            'model_loaded': model_loaded,
            'tensorflow_version': tf.__version__,
            'total_breeds': len(DOG_BREEDS),
            'model_path': MODEL_PATH
        }
        
        try:
            info['input_shape'] = str(model.input_shape)
            info['output_shape'] = str(model.output_shape)
            info['total_parameters'] = int(model.count_params())
            info['layers'] = len(model.layers)
        except Exception as e:
            info['model_info_error'] = str(e)
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/breeds')
def get_breeds():
    """Get list of all supported breeds"""
    return jsonify({
        'success': True,
        'breeds': DOG_BREEDS,
        'total': len(DOG_BREEDS)
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB'}), 413

if __name__ == '__main__':
    print("Initializing Dog Breed Classifier...")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Model Path: {MODEL_PATH}")
    
    # Initialize model
    initialize_model()
    
    if model_loaded:
        print("✅ Model loaded successfully!")
    else:
        print("❌ Model loading failed!")
        print("The application will still run, but predictions will not work.")
    
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    # Fix for Windows signal handling issue
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        print(f"Error starting with debug mode: {e}")
        print("Starting in production mode...")
        app.run(host='0.0.0.0', port=5000)

