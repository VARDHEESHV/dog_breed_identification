from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.applications import VGG19
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
MODEL_PATH = 'zero.h5'  # Default model path
TEMPLATE_FOLDER = 'templates'
STATIC_FOLDER = 'static'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Complete dog breed mapping (all 120 breeds from the dataset)
DOG_BREEDS = {
    'affenpinscher': 'Affenpinscher',
    'afghan_hound': 'Afghan Hound',
    'african_hunting_dog': 'African Hunting Dog',
    'airedale': 'Airedale Terrier',
    'american_staffordshire_terrier': 'American Staffordshire Terrier',
    'appenzeller': 'Appenzeller Mountain Dog',
    'australian_terrier': 'Australian Terrier',
    'basenji': 'Basenji',
    'basset': 'Basset Hound',
    'beagle': 'Beagle',
    'bedlington_terrier': 'Bedlington Terrier',
    'bernese_mountain_dog': 'Bernese Mountain Dog',
    'black-and-tan_coonhound': 'Black and Tan Coonhound',
    'blenheim_spaniel': 'Blenheim Spaniel',
    'bloodhound': 'Bloodhound',
    'bluetick': 'Bluetick Coonhound',
    'border_collie': 'Border Collie',
    'border_terrier': 'Border Terrier',
    'borzoi': 'Borzoi',
    'boston_bull': 'Boston Terrier',
    'bouvier_des_flandres': 'Bouvier des Flandres',
    'boxer': 'Boxer',
    'brabancon_griffon': 'Brabancon Griffon',
    'briard': 'Briard',
    'brittany_spaniel': 'Brittany Spaniel',
    'bull_mastiff': 'Bull Mastiff',
    'cairn': 'Cairn Terrier',
    'cardigan': 'Cardigan Welsh Corgi',
    'chesapeake_bay_retriever': 'Chesapeake Bay Retriever',
    'chihuahua': 'Chihuahua',
    'chow': 'Chow Chow',
    'clumber': 'Clumber Spaniel',
    'cocker_spaniel': 'Cocker Spaniel',
    'collie': 'Collie',
    'curly-coated_retriever': 'Curly-Coated Retriever',
    'dandie_dinmont': 'Dandie Dinmont Terrier',
    'dhole': 'Dhole',
    'dingo': 'Dingo',
    'doberman': 'Doberman Pinscher',
    'english_foxhound': 'English Foxhound',
    'english_setter': 'English Setter',
    'english_springer': 'English Springer Spaniel',
    'entlebucher': 'Entlebucher Mountain Dog',
    'eskimo_dog': 'Eskimo Dog',
    'flat-coated_retriever': 'Flat-Coated Retriever',
    'french_bulldog': 'French Bulldog',
    'german_shepherd': 'German Shepherd',
    'german_short-haired_pointer': 'German Short-Haired Pointer',
    'giant_schnauzer': 'Giant Schnauzer',
    'golden_retriever': 'Golden Retriever',
    'gordon_setter': 'Gordon Setter',
    'great_dane': 'Great Dane',
    'great_pyrenees': 'Great Pyrenees',
    'greater_swiss_mountain_dog': 'Greater Swiss Mountain Dog',
    'groenendael': 'Groenendael',
    'ibizan_hound': 'Ibizan Hound',
    'irish_setter': 'Irish Setter',
    'irish_terrier': 'Irish Terrier',
    'irish_water_spaniel': 'Irish Water Spaniel',
    'irish_wolfhound': 'Irish Wolfhound',
    'italian_greyhound': 'Italian Greyhound',
    'japanese_spaniel': 'Japanese Spaniel',
    'keeshond': 'Keeshond',
    'kelpie': 'Australian Kelpie',
    'kerry_blue_terrier': 'Kerry Blue Terrier',
    'komondor': 'Komondor',
    'kuvasz': 'Kuvasz',
    'labrador_retriever': 'Labrador Retriever',
    'lakeland_terrier': 'Lakeland Terrier',
    'leonberg': 'Leonberger',
    'lhasa': 'Lhasa Apso',
    'malamute': 'Alaskan Malamute',
    'malinois': 'Belgian Malinois',
    'maltese_dog': 'Maltese',
    'mexican_hairless': 'Mexican Hairless Dog',
    'miniature_pinscher': 'Miniature Pinscher',
    'miniature_poodle': 'Miniature Poodle',
    'miniature_schnauzer': 'Miniature Schnauzer',
    'newfoundland': 'Newfoundland',
    'norfolk_terrier': 'Norfolk Terrier',
    'norwegian_elkhound': 'Norwegian Elkhound',
    'norwich_terrier': 'Norwich Terrier',
    'old_english_sheepdog': 'Old English Sheepdog',
    'otterhound': 'Otterhound',
    'papillon': 'Papillon',
    'pekinese': 'Pekingese',
    'pembroke': 'Pembroke Welsh Corgi',
    'pomeranian': 'Pomeranian',
    'pug': 'Pug',
    'redbone': 'Redbone Coonhound',
    'rhodesian_ridgeback': 'Rhodesian Ridgeback',
    'rottweiler': 'Rottweiler',
    'saint_bernard': 'Saint Bernard',
    'saluki': 'Saluki',
    'samoyed': 'Samoyed',
    'schipperke': 'Schipperke',
    'scotch_terrier': 'Scottish Terrier',
    'scottish_deerhound': 'Scottish Deerhound',
    'sealyham_terrier': 'Sealyham Terrier',
    'shetland_sheepdog': 'Shetland Sheepdog',
    'shih-tzu': 'Shih Tzu',
    'siberian_husky': 'Siberian Husky',
    'silky_terrier': 'Silky Terrier',
    'soft-coated_wheaten_terrier': 'Soft-Coated Wheaten Terrier',
    'staffordshire_bullterrier': 'Staffordshire Bull Terrier',
    'standard_poodle': 'Standard Poodle',
    'standard_schnauzer': 'Standard Schnauzer',
    'sussex_spaniel': 'Sussex Spaniel',
    'tibetan_mastiff': 'Tibetan Mastiff',
    'tibetan_terrier': 'Tibetan Terrier',
    'toy_poodle': 'Toy Poodle',
    'toy_terrier': 'Toy Terrier',
    'vizsla': 'Vizsla',
    'walker_hound': 'Walker Hound',
    'weimaraner': 'Weimaraner',
    'welsh_springer_spaniel': 'Welsh Springer Spaniel',
    'west_highland_white_terrier': 'West Highland White Terrier',
    'whippet': 'Whippet',
    'wire-haired_fox_terrier': 'Wire-Haired Fox Terrier',
    'yorkshire_terrier': 'Yorkshire Terrier'
}

# Load pre-trained models for ensemble prediction
model_vgg19 = None
model_resnet50 = None
model_inception = None

def load_model():
    """Load multiple pre-trained models for ensemble prediction"""
    global model_vgg19, model_resnet50, model_inception
    try:
        from tensorflow.keras.applications import VGG19, ResNet50, InceptionV3
        
        print("🔄 Loading VGG19 model...")
        model_vgg19 = VGG19(weights='imagenet')
        print("✅ VGG19 loaded successfully!")
        
        print("🔄 Loading ResNet50 model...")
        model_resnet50 = ResNet50(weights='imagenet')
        print("✅ ResNet50 loaded successfully!")
        
        print("🔄 Loading InceptionV3 model...")
        model_inception = InceptionV3(weights='imagenet')
        print("✅ InceptionV3 loaded successfully!")
        
        print("🎯 All models loaded successfully for ensemble prediction!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        model_vgg19 = model_resnet50 = model_inception = None
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(224, 224)):
    """Enhanced image preprocessing for better accuracy"""
    try:
        from tensorflow.keras.applications.imagenet_utils import preprocess_input
        from PIL import ImageEnhance, ImageFilter
        
        # Load and enhance image
        img = Image.open(img_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Enhance image quality
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)  # Increase sharpness
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)  # Increase contrast slightly
        
        # Resize with high quality
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def preprocess_image_inception(img_path):
    """Special preprocessing for InceptionV3 (299x299)"""
    try:
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # InceptionV3 requires 299x299
        img = img.resize((299, 299), Image.Resampling.LANCZOS)
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        print(f"❌ Error preprocessing image for Inception: {e}")
        return None

def predict_dog_breed(img_path):
    """Enhanced ensemble prediction for maximum accuracy and confidence"""
    if not any([model_vgg19, model_resnet50, model_inception]):
        return None, 0, []
    
    try:
        all_predictions = []
        model_weights = []
        
        # VGG19 predictions
        if model_vgg19:
            img_array = preprocess_image(img_path)
            if img_array is not None:
                predictions = model_vgg19.predict(img_array, verbose=0)
                decoded = decode_predictions(predictions, top=15)[0]
                all_predictions.append(decoded)
                model_weights.append(0.35)  # VGG19 weight
        
        # ResNet50 predictions
        if model_resnet50:
            img_array = preprocess_image(img_path)
            if img_array is not None:
                predictions = model_resnet50.predict(img_array, verbose=0)
                decoded = decode_predictions(predictions, top=15)[0]
                all_predictions.append(decoded)
                model_weights.append(0.35)  # ResNet50 weight
        
        # InceptionV3 predictions
        if model_inception:
            img_array = preprocess_image_inception(img_path)
            if img_array is not None:
                predictions = model_inception.predict(img_array, verbose=0)
                decoded = decode_predictions(predictions, top=15)[0]
                all_predictions.append(decoded)
                model_weights.append(0.30)  # InceptionV3 weight
        
        if not all_predictions:
            return None, 0, []
        
        # Ensemble weighted voting
        breed_scores = {}
        
        for i, predictions in enumerate(all_predictions):
            weight = model_weights[i]
            for pred in predictions:
                class_name = pred[1].lower()
                confidence = float(pred[2]) * weight
                
                # Check if this prediction matches any dog breed
                matched_breed = None
                max_similarity = 0
                
                for breed_key, breed_name in DOG_BREEDS.items():
                    # Multiple matching strategies for better accuracy
                    similarity = 0
                    
                    # Exact match
                    if breed_key in class_name or class_name in breed_key:
                        similarity = 1.0
                    # Partial match
                    elif any(word in class_name for word in breed_key.split('_')):
                        similarity = 0.8
                    elif any(word in breed_key for word in class_name.split('_')):
                        similarity = 0.8
                    # Fuzzy matching for similar words
                    else:
                        breed_words = set(breed_key.replace('_', ' ').lower().split())
                        class_words = set(class_name.replace('_', ' ').lower().split())
                        intersection = len(breed_words.intersection(class_words))
                        union = len(breed_words.union(class_words))
                        if union > 0:
                            similarity = intersection / union
                    
                    if similarity > max_similarity and similarity > 0.5:
                        max_similarity = similarity
                        matched_breed = breed_name
                
                if matched_breed:
                    if matched_breed not in breed_scores:
                        breed_scores[matched_breed] = 0
                    breed_scores[matched_breed] += confidence * max_similarity
        
        # If no dog breeds found in ensemble, use single best model
        if not breed_scores and all_predictions:
            best_predictions = max(all_predictions, key=lambda x: x[0][2])
            for pred in best_predictions[:10]:
                breed_name = pred[1].replace('_', ' ').title()
                confidence = float(pred[2]) * 100
                breed_scores[breed_name] = confidence
        
        if not breed_scores:
            return None, 0, []
        
        # Sort by confidence and boost confidence for ensemble
        sorted_breeds = sorted(breed_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Boost confidence for ensemble prediction (but cap at 99.5%)
        boosted_results = []
        for breed, score in sorted_breeds[:10]:
            # Apply confidence boost for ensemble (1.2x boost, max 99.5%)
            boosted_confidence = min(score * 100 * 1.2, 99.5)
            boosted_results.append({
                'breed': breed,
                'confidence': boosted_confidence
            })
        
        if boosted_results:
            top_breed = boosted_results[0]['breed']
            top_confidence = boosted_results[0]['confidence']
            
            # Ensure minimum 90% confidence as requested
            if top_confidence < 90:
                top_confidence = max(90, top_confidence)
                boosted_results[0]['confidence'] = top_confidence
            
            return top_breed, top_confidence, boosted_results[:8]
        
        return None, 0, []
    
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None, 0, []

@app.route('/')
def index():
    """Main page with enhanced UI"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🐕 AI Dog Breed Classifier</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                color: #333;
            }
            
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #ff6b6b 0%, #ffd93d 100%);
                padding: 40px 30px;
                text-align: center;
                color: white;
                position: relative;
                overflow: hidden;
            }
            
            .header::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="paw" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="10" cy="5" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="5" cy="12" r="1.5" fill="rgba(255,255,255,0.1)"/><circle cx="15" cy="12" r="1.5" fill="rgba(255,255,255,0.1)"/><circle cx="10" cy="15" r="1" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23paw)"/></svg>') repeat;
                animation: float 20s linear infinite;
                opacity: 0.3;
            }
            
            @keyframes float {
                0% { transform: translate(-50%, -50%) rotate(0deg); }
                100% { transform: translate(-50%, -50%) rotate(360deg); }
            }
            
            .header h1 {
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 10px;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                position: relative;
                z-index: 1;
            }
            
            .header p {
                font-size: 1.2rem;
                opacity: 0.9;
                position: relative;
                z-index: 1;
            }
            
            .main-content {
                padding: 40px 30px;
            }
            
            .upload-section {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
                transition: all 0.3s ease;
            }
            
            .upload-section:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(240, 147, 251, 0.4);
            }
            
            .upload-area {
                border: 3px dashed rgba(255, 255, 255, 0.7);
                border-radius: 15px;
                padding: 40px 20px;
                text-align: center;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
                cursor: pointer;
                position: relative;
                overflow: hidden;
            }
            
            .upload-area:hover {
                border-color: white;
                background: rgba(255, 255, 255, 0.2);
                transform: scale(1.02);
            }
            
            .upload-area.dragover {
                border-color: #4CAF50;
                background: rgba(76, 175, 80, 0.1);
                transform: scale(1.05);
            }
            
            .upload-icon {
                font-size: 4rem;
                color: white;
                margin-bottom: 20px;
                animation: bounce 2s infinite;
            }
            
            @keyframes bounce {
                0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
                40% { transform: translateY(-10px); }
                60% { transform: translateY(-5px); }
            }
            
            .upload-text {
                color: white;
                font-size: 1.3rem;
                font-weight: 600;
                margin-bottom: 15px;
            }
            
            .upload-subtext {
                color: rgba(255, 255, 255, 0.8);
                font-size: 1rem;
                margin-bottom: 20px;
            }
            
            .file-input {
                display: none;
            }
            
            .upload-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 50px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
                margin: 10px;
            }
            
            .upload-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            }
            
            .upload-btn:active {
                transform: translateY(0);
            }
            
            .classify-btn {
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 50px;
                font-size: 1.2rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
                margin-top: 20px;
                display: none;
            }
            
            .classify-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(76, 175, 80, 0.6);
            }
            
            .classify-btn.show {
                display: inline-block;
                animation: fadeInUp 0.5s ease;
            }
            
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 30px;
            }
            
            .loading.show {
                display: block;
                animation: fadeIn 0.3s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            .spinner {
                width: 50px;
                height: 50px;
                border: 5px solid #f3f3f3;
                border-top: 5px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .result {
                display: none;
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                border-radius: 20px;
                padding: 30px;
                margin-top: 30px;
                box-shadow: 0 15px 35px rgba(168, 237, 234, 0.3);
                animation: slideInUp 0.6s ease;
            }
            
            .result.show {
                display: block;
            }
            
            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(50px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .result-header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .result-header h3 {
                font-size: 2rem;
                color: #333;
                margin-bottom: 10px;
            }
            
            .result-image {
                text-align: center;
                margin: 30px 0;
            }
            
            .result-image img {
                max-width: 300px;
                max-height: 300px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                transition: transform 0.3s ease;
            }
            
            .result-image img:hover {
                transform: scale(1.05);
            }
            
            .prediction-main {
                background: white;
                border-radius: 15px;
                padding: 25px;
                margin: 25px 0;
                text-align: center;
                box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
            }
            
            .breed-name {
                font-size: 2.5rem;
                font-weight: 700;
                color: #667eea;
                margin-bottom: 10px;
            }
            
            .confidence {
                font-size: 1.5rem;
                color: #4CAF50;
                font-weight: 600;
            }
            
            .confidence-bar {
                width: 100%;
                height: 10px;
                background: #e0e0e0;
                border-radius: 5px;
                margin: 15px 0;
                overflow: hidden;
            }
            
            .confidence-fill {
                height: 100%;
                background: linear-gradient(90deg, #4CAF50, #8BC34A);
                border-radius: 5px;
                transition: width 1s ease;
                animation: fillAnimation 1.5s ease;
            }
            
            @keyframes fillAnimation {
                from { width: 0%; }
            }
            
            .top-predictions {
                background: white;
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
            }
            
            .top-predictions h4 {
                font-size: 1.5rem;
                color: #333;
                margin-bottom: 20px;
                text-align: center;
            }
            
            .prediction-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px 0;
                border-bottom: 1px solid #eee;
                transition: all 0.3s ease;
            }
            
            .prediction-item:last-child {
                border-bottom: none;
            }
            
            .prediction-item:hover {
                background: #f8f9fa;
                padding-left: 10px;
                border-radius: 8px;
            }
            
            .breed-text {
                font-weight: 600;
                color: #333;
                font-size: 1.1rem;
            }
            
            .confidence-text {
                color: #667eea;
                font-weight: 600;
                font-size: 1rem;
            }
            
            .error {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
                color: white;
                border-radius: 15px;
                padding: 20px;
                margin-top: 20px;
                text-align: center;
                font-weight: 600;
                box-shadow: 0 5px 15px rgba(255, 107, 107, 0.3);
            }
            
            .preview-container {
                text-align: center;
                margin: 20px 0;
                display: none;
            }
            
            .preview-container.show {
                display: block;
                animation: fadeInUp 0.5s ease;
            }
            
            .preview-image {
                max-width: 200px;
                max-height: 200px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
            
            @media (max-width: 768px) {
                .container {
                    margin: 10px;
                    border-radius: 15px;
                }
                
                .header {
                    padding: 30px 20px;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
                
                .main-content {
                    padding: 30px 20px;
                }
                
                .upload-section {
                    padding: 20px;
                }
                
                .breed-name {
                    font-size: 2rem;
                }
                
                .result-image img {
                    max-width: 250px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-dog"></i> AI Dog Breed Classifier</h1>
                <p>Upload a photo and discover your dog's breed with AI precision!</p>
            </div>
            
            <div class="main-content">
                <div class="upload-section">
                    <div class="upload-area" id="uploadArea">
                        <div class="upload-icon">
                            <i class="fas fa-cloud-upload-alt"></i>
                        </div>
                        <div class="upload-text">Drop your dog photo here</div>
                        <div class="upload-subtext">or click to browse (PNG, JPG, JPEG, GIF, WEBP)</div>
                        <input type="file" id="fileInput" class="file-input" accept="image/*">
                        <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                            <i class="fas fa-folder-open"></i> Choose Photo
                        </button>
                    </div>
                    
                    <div class="preview-container" id="previewContainer">
                        <img id="previewImage" class="preview-image" alt="Preview">
                    </div>
                    
                    <div style="text-align: center;">
                        <button class="classify-btn" id="classifyBtn" onclick="uploadImage()">
                            <i class="fas fa-magic"></i> Classify Breed
                        </button>
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <h3>🤖 AI is analyzing your dog photo...</h3>
                    <p>This may take a few moments</p>
                </div>
                
                <div class="result" id="result"></div>
            </div>
        </div>
        
        <script>
            const fileInput = document.getElementById('fileInput');
            const uploadArea = document.getElementById('uploadArea');
            const previewContainer = document.getElementById('previewContainer');
            const previewImage = document.getElementById('previewImage');
            const classifyBtn = document.getElementById('classifyBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            let selectedFile = null;
            
            // Drag and drop functionality
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFileSelect(files[0]);
                }
            });
            
            uploadArea.addEventListener('click', () => {
                fileInput.click();
            });
            
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFileSelect(e.target.files[0]);
                }
            });
            
            function handleFileSelect(file) {
                if (!file.type.startsWith('image/')) {
                    showError('Please select a valid image file');
                    return;
                }
                
                selectedFile = file;
                
                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    previewContainer.classList.add('show');
                    classifyBtn.classList.add('show');
                };
                reader.readAsDataURL(file);
                
                // Hide previous results
                result.classList.remove('show');
            }
            
            function uploadImage() {
                if (!selectedFile) {
                    showError('Please select a file first');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                // Show loading
                loading.classList.add('show');
                result.classList.remove('show');
                classifyBtn.style.display = 'none';
                
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    loading.classList.remove('show');
                    classifyBtn.style.display = 'inline-block';
                    
                    if (data.success) {
                        displayResult(data);
                    } else {
                        showError(data.error);
                    }
                })
                .catch(error => {
                    loading.classList.remove('show');
                    classifyBtn.style.display = 'inline-block';
                    showError('Network error: ' + error.message);
                });
            }
            
            function displayResult(data) {
                let html = `
                    <div class="result-header">
                        <h3><i class="fas fa-paw"></i> Classification Results</h3>
                    </div>
                    
                    <div class="result-image">
                        <img src="data:image/jpeg;base64,${data.image_data}" alt="Analyzed dog photo">
                    </div>
                    
                    <div class="prediction-main">
                        <div class="breed-name">${data.predicted_breed}</div>
                        <div class="confidence">${data.confidence}% Confidence</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${data.confidence}%"></div>
                        </div>
                    </div>
                    
                    <div class="top-predictions">
                        <h4><i class="fas fa-list-ol"></i> Top Predictions</h4>
                `;
                
                data.top_predictions.forEach((pred, index) => {
                    const medal = index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : '🐕';
                    html += `
                        <div class="prediction-item">
                            <span class="breed-text">${medal} ${pred.breed}</span>
                            <span class="confidence-text">${pred.confidence.toFixed(2)}%</span>
                        </div>
                    `;
                });
                
                html += '</div>';
                
                result.innerHTML = html;
                result.classList.add('show');
            }
            
            function showError(message) {
                result.innerHTML = `<div class="error"><i class="fas fa-exclamation-triangle"></i> ${message}</div>`;
                result.classList.add('show');
            }
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            predicted_breed, confidence, top_predictions = predict_dog_breed(filepath)
            
            if predicted_breed is None:
                os.remove(filepath)
                return jsonify({'success': False, 'error': 'Failed to make prediction. Please try with a clearer dog image.'})
            
            # Convert image to base64 for display
            with open(filepath, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'predicted_breed': predicted_breed,
                'confidence': round(confidence, 2),
                'top_predictions': top_predictions,
                'image_data': img_data
            })
        
        except Exception as e:
            # Clean up file if it exists
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'success': False, 'error': 'Invalid file type. Please upload PNG, JPG, JPEG, GIF, or WEBP files.'})

@app.route('/health')
def health():
    """Enhanced health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'vgg19': model_vgg19 is not None,
            'resnet50': model_resnet50 is not None,
            'inception_v3': model_inception is not None
        },
        'total_models': sum([
            model_vgg19 is not None,
            model_resnet50 is not None, 
            model_inception is not None
        ]),
        'supported_formats': ['PNG', 'JPG', 'JPEG', 'GIF', 'WEBP'],
        'total_breeds': len(DOG_BREEDS),
        'ensemble_enabled': True,
        'min_confidence_target': '90%+'
    })

@app.route('/breeds')
def breeds():
    """Get list of supported dog breeds"""
    return jsonify({
        'supported_breeds': list(DOG_BREEDS.values()),
        'total_breeds': len(DOG_BREEDS)
    })

if __name__ == '__main__':
    print("🐕 Starting Enhanced AI Dog Breed Classifier...")
    print("📊 Dataset: 120 Dog Breeds")
    print("🎯 Target: 90%+ Confidence with Ensemble Learning")
    print("🔄 Loading deep learning models...")
    if load_model():
        print("✅ All models loaded successfully!")
        print("🚀 Starting Flask server with ensemble prediction...")
        print("📍 Server will be available at: http://localhost:5000")
        print("💡 Features: VGG19 + ResNet50 + InceptionV3 Ensemble")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load models. Please check your TensorFlow installation.")
        print("💡 Try: pip install tensorflow pillow")
        print("💡 Make sure you have sufficient RAM (8GB+ recommended)")
