"""
ROBLOX CHARACTER RECOGNITION - TRAINER & TESTER

This script provides:
1. Model training with data augmentation
2. Model testing and evaluation
3. Real-time screenshot testing
4. Performance analysis and visualization
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pyautogui
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow")
    TF_AVAILABLE = False

class RobloxCharacterTrainer:
    def __init__(self, data_dir="roblox_training_data"):
        self.data_dir = data_dir
        self.dataset_dir = f"{data_dir}/dataset"
        self.models_dir = f"{data_dir}/models"
        self.screenshots_dir = f"{data_dir}/screenshots"
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load character classes
        self.character_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-']
        self.num_classes = len(self.character_classes)
        
        # Model
        self.model = None
        
        # Scan region (same as labeler)
        self.scan_region = {
            'x': 792,
            'y': 484, 
            'width': 143,
            'height': 50
        }
    
    def load_dataset(self):
        """Load the exported dataset"""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available")
            return False
        
        try:
            # Check if dataset exists
            X_path = f"{self.dataset_dir}/X_train.npy"
            y_path = f"{self.dataset_dir}/y_train.npy"
            
            if not os.path.exists(X_path) or not os.path.exists(y_path):
                print(f"❌ Dataset not found. Please export dataset from labeler first.")
                print(f"Expected files: {X_path}, {y_path}")
                return False
            
            # Load data
            X = np.load(X_path)
            y = np.load(y_path)
            
            print(f"✅ Loaded dataset: {X.shape[0]} samples")
            print(f"📊 Image shape: {X.shape[1:]}")
            print(f"🏷️ Classes: {self.num_classes}")
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📚 Training samples: {self.X_train.shape[0]}")
            print(f"🧪 Testing samples: {self.X_test.shape[0]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return False
    
    def augment_data(self, X, y, augment_factor=3):
        """Apply data augmentation to increase dataset size"""
        print("🔄 Applying data augmentation...")
        
        augmented_X = []
        augmented_y = []
        
        for i in range(len(X)):
            img = X[i].reshape(32, 32)
            label = y[i]
            
            # Original image
            augmented_X.append(img)
            augmented_y.append(label)
            
            # Generate augmented versions
            for _ in range(augment_factor):
                aug_img = img.copy()
                
                # Random rotation (-15 to 15 degrees)
                angle = np.random.uniform(-15, 15)
                center = (16, 16)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                aug_img = cv2.warpAffine(aug_img, M, (32, 32))
                
                # Random translation (-3 to 3 pixels)
                tx = np.random.randint(-3, 4)
                ty = np.random.randint(-3, 4)
                M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
                aug_img = cv2.warpAffine(aug_img, M, (32, 32))
                
                # Random noise
                noise = np.random.normal(0, 0.05, aug_img.shape)
                aug_img = np.clip(aug_img + noise, 0, 1)
                
                # Random brightness
                brightness = np.random.uniform(0.8, 1.2)
                aug_img = np.clip(aug_img * brightness, 0, 1)
                
                augmented_X.append(aug_img)
                augmented_y.append(label)
        
        # Convert to arrays
        augmented_X = np.array(augmented_X).reshape(-1, 32, 32, 1)
        augmented_y = np.array(augmented_y)
        
        print(f"✅ Augmentation complete: {len(augmented_X)} total samples")
        return augmented_X, augmented_y
    
    def create_model(self):
        """Create CNN model for character recognition"""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        print("🏗️ Creating CNN model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(32, 32, 1)),
            
            # First convolution block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolution block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolution block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model created successfully")
        print(f"📊 Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, use_augmentation=True, epochs=50):
        """Train the model"""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available")
            return False
        
        if not self.load_dataset():
            return False
        
        # Apply data augmentation if requested
        if use_augmentation:
            X_train_aug, y_train_aug = self.augment_data(self.X_train, self.y_train)
        else:
            X_train_aug, y_train_aug = self.X_train, self.y_train
        
        # Create model
        self.model = self.create_model()
        if self.model is None:
            return False
        
        print("🚀 Starting training...")
        print(f"📚 Training on {len(X_train_aug)} samples for {epochs} epochs")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                f"{self.models_dir}/best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train_aug, y_train_aug,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.models_dir}/roblox_model_{timestamp}.h5"
        self.model.save(model_path)
        
        print(f"✅ Training complete! Model saved to {model_path}")
        
        # Plot training history
        self.plot_training_history(history)
        
        return True
    
    def plot_training_history(self, history):
        """Plot training history"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Accuracy plot
            ax1.plot(history.history['accuracy'], label='Training')
            ax1.plot(history.history['val_accuracy'], label='Validation')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True)
            
            # Loss plot
            ax2.plot(history.history['loss'], label='Training')
            ax2.plot(history.history['val_loss'], label='Validation')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = f"{self.models_dir}/training_history.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history saved to {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not plot training history: {e}")
    
    def load_model(self, model_path=None):
        """Load a trained model"""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available")
            return False
        
        if model_path is None:
            # Find latest model
            if not os.path.exists(self.models_dir):
                print(f"❌ Models directory not found: {self.models_dir}")
                return False
            
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.h5')]
            if not model_files:
                print("❌ No trained models found")
                return False
            
            # Use best model if available, otherwise latest
            if 'best_model.h5' in model_files:
                model_path = f"{self.models_dir}/best_model.h5"
            else:
                model_files.sort()
                model_path = f"{self.models_dir}/{model_files[-1]}"
        
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✅ Loaded model: {os.path.basename(model_path)}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        if self.model is None:
            print("❌ No model loaded")
            return
        
        if not hasattr(self, 'X_test'):
            if not self.load_dataset():
                return
        
        print("🧪 Evaluating model...")
        
        # Predictions
        predictions = self.model.predict(self.X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        # Classification report
        print("\n📊 Classification Report:")
        print("=" * 50)
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.character_classes,
            digits=3
        )
        print(report)
        
        # Confusion matrix
        self.plot_confusion_matrix(self.y_test, y_pred)
        
        # Per-class accuracy
        print("\n🎯 Per-class Accuracy:")
        print("=" * 30)
        for i, char in enumerate(self.character_classes):
            mask = self.y_test == i
            if np.sum(mask) > 0:
                acc = np.mean(y_pred[mask] == self.y_test[mask])
                print(f"'{char}': {acc:.3f} ({np.sum(mask)} samples)")
        
        # Overall accuracy
        overall_acc = np.mean(y_pred == self.y_test)
        print(f"\n🏆 Overall Accuracy: {overall_acc:.3f}")
        
        return overall_acc
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.character_classes,
                       yticklabels=self.character_classes)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            # Save plot
            plot_path = f"{self.models_dir}/confusion_matrix.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not plot confusion matrix: {e}")
    
    def detect_characters_in_image(self, img):
        """Detect and segment characters in an image"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection methods (same as labeler)
        methods = [
            ("Normal OTSU", cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("Inverted OTSU", cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
            ("Adaptive", cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 11, 2))
        ]
        
        best_regions = []
        
        for method_name, binary_img in methods:
            regions = self.find_regions_in_binary(binary_img)
            if len(regions) > len(best_regions):
                best_regions = regions
        
        return best_regions
    
    def find_regions_in_binary(self, binary_img):
        """Find character regions in binary image"""
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Size filtering (same as labeler)
            if (3 <= w <= 40 and 8 <= h <= 45 and area >= 15 and 0.2 <= w/h <= 5.0):
                regions.append((x, y, w, h))
        
        # Sort by x position
        regions.sort(key=lambda r: r[0])
        return regions
    
    def predict_characters(self, img):
        """Predict characters in an image"""
        if self.model is None:
            print("❌ No model loaded")
            return []
        
        # Detect character regions
        regions = self.detect_characters_in_image(img)
        if not regions:
            print("⚠️ No characters detected")
            return []
        
        predictions = []
        
        for i, (x, y, w, h) in enumerate(regions):
            # Extract character region
            char_img = img[y:y+h, x:x+w]
            
            # Resize to 32x32
            char_img = cv2.resize(char_img, (32, 32))
            
            # Normalize
            char_img = char_img.astype(np.float32) / 255.0
            
            # Reshape for model
            char_img = char_img.reshape(1, 32, 32, 1)
            
            # Predict
            pred = self.model.predict(char_img, verbose=0)
            class_idx = np.argmax(pred)
            confidence = pred[0][class_idx]
            
            predicted_char = self.character_classes[class_idx]
            
            predictions.append({
                'char': predicted_char,
                'confidence': confidence,
                'bbox': (x, y, w, h),
                'region_id': i
            })
        
        return predictions
    
    def test_screenshot(self):
        """Take a screenshot and test the model"""
        if self.model is None:
            if not self.load_model():
                return
        
        try:
            print("📸 Taking screenshot...")
            
            # Take screenshot
            screenshot = pyautogui.screenshot(region=(
                self.scan_region['x'], self.scan_region['y'],
                self.scan_region['width'], self.scan_region['height']
            ))
            
            # Convert to numpy array
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Predict characters
            predictions = self.predict_characters(img)
            
            if predictions:
                print(f"🎯 Detected {len(predictions)} characters:")
                result_text = ""
                for pred in predictions:
                    print(f"  '{pred['char']}' (confidence: {pred['confidence']:.3f})")
                    result_text += pred['char']
                
                print(f"📝 Full text: {result_text}")
                
                # Visualize results
                self.visualize_predictions(img, predictions)
                
                return result_text
            else:
                print("❌ No characters detected")
                return ""
                
        except Exception as e:
            print(f"❌ Screenshot test failed: {e}")
            return ""
    
    def visualize_predictions(self, img, predictions):
        """Visualize predictions on image"""
        try:
            # Create visualization
            vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Scale up for visibility
            scale = 8
            h, w = vis_img.shape[:2]
            vis_img = cv2.resize(vis_img, (w * scale, h * scale), 
                               interpolation=cv2.INTER_NEAREST)
            
            # Draw predictions
            for pred in predictions:
                x, y, w, h = pred['bbox']
                x, y, w, h = x * scale, y * scale, w * scale, h * scale
                
                # Color based on confidence
                confidence = pred['confidence']
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green - high confidence
                elif confidence > 0.6:
                    color = (255, 255, 0)  # Yellow - medium confidence
                else:
                    color = (255, 0, 0)  # Red - low confidence
                
                # Draw rectangle
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
                
                # Draw prediction text
                text = f"{pred['char']} ({pred['confidence']:.2f})"
                cv2.putText(vis_img, text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show image
            plt.figure(figsize=(12, 6))
            plt.imshow(vis_img)
            plt.title("Character Predictions")
            plt.axis('off')
            
            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_path = f"{self.models_dir}/prediction_{timestamp}.png"
            plt.savefig(vis_path, dpi=300, bbox_inches='tight')
            print(f"🖼️ Visualization saved to {vis_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not visualize predictions: {e}")
    
    def test_existing_screenshots(self, num_tests=5):
        """Test model on existing screenshots"""
        if self.model is None:
            if not self.load_model():
                return
        
        if not os.path.exists(self.screenshots_dir):
            print("❌ No screenshots directory found")
            return
        
        screenshot_files = [f for f in os.listdir(self.screenshots_dir) if f.endswith('.png')]
        
        if not screenshot_files:
            print("❌ No screenshots found")
            return
        
        # Test random screenshots
        import random
        test_files = random.sample(screenshot_files, min(num_tests, len(screenshot_files)))
        
        print(f"🧪 Testing {len(test_files)} random screenshots...")
        
        for i, filename in enumerate(test_files):
            print(f"\n📸 Testing {i+1}/{len(test_files)}: {filename}")
            
            # Load screenshot
            img_path = os.path.join(self.screenshots_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"❌ Could not load {filename}")
                continue
            
            # Predict
            predictions = self.predict_characters(img)
            
            if predictions:
                result_text = "".join([pred['char'] for pred in predictions])
                print(f"📝 Predicted: {result_text}")
                
                # Show confidence scores
                for pred in predictions:
                    print(f"  '{pred['char']}': {pred['confidence']:.3f}")
            else:
                print("❌ No characters detected")

def main():
    """Main function with menu interface"""
    if not TF_AVAILABLE:
        print("❌ TensorFlow is required for training and testing")
        print("Install with: pip install tensorflow")
        return
    
    trainer = RobloxCharacterTrainer()
    
    while True:
        print("\n" + "="*50)
        print("🤖 ROBLOX CHARACTER RECOGNITION - TRAINER & TESTER")
        print("="*50)
        print("1. 🚀 Train new model")
        print("2. 📊 Evaluate existing model")
        print("3. 📸 Test live screenshot")
        print("4. 🧪 Test existing screenshots")
        print("5. 📂 Load specific model")
        print("6. ❌ Exit")
        print("="*50)
        
        try:
            choice = input("Select option (1-6): ").strip()
            
            if choice == '1':
                print("\n🚀 Training new model...")
                use_aug = input("Use data augmentation? (y/n): ").lower() == 'y'
                epochs = int(input("Number of epochs (default 50): ") or "50")
                trainer.train_model(use_augmentation=use_aug, epochs=epochs)
                
            elif choice == '2':
                print("\n📊 Evaluating model...")
                if trainer.model is None:
                    trainer.load_model()
                trainer.evaluate_model()
                
            elif choice == '3':
                print("\n📸 Testing live screenshot...")
                print("Make sure Roblox math problem is visible!")
                input("Press Enter when ready...")
                trainer.test_screenshot()
                
            elif choice == '4':
                print("\n🧪 Testing existing screenshots...")
                num_tests = int(input("Number of screenshots to test (default 5): ") or "5")
                trainer.test_existing_screenshots(num_tests)
                
            elif choice == '5':
                print("\n📂 Loading specific model...")
                model_path = input("Enter model path (or press Enter for auto): ").strip()
                if not model_path:
                    model_path = None
                trainer.load_model(model_path)
                
            elif choice == '6':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()