"""
SCREENSHOT DEBUG TOOL

This script helps debug your bot's screenshot and OCR functionality.
It captures screenshots, processes them, and shows you what the OCR sees.

USAGE:
python debug_screenshot.py

FEATURES:
- Captures screenshots from your defined region
- Shows original and processed images
- Tests OCR with different configurations
- Saves debug images for analysis
- Real-time coordinate display
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from datetime import datetime

# Fix X11 display issues on Linux
if sys.platform.startswith('linux'):
    os.environ['DISPLAY'] = ':0'
    try:
        os.system('xhost +local: 2>/dev/null')
    except:
        pass

import pyautogui

# =============================================================================
# CONFIGURATION - UPDATE WITH YOUR COORDINATES
# =============================================================================

SCAN_REGION = {
    'x': 792,       # X coordinate of top-left corner
    'y': 484,       # Y coordinate of top-left corner
    'width': 133,   # Width of the region
    'height': 50    # Height of the region
}

# OCR Configurations to test
OCR_CONFIGS = [
    ('Default', '--psm 6 -c tessedit_char_whitelist=0123456789+-*/'),
    ('PSM 7', '--psm 7 -c tessedit_char_whitelist=0123456789+-*/'),
    ('PSM 8', '--psm 8 -c tessedit_char_whitelist=0123456789+-*/'),
    ('PSM 13', '--psm 13 -c tessedit_char_whitelist=0123456789+-*/'),
    ('No Whitelist', '--psm 6'),
    ('Digits Only', '--psm 6 -c tessedit_char_whitelist=0123456789'),
]

# =============================================================================
# DEBUG FUNCTIONS
# =============================================================================

def create_debug_folder():
    """Create folder for debug images"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"debug_screenshots_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def capture_and_analyze():
    """Capture screenshot and analyze with different methods"""
    print("=" * 80)
    print("📸 CAPTURING SCREENSHOT")
    print("=" * 80)
    
    try:
        # Capture screenshot
        screenshot = pyautogui.screenshot(region=(
            SCAN_REGION['x'], 
            SCAN_REGION['y'], 
            SCAN_REGION['width'], 
            SCAN_REGION['height']
        ))
        
        print(f"✓ Screenshot captured: {SCAN_REGION['width']}x{SCAN_REGION['height']} pixels")
        
        return screenshot
        
    except Exception as e:
        print(f"❌ Screenshot failed: {e}")
        return None

def preprocess_image_variants(image):
    """Create different preprocessing variants"""
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    variants = {}
    
    # Original grayscale
    variants['grayscale'] = gray
    
    # Binary threshold (OTSU)
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['threshold_otsu'] = thresh_otsu
    
    # Binary threshold (fixed)
    _, thresh_fixed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    variants['threshold_fixed'] = thresh_fixed
    
    # Inverted binary
    _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants['threshold_inverted'] = thresh_inv
    
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    variants['adaptive'] = adaptive
    
    # Morphological operations
    kernel = np.ones((2,2), np.uint8)
    morphed = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
    variants['morphological'] = morphed
    
    # Noise reduction
    denoised = cv2.medianBlur(gray, 3)
    _, denoised_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['denoised'] = denoised_thresh
    
    return variants

def test_ocr_configs(image_variants, debug_folder):
    """Test different OCR configurations on all image variants"""
    print("\n" + "=" * 80)
    print("🔍 TESTING OCR CONFIGURATIONS")
    print("=" * 80)
    
    results = {}
    
    for variant_name, variant_image in image_variants.items():
        print(f"\n📊 Testing variant: {variant_name}")
        print("-" * 50)
        
        results[variant_name] = {}
        
        # Save variant image
        variant_path = os.path.join(debug_folder, f"variant_{variant_name}.png")
        cv2.imwrite(variant_path, variant_image)
        
        for config_name, config in OCR_CONFIGS:
            try:
                # Convert to PIL Image for pytesseract
                pil_image = Image.fromarray(variant_image)
                
                # Run OCR
                text = pytesseract.image_to_string(pil_image, config=config)
                cleaned_text = text.strip().replace(' ', '').replace('\n', '')
                
                results[variant_name][config_name] = cleaned_text
                
                # Print result
                if cleaned_text:
                    print(f"  {config_name:15}: '{cleaned_text}'")
                else:
                    print(f"  {config_name:15}: (empty)")
                    
            except Exception as e:
                results[variant_name][config_name] = f"ERROR: {e}"
                print(f"  {config_name:15}: ERROR - {e}")
    
    return results

def create_comparison_image(original, variants, debug_folder):
    """Create a side-by-side comparison of all variants"""
    print("\n📋 Creating comparison image...")
    
    # Calculate grid size
    num_variants = len(variants) + 1  # +1 for original
    cols = 3
    rows = (num_variants + cols - 1) // cols
    
    # Get image dimensions
    h, w = list(variants.values())[0].shape
    
    # Create comparison image
    comparison = np.zeros((rows * h, cols * w), dtype=np.uint8)
    
    # Add original image (converted to grayscale)
    original_gray = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
    comparison[0:h, 0:w] = original_gray
    
    # Add labels
    comparison_labeled = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Label original
    cv2.putText(comparison_labeled, 'ORIGINAL', (5, 20), font, 0.5, (0, 255, 0), 1)
    
    # Add variants
    idx = 1
    for name, variant in variants.items():
        row = idx // cols
        col = idx % cols
        
        y_start = row * h
        y_end = y_start + h
        x_start = col * w
        x_end = x_start + w
        
        if y_end <= comparison.shape[0] and x_end <= comparison.shape[1]:
            comparison[y_start:y_end, x_start:x_end] = variant
            
            # Add label
            cv2.putText(comparison_labeled, name.upper(), (x_start + 5, y_start + 20), 
                       font, 0.4, (0, 255, 255), 1)
        
        idx += 1
    
    # Save comparison
    comparison_path = os.path.join(debug_folder, "comparison.png")
    cv2.imwrite(comparison_path, comparison_labeled)
    print(f"✓ Comparison saved: {comparison_path}")

def save_results_report(results, debug_folder):
    """Save detailed results to text file"""
    report_path = os.path.join(debug_folder, "ocr_results.txt")
    
    with open(report_path, 'w') as f:
        f.write("OCR DEBUG RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Scan Region: {SCAN_REGION}\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        
        for variant_name, configs in results.items():
            f.write(f"\n{variant_name.upper()}\n")
            f.write("-" * 30 + "\n")
            
            for config_name, result in configs.items():
                f.write(f"{config_name:20}: '{result}'\n")
        
        # Summary of successful detections
        f.write("\n\nSUMMARY - SUCCESSFUL MATH DETECTIONS\n")
        f.write("=" * 40 + "\n")
        
        for variant_name, configs in results.items():
            for config_name, result in configs.items():
                if result and any(op in result for op in ['+', '-', '*', '/']):
                    f.write(f"{variant_name} + {config_name}: '{result}'\n")
    
    print(f"✓ Report saved: {report_path}")

def live_screenshot_mode():
    """Continuous screenshot mode for real-time debugging"""
    print("\n" + "=" * 80)
    print("🎥 LIVE SCREENSHOT MODE")
    print("=" * 80)
    print("Press 'q' to quit, 's' to save current screenshot, 'c' to capture & analyze")
    print("Screenshots updating every 2 seconds...")
    
    try:
        while True:
            # Capture screenshot
            screenshot = pyautogui.screenshot(region=(
                SCAN_REGION['x'], 
                SCAN_REGION['y'], 
                SCAN_REGION['width'], 
                SCAN_REGION['height']
            ))
            
            # Quick OCR test
            gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            try:
                text = pytesseract.image_to_string(Image.fromarray(thresh), 
                                                 config='--psm 6 -c tessedit_char_whitelist=0123456789+-*/')
                text = text.strip().replace(' ', '').replace('\n', '')
            except:
                text = "(OCR failed)"
            
            # Display current status
            print(f"\r📸 Live: '{text:20}' | Region: {SCAN_REGION['x']},{SCAN_REGION['y']} | Size: {SCAN_REGION['width']}x{SCAN_REGION['height']}", end="", flush=True)
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n✓ Live mode stopped")

def interactive_region_adjuster():
    """Interactive mode to adjust scan region"""
    print("\n" + "=" * 80)
    print("⚙️ INTERACTIVE REGION ADJUSTER")
    print("=" * 80)
    print("Commands:")
    print("  w/s: move up/down    a/d: move left/right")
    print("  +/-: resize width    8/2: resize height")
    print("  t: test current region    q: quit")
    
    current_region = SCAN_REGION.copy()
    
    while True:
        print(f"\nCurrent region: x={current_region['x']}, y={current_region['y']}, "
              f"w={current_region['width']}, h={current_region['height']}")
        
        try:
            cmd = input("Command: ").lower().strip()
            
            if cmd == 'q':
                break
            elif cmd == 'w':
                current_region['y'] -= 10
            elif cmd == 's':
                current_region['y'] += 10
            elif cmd == 'a':
                current_region['x'] -= 10
            elif cmd == 'd':
                current_region['x'] += 10
            elif cmd == '+':
                current_region['width'] += 10
            elif cmd == '-':
                current_region['width'] = max(10, current_region['width'] - 10)
            elif cmd == '8':
                current_region['height'] += 5
            elif cmd == '2':
                current_region['height'] = max(5, current_region['height'] - 5)
            elif cmd == 't':
                # Test current region
                test_screenshot = pyautogui.screenshot(region=(
                    current_region['x'], current_region['y'], 
                    current_region['width'], current_region['height']
                ))
                test_screenshot.save(f"test_region_{int(time.time())}.png")
                print("✓ Test screenshot saved")
            else:
                print("Invalid command")
                
        except KeyboardInterrupt:
            break
    
    print(f"\nFinal region: {current_region}")
    return current_region

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main debug function"""
    print("🔧 SCREENSHOT DEBUG TOOL")
    print("=" * 80)
    print(f"Current scan region: {SCAN_REGION}")
    print("\nOptions:")
    print("1. Capture & analyze once")
    print("2. Live screenshot mode")
    print("3. Interactive region adjuster")
    print("4. Quick OCR test")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                # Full analysis
                debug_folder = create_debug_folder()
                print(f"📁 Debug folder: {debug_folder}")
                
                screenshot = capture_and_analyze()
                if screenshot:
                    screenshot.save(os.path.join(debug_folder, "original.png"))
                    
                    variants = preprocess_image_variants(screenshot)
                    results = test_ocr_configs(variants, debug_folder)
                    create_comparison_image(screenshot, variants, debug_folder)
                    save_results_report(results, debug_folder)
                    
                    print(f"\n✅ Analysis complete! Check folder: {debug_folder}")
                
            elif choice == '2':
                live_screenshot_mode()
                
            elif choice == '3':
                new_region = interactive_region_adjuster()
                if input("Use this region? (y/n): ").lower() == 'y':
                    SCAN_REGION.update(new_region)
                    print(f"✓ Updated scan region: {SCAN_REGION}")
                    
            elif choice == '4':
                # Quick test
                screenshot = capture_and_analyze()
                if screenshot:
                    gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    text = pytesseract.image_to_string(Image.fromarray(thresh), 
                                                     config='--psm 6 -c tessedit_char_whitelist=0123456789+-*/')
                    text = text.strip().replace(' ', '').replace('\n', '')
                    
                    print(f"\n🔍 Quick OCR result: '{text}'")
                    
                    # Save quick test
                    screenshot.save("quick_test.png")
                    cv2.imwrite("quick_test_processed.png", thresh)
                    print("💾 Saved: quick_test.png and quick_test_processed.png")
                
            elif choice == '5':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Check dependencies
    try:
        import pyautogui, cv2, pytesseract
        print("✓ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        exit(1)
    
    # Check Tesseract
    try:
        pytesseract.get_tesseract_version()
        print("✓ Tesseract OCR available")
    except Exception as e:
        print(f"❌ Tesseract issue: {e}")
    
    main()