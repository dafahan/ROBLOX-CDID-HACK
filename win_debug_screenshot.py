"""
WINDOWS SCREENSHOT DEBUG TOOL

This script helps debug your bot's screenshot and OCR functionality on Windows.
It captures screenshots, processes them, and shows you what the OCR sees.

USAGE:
python windows_debug.py

FEATURES:
- Automatic Tesseract path detection for Windows
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

# Windows-specific Tesseract setup
if sys.platform.startswith('win'):
    # Common Tesseract installation paths on Windows
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
        r'C:\tesseract\tesseract.exe'
    ]
    
    # Find and set Tesseract path
    tesseract_found = False
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_found = True
            print(f"✓ Found Tesseract at: {path}")
            break
    
    if not tesseract_found:
        print("❌ Tesseract not found. Please install from:")
        print("https://github.com/UB-Mannheim/tesseract/wiki")
        print("Common installation paths:")
        for path in tesseract_paths:
            print(f"  {path}")
        input("Press Enter to continue anyway (OCR will fail)...")

import pyautogui

# =============================================================================
# CONFIGURATION - UPDATE WITH YOUR COORDINATES
# =============================================================================

# Update these coordinates with your actual values
SCAN_REGION = {
    'x': 792,       # X coordinate of top-left corner
    'y': 484,       # Y coordinate of top-left corner
    'width': 133,   # Width of the region
    'height': 50    # Height of the region
}

# Click coordinates for testing
INPUT_COORDS = (997, 758)    # Center of input field
SUBMIT_COORDS = (1134, 758)  # Center of submit button

# OCR Configurations to test
OCR_CONFIGS = [
    ('Default', '--psm 6 -c tessedit_char_whitelist=0123456789+-*/'),
    ('PSM 7', '--psm 7 -c tessedit_char_whitelist=0123456789+-*/'),
    ('PSM 8', '--psm 8 -c tessedit_char_whitelist=0123456789+-*/'),
    ('PSM 13', '--psm 13 -c tessedit_char_whitelist=0123456789+-*/'),
    ('No Whitelist', '--psm 6'),
    ('Digits Only', '--psm 6 -c tessedit_char_whitelist=0123456789'),
    ('Math Only', '--psm 6 -c tessedit_char_whitelist=0123456789+-'),
    ('Simple', '--psm 8 -c tessedit_char_whitelist=0123456789+-'),
]

# =============================================================================
# WINDOWS-SPECIFIC FUNCTIONS
# =============================================================================

def test_tesseract_installation():
    """Test if Tesseract is properly installed and accessible"""
    print("🔍 Testing Tesseract installation...")
    
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract version: {version}")
        
        # Test with simple image
        test_image = Image.new('RGB', (100, 30), color='white')
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(test_image)
        draw.text((10, 5), "123", fill='black')
        
        result = pytesseract.image_to_string(test_image, config='--psm 8 -c tessedit_char_whitelist=0123456789')
        print(f"✓ Test OCR result: '{result.strip()}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Install to default location: C:\\Program Files\\Tesseract-OCR\\")
        print("3. Restart this script")
        return False

def windows_coordinate_finder():
    """Windows coordinate finder with enhanced display"""
    print("\n" + "=" * 60)
    print("🎯 WINDOWS COORDINATE FINDER")
    print("=" * 60)
    print("Move your mouse over different elements:")
    print("1. Math problem text (for SCAN_REGION)")
    print("2. Input field center (for INPUT_COORDS)")
    print("3. Submit button center (for SUBMIT_COORDS)")
    print()
    print("Press Ctrl+C when done")
    print("Current mouse position:")
    
    try:
        while True:
            x, y = pyautogui.position()
            print(f"\rMouse: X={x:4d}, Y={y:4d} | Region: ({x}, {y}, width, height) | Point: ({x}, {y})", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n\n✅ Last position: X={x}, Y={y}")
        print("\nUpdate your script with these coordinates:")
        print(f"For SCAN_REGION: x={x}, y={y}")
        print(f"For click coordinates: ({x}, {y})")

def test_click_coordinates():
    """Test clicking at specified coordinates"""
    print("\n🎯 Testing click coordinates...")
    print("This will test clicking your input field and submit button")
    print("Make sure your browser/application is visible!")
    
    input("Press Enter to test INPUT_COORDS click...")
    try:
        pyautogui.click(INPUT_COORDS[0], INPUT_COORDS[1])
        print(f"✓ Clicked input field at {INPUT_COORDS}")
        time.sleep(1)
    except Exception as e:
        print(f"❌ Failed to click input field: {e}")
    
    input("Press Enter to test SUBMIT_COORDS click...")
    try:
        pyautogui.click(SUBMIT_COORDS[0], SUBMIT_COORDS[1])
        print(f"✓ Clicked submit button at {SUBMIT_COORDS}")
        time.sleep(1)
    except Exception as e:
        print(f"❌ Failed to click submit button: {e}")

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
        print(f"  Region: x={SCAN_REGION['x']}, y={SCAN_REGION['y']}")
        
        return screenshot
        
    except Exception as e:
        print(f"❌ Screenshot failed: {e}")
        print("Make sure the coordinates are within your screen bounds!")
        return None

def preprocess_image_variants(image):
    """Create different preprocessing variants"""
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    variants = {}
    
    # Original grayscale
    variants['1_grayscale'] = gray
    
    # Binary threshold (OTSU)
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['2_threshold_otsu'] = thresh_otsu
    
    # Binary threshold (fixed)
    _, thresh_fixed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    variants['3_threshold_fixed'] = thresh_fixed
    
    # Inverted binary (white text on black background)
    _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants['4_threshold_inverted'] = thresh_inv
    
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    variants['5_adaptive'] = adaptive
    
    # Morphological operations
    kernel = np.ones((2,2), np.uint8)
    morphed = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
    variants['6_morphological'] = morphed
    
    # Noise reduction
    denoised = cv2.medianBlur(gray, 3)
    _, denoised_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['7_denoised'] = denoised_thresh
    
    # High contrast
    contrast = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
    _, contrast_thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['8_high_contrast'] = contrast_thresh
    
    return variants

def test_ocr_configs(image_variants, debug_folder):
    """Test different OCR configurations on all image variants"""
    print("\n" + "=" * 80)
    print("🔍 TESTING OCR CONFIGURATIONS")
    print("=" * 80)
    
    results = {}
    best_results = []
    
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
                
                # Print result with color coding
                if cleaned_text:
                    # Check if it's a valid math expression
                    if any(op in cleaned_text for op in ['+', '-', '*', '/']):
                        print(f"  {config_name:15}: '{cleaned_text}' ✅ MATH DETECTED")
                        best_results.append((variant_name, config_name, cleaned_text))
                    else:
                        print(f"  {config_name:15}: '{cleaned_text}' ⚠️ TEXT ONLY")
                else:
                    print(f"  {config_name:15}: (empty)")
                    
            except Exception as e:
                results[variant_name][config_name] = f"ERROR: {e}"
                print(f"  {config_name:15}: ERROR - {e}")
    
    # Show best results summary
    if best_results:
        print(f"\n🎯 BEST RESULTS (Math expressions detected):")
        print("=" * 60)
        for variant, config, text in best_results:
            print(f"  {variant} + {config}: '{text}'")
    else:
        print(f"\n❌ NO MATH EXPRESSIONS DETECTED")
        print("Try adjusting your SCAN_REGION coordinates or check image quality")
    
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
            label = name.replace('_', ' ').upper()
            cv2.putText(comparison_labeled, label, (x_start + 5, y_start + 20), 
                       font, 0.3, (0, 255, 255), 1)
        
        idx += 1
    
    # Save comparison
    comparison_path = os.path.join(debug_folder, "comparison.png")
    cv2.imwrite(comparison_path, comparison_labeled)
    print(f"✓ Comparison saved: {comparison_path}")

def save_results_report(results, debug_folder):
    """Save detailed results to text file"""
    report_path = os.path.join(debug_folder, "ocr_results.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("WINDOWS OCR DEBUG RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Scan Region: {SCAN_REGION}\n")
        f.write(f"Input Coords: {INPUT_COORDS}\n")
        f.write(f"Submit Coords: {SUBMIT_COORDS}\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        
        # Tesseract info
        try:
            f.write(f"Tesseract Path: {pytesseract.pytesseract.tesseract_cmd}\n")
            f.write(f"Tesseract Version: {pytesseract.get_tesseract_version()}\n\n")
        except:
            f.write("Tesseract: Not accessible\n\n")
        
        for variant_name, configs in results.items():
            f.write(f"\n{variant_name.upper()}\n")
            f.write("-" * 30 + "\n")
            
            for config_name, result in configs.items():
                f.write(f"{config_name:20}: '{result}'\n")
        
        # Summary of successful detections
        f.write("\n\nSUMMARY - SUCCESSFUL MATH DETECTIONS\n")
        f.write("=" * 40 + "\n")
        
        math_found = False
        for variant_name, configs in results.items():
            for config_name, result in configs.items():
                if result and any(op in result for op in ['+', '-', '*', '/']):
                    f.write(f"{variant_name} + {config_name}: '{result}'\n")
                    math_found = True
        
        if not math_found:
            f.write("No math expressions detected.\n")
            f.write("Recommendations:\n")
            f.write("1. Check SCAN_REGION coordinates\n")
            f.write("2. Ensure good contrast in captured area\n")
            f.write("3. Try different OCR PSM modes\n")
    
    print(f"✓ Report saved: {report_path}")

def quick_ocr_test():
    """Quick OCR test with current settings"""
    print("\n🔍 QUICK OCR TEST")
    print("=" * 40)
    
    screenshot = capture_and_analyze()
    if screenshot:
        # Save original
        screenshot.save("quick_test_original.png")
        
        # Process image
        gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save processed
        cv2.imwrite("quick_test_processed.png", thresh)
        
        # Test OCR
        try:
            text = pytesseract.image_to_string(Image.fromarray(thresh), 
                                             config='--psm 6 -c tessedit_char_whitelist=0123456789+-*/')
            text = text.strip().replace(' ', '').replace('\n', '')
            
            print(f"🔍 OCR Result: '{text}'")
            
            if text and any(op in text for op in ['+', '-', '*', '/']):
                print("✅ Math expression detected!")
                
                # Try to calculate
                try:
                    import re
                    if re.match(r'^[0-9+\-*/\s]+$', text):
                        result = eval(text)
                        print(f"🧮 Calculation: {text} = {result}")
                    else:
                        print("⚠️ Contains invalid characters for calculation")
                except Exception as e:
                    print(f"❌ Calculation failed: {e}")
            else:
                print("❌ No math expression found")
                
        except Exception as e:
            print(f"❌ OCR failed: {e}")
        
        print("💾 Files saved:")
        print("  - quick_test_original.png")
        print("  - quick_test_processed.png")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main debug function for Windows"""
    print("🪟 WINDOWS SCREENSHOT DEBUG TOOL")
    print("=" * 80)
    print(f"Current scan region: {SCAN_REGION}")
    print(f"Input coordinates: {INPUT_COORDS}")
    print(f"Submit coordinates: {SUBMIT_COORDS}")
    print("\nOptions:")
    print("1. Full analysis (capture & test all OCR configs)")
    print("2. Quick OCR test")
    print("3. Find coordinates")
    print("4. Test click coordinates")
    print("5. Test Tesseract installation")
    print("6. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
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
                    print("Open 'comparison.png' to see all image variants")
                    print("Check 'ocr_results.txt' for detailed results")
                
            elif choice == '2':
                quick_ocr_test()
                
            elif choice == '3':
                windows_coordinate_finder()
                    
            elif choice == '4':
                test_click_coordinates()
                
            elif choice == '5':
                test_tesseract_installation()
                
            elif choice == '6':
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
    print("🔍 Checking dependencies...")
    missing_deps = []
    
    try:
        import pyautogui
        print("✓ PyAutoGUI available")
    except ImportError:
        missing_deps.append("pyautogui")
    
    try:
        import cv2
        print("✓ OpenCV available")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import pytesseract
        print("✓ Pytesseract available")
    except ImportError:
        missing_deps.append("pytesseract")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        exit(1)
    
    print("✓ All Python dependencies available")
    
    # Check Tesseract
    if not test_tesseract_installation():
        print("\n⚠️ Tesseract has issues, but you can still test coordinates")
    
    main()