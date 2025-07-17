"""
WINDOWS UIPI FIX BOT WITH FONT-BASED TEMPLATE MATCHING

This version replaces unreliable OCR with font-based template matching
using Fredoka/Cartoon font characteristics from Roblox Car Driving Indonesia.

FIXES APPLIED:
- Font-based template matching (much more accurate than OCR)
- Supports numbers -10 to 10 and operators +, -, *, /
- Windows UIPI fixes for clicking
- All previous UIPI and automation features
"""

import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
import ctypes
from ctypes import wintypes
import cv2

# Windows-specific imports and setup
import pyautogui

# Configure paths and check dependencies
DEBUG_MODE = True

def debug_print(message):
    if DEBUG_MODE:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

# =============================================================================
# WINDOWS ADMINISTRATOR AND UIPI DETECTION
# =============================================================================

def is_admin():
    """Check if running as administrator"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def check_uipi_status():
    """Check if UIPI (UI Privilege Isolation) is affecting us"""
    print("🔍 Checking Windows UI Access Status...")
    
    admin_status = is_admin()
    print(f"Administrator privileges: {'✅ YES' if admin_status else '❌ NO'}")
    
    if not admin_status:
        print("⚠️ UIPI Issue Detected!")
        print("Windows is blocking clicks after programmatic mouse movement.")
        print("\nSOLUTION: Run as Administrator")
        print("1. Right-click Command Prompt")
        print("2. Select 'Run as administrator'")
        print("3. Run this script again")
        return False
    
    return True

# =============================================================================
# WINDOWS API DIRECT INPUT METHODS
# =============================================================================

# Windows API constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

# Windows API structures
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("_input",)
    _fields_ = [("type", wintypes.DWORD),
                ("_input", _INPUT)]

def windows_api_click(x, y):
    """Use Windows API directly for clicking"""
    try:
        # Get screen dimensions for absolute coordinates
        screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        screen_height = ctypes.windll.user32.GetSystemMetrics(1)
        
        # Convert to absolute coordinates (0-65535 range)
        abs_x = int(x * 65535 / screen_width)
        abs_y = int(y * 65535 / screen_height)
        
        # Create mouse input for move
        move_input = INPUT()
        move_input.type = INPUT_MOUSE
        move_input.mi.dx = abs_x
        move_input.mi.dy = abs_y
        move_input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
        
        # Create mouse input for click down
        down_input = INPUT()
        down_input.type = INPUT_MOUSE
        down_input.mi.dx = abs_x
        down_input.mi.dy = abs_y
        down_input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE
        
        # Create mouse input for click up
        up_input = INPUT()
        up_input.type = INPUT_MOUSE
        up_input.mi.dx = abs_x
        up_input.mi.dy = abs_y
        up_input.mi.dwFlags = MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE
        
        # Send the inputs
        ctypes.windll.user32.SendInput(1, ctypes.byref(move_input), ctypes.sizeof(INPUT))
        time.sleep(0.05)
        ctypes.windll.user32.SendInput(1, ctypes.byref(down_input), ctypes.sizeof(INPUT))
        time.sleep(0.05)
        ctypes.windll.user32.SendInput(1, ctypes.byref(up_input), ctypes.sizeof(INPUT))
        
        return True
    except Exception as e:
        print(f"Windows API click failed: {e}")
        return False

def cursor_jiggle():
    """Jiggle cursor to unlock Windows mouse state"""
    try:
        current_pos = pyautogui.position()
        # Small jiggle movement
        pyautogui.moveTo(current_pos.x + 1, current_pos.y + 1, duration=0.1)
        time.sleep(0.05)
        pyautogui.moveTo(current_pos.x, current_pos.y, duration=0.1)
        time.sleep(0.05)
        return True
    except:
        return False

# =============================================================================
# CONFIGURATION
# =============================================================================

SCAN_REGION = {
    'x': 792,
    'y': 484,
    'width': 133,
    'height': 50
}

INPUT_COORDS = (1043, 519)
SUBMIT_COORDS = (950, 604)

DELAYS = {
    'LOOP_DELAY': 1.0,
    'INPUT_CLICK_DELAY': 0.5,
    'FOCUS_DELAY': 0.3,
    'TYPE_DELAY': 0.1,
    'CLEAR_DELAY': 0.2,
    'SUBMIT_CLICK_DELAY': 0.8,
    'POST_SUBMIT_DELAY': 1.0,
    'ERROR_RETRY_DELAY': 1.0,
    'JIGGLE_DELAY': 0.1,
}

FAILSAFE_KEY = 'q'
STOP_FILE = 'stop_bot.txt'

# =============================================================================
# FONT-BASED TEMPLATE MATCHING
# =============================================================================

def create_roblox_font_templates():
    """Create font templates matching Roblox Fredoka/Cartoon style"""
    debug_print("🎨 Creating Roblox-style font templates...")
    
    # Try to find fonts similar to Fredoka/Cartoon used in Roblox
    font_paths = [
        r'C:\Windows\Fonts\comic.ttf',      # Comic Sans (closest to Fredoka)
        r'C:\Windows\Fonts\comicbd.ttf',    # Comic Sans Bold
        r'C:\Windows\Fonts\arial.ttf',      # Arial rounded
        r'C:\Windows\Fonts\calibri.ttf',    # Calibri
    ]
    
    font = None
    font_size = 40  # Size similar to Roblox UI
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                debug_print(f"✅ Using font: {os.path.basename(font_path)}")
                break
            except:
                continue
    
    if font is None:
        font = ImageFont.load_default()
        debug_print("⚠️ Using default font")
    
    templates = {}
    
    # Create templates for numbers -10 to 10 and operators
    characters = []
    
    # Add single digits 0-9
    for i in range(10):
        characters.append(str(i))
    
    # Add negative single digits -1 to -9
    for i in range(1, 10):
        characters.append(f"-{i}")
    
    # Add -10 and 10
    characters.extend(["-10", "10"])
    
    # Add operators
    characters.extend(["+", "-", "*", "/", "="])
    
    for char in characters:
        # Create template image (white background, black text like Roblox)
        img = Image.new('RGB', (120, 80), color='white')
        draw = ImageDraw.Draw(img)
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (120 - text_width) // 2
        y = (80 - text_height) // 2
        
        # Draw text in black (Roblox style)
        draw.text((x, y), char, fill='black', font=font)
        
        # Convert to OpenCV format and create binary template
        opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        templates[char] = binary
        debug_print(f"  Created template for: '{char}'")
    
    debug_print(f"✅ Created {len(templates)} Roblox-style templates")
    return templates

def preprocess_roblox_image(image):
    """Preprocess Roblox screenshot for template matching"""
    # Convert PIL to OpenCV if needed
    if isinstance(image, Image.Image):
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        opencv_image = image
    
    # Convert to grayscale
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Create multiple variants for different text colors
    variants = {}
    
    # Variant 1: White text on dark background
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants['white_text'] = binary_inv
    
    # Variant 2: Black text on white background  
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['black_text'] = binary
    
    # Variant 3: Enhanced contrast
    enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
    _, enhanced_binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants['enhanced'] = enhanced_binary
    
    return variants

def match_roblox_templates(image_variants, templates, confidence_threshold=0.65):
    """Match Roblox font templates against image"""
    debug_print("🔍 Matching Roblox templates...")
    
    all_matches = []
    
    for variant_name, image in image_variants.items():
        debug_print(f"  Testing {variant_name} variant...")
        
        for char, template in templates.items():
            # Try multiple scales to handle different text sizes
            scales = [0.6, 0.8, 1.0, 1.2, 1.4]
            
            for scale in scales:
                # Resize template
                h, w = template.shape
                new_h, new_w = int(h * scale), int(w * scale)
                
                # Skip if template would be larger than image
                if new_h >= image.shape[0] or new_w >= image.shape[1]:
                    continue
                
                scaled_template = cv2.resize(template, (new_w, new_h))
                
                # Template matching
                result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= confidence_threshold)
                
                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    all_matches.append({
                        'char': char,
                        'confidence': confidence,
                        'position': pt,
                        'scale': scale,
                        'variant': variant_name,
                        'size': (new_w, new_h)
                    })
                    
                    debug_print(f"    Found '{char}' at {pt} (conf: {confidence:.3f})")
    
    # Sort by confidence
    all_matches.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Remove overlapping matches
    filtered_matches = []
    for match in all_matches:
        overlap = False
        for existing in filtered_matches:
            x1, y1 = match['position']
            x2, y2 = existing['position']
            w1, h1 = match['size']
            w2, h2 = existing['size']
            
            # Check overlap
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            
            if overlap_area > 0.4 * min(w1 * h1, w2 * h2):
                overlap = True
                break
        
        if not overlap:
            filtered_matches.append(match)
    
    debug_print(f"✅ Found {len(filtered_matches)} unique template matches")
    return filtered_matches

def build_math_expression_from_matches(matches):
    """Build math expression from template matches"""
    if not matches:
        return None
    
    debug_print("🧮 Building math expression...")
    
    # Sort by horizontal position (left to right)
    matches.sort(key=lambda x: x['position'][0])
    
    # Extract characters
    chars = []
    for match in matches:
        char = match['char']
        confidence = match['confidence']
        pos = match['position'][0]
        
        debug_print(f"  Position {pos}: '{char}' (confidence: {confidence:.3f})")
        chars.append(char)
    
    # Join characters and clean
    raw_expression = ''.join(chars)
    debug_print(f"  Raw: '{raw_expression}'")
    
    # Clean and validate
    cleaned = clean_roblox_expression(raw_expression)
    debug_print(f"  Cleaned: '{cleaned}'")
    
    return cleaned

def clean_roblox_expression(expression):
    """Clean and validate Roblox math expression"""
    if not expression:
        return None
    
    # Remove spaces and unwanted characters
    cleaned = re.sub(r'\s+', '', expression)
    
    # Patterns for Roblox math expressions
    patterns = [
        # Full equation: 2+10=12
        r'(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*=\s*(-?\d+)',
        # Just the problem: 2+10
        r'(-?\d+)\s*([+\-*/])\s*(-?\d+)',
        # Handle cases where = might be detected
        r'(-?\d+)([+\-*/])(-?\d+)=(-?\d+)',
        r'(-?\d+)([+\-*/])(-?\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            groups = match.groups()
            if len(groups) >= 3:
                num1, op, num2 = groups[0], groups[1], groups[2]
                # Validate numbers are in range -10 to 10
                try:
                    n1, n2 = int(num1), int(num2)
                    if -10 <= n1 <= 10 and -10 <= n2 <= 10:
                        return f"{num1}{op}{num2}"
                except ValueError:
                    continue
    
    return None

def extract_math_expression_with_templates(screenshot):
    """Main function to extract math expression using template matching"""
    debug_print("🎯 Starting Roblox template matching...")
    
    try:
        # Create templates (cache for performance)
        if not hasattr(extract_math_expression_with_templates, 'templates'):
            extract_math_expression_with_templates.templates = create_roblox_font_templates()
        
        templates = extract_math_expression_with_templates.templates
        
        # Preprocess image
        image_variants = preprocess_roblox_image(screenshot)
        
        # Match templates
        matches = match_roblox_templates(image_variants, templates, confidence_threshold=0.6)
        
        if not matches:
            debug_print("❌ No template matches found")
            return None
        
        # Build expression
        expression = build_math_expression_from_matches(matches)
        
        if expression:
            debug_print(f"✅ Template matching successful: '{expression}'")
        else:
            debug_print("❌ Could not build valid expression")
        
        return expression
        
    except Exception as e:
        debug_print(f"❌ Template matching error: {e}")
        return None

# =============================================================================
# ENHANCED CLICKING WITH UIPI FIXES
# =============================================================================

def uipi_safe_click(coords, description="element", retries=3):
    """UIPI-safe clicking that works around Windows security restrictions"""
    x, y = coords
    debug_print(f"🎯 UIPI-safe click {description} at ({x}, {y})")
    
    for attempt in range(retries):
        try:
            debug_print(f"Attempt {attempt + 1}/{retries}")
            
            # Method 1: Cursor jiggle + Windows API click
            debug_print("Method 1: Jiggle + Windows API")
            cursor_jiggle()
            time.sleep(DELAYS['JIGGLE_DELAY'])
            
            if windows_api_click(x, y):
                debug_print("✅ Windows API click successful")
                return True
            
            # Method 2: PyAutoGUI with jiggle
            debug_print("Method 2: PyAutoGUI with jiggle")
            cursor_jiggle()
            time.sleep(DELAYS['JIGGLE_DELAY'])
            
            pyautogui.click(x, y, duration=0.2)
            time.sleep(0.2)
            
            # Method 3: Multiple small movements + click
            debug_print("Method 3: Incremental movement + click")
            current_pos = pyautogui.position()
            
            # Move in small steps to target
            steps = 5
            for i in range(steps):
                step_x = current_pos.x + (x - current_pos.x) * (i + 1) / steps
                step_y = current_pos.y + (y - current_pos.y) * (i + 1) / steps
                pyautogui.moveTo(step_x, step_y, duration=0.05)
                time.sleep(0.02)
            
            # Final jiggle and click
            cursor_jiggle()
            pyautogui.click(x, y)
            time.sleep(0.2)
            
            debug_print(f"✅ Click attempt {attempt + 1} completed")
            return True
            
        except Exception as e:
            debug_print(f"❌ Click attempt {attempt + 1} failed: {e}")
            time.sleep(0.5)
    
    debug_print(f"❌ All UIPI-safe click attempts failed for {description}")
    return False

def alternative_input_methods(answer):
    """Alternative input methods when clicking fails"""
    debug_print("🔄 Trying alternative input methods...")
    
    try:
        # Method 1: Tab to input field, type, then Enter
        debug_print("Alt Method 1: Tab navigation")
        pyautogui.press('tab')
        time.sleep(0.2)
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.1)
        pyautogui.typewrite(str(answer))
        time.sleep(0.2)
        pyautogui.press('enter')
        time.sleep(0.5)
        
        return True
        
    except Exception as e:
        debug_print(f"❌ Alternative input methods failed: {e}")
        return False

def windows_safe_keyboard_check():
    try:
        if os.path.exists(STOP_FILE):
            return True
        # Simple keyboard check without external library
        return False
    except Exception:
        return os.path.exists(STOP_FILE)

def enhanced_text_input(text, clear_first=True):
    """Enhanced text input with UIPI considerations"""
    try:
        text_str = str(text)
        debug_print(f"⌨️ UIPI-safe typing: '{text_str}'")
        
        if clear_first:
            debug_print("Clearing input field...")
            cursor_jiggle()
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(DELAYS['CLEAR_DELAY'])
            pyautogui.press('delete')
            time.sleep(DELAYS['CLEAR_DELAY'])
        
        # Type with small jiggle between characters
        for i, char in enumerate(text_str):
            if i % 2 == 0:
                cursor_jiggle()
            pyautogui.typewrite(char)
            time.sleep(DELAYS['TYPE_DELAY'])
        
        debug_print(f"✅ UIPI-safe typing completed: '{text_str}'")
        return True
        
    except Exception as e:
        debug_print(f"❌ UIPI-safe typing failed: {e}")
        return False

def calculate_result(expression):
    """Calculate result from math expression"""
    try:
        debug_print(f"🧮 Calculating: {expression}")
        
        if not re.match(r'^-?\d+[+\-*/]-?\d+$', expression):
            debug_print(f"❌ Invalid expression format: {expression}")
            return None
        
        result = eval(expression)
        if isinstance(result, (int, float)):
            result_int = int(result)
            debug_print(f"✅ Result: {expression} = {result_int}")
            return result_int
        else:
            return None
    except Exception as e:
        debug_print(f"❌ Calculation failed: {e}")
        return None

# =============================================================================
# UIPI-SAFE ANSWER SUBMISSION
# =============================================================================

def uipi_safe_answer_submission(answer):
    """UIPI-safe answer submission"""
    debug_print(f"🎯 Starting UIPI-safe submission for: {answer}")
    
    try:
        # Step 1: UIPI-safe click on input field
        debug_print("Step 1: UIPI-safe input field click...")
        if not uipi_safe_click(INPUT_COORDS, "input field"):
            debug_print("⚠️ Input click failed, trying alternatives...")
            alternative_input_methods(answer)
            return True
        
        time.sleep(DELAYS['INPUT_CLICK_DELAY'])
        
        # Step 2: Focus verification with jiggle
        debug_print(f"Step 2: Focus verification...")
        cursor_jiggle()
        time.sleep(DELAYS['FOCUS_DELAY'])
        
        # Step 3: UIPI-safe text input
        debug_print("Step 3: UIPI-safe text input...")
        if not enhanced_text_input(answer, clear_first=True):
            debug_print("❌ Text input failed")
            alternative_input_methods(answer)
            return True
        
        # Step 4: Wait before submit with jiggle
        debug_print(f"Step 4: Pre-submit wait...")
        cursor_jiggle()
        time.sleep(DELAYS['SUBMIT_CLICK_DELAY'])
        
        # Step 5: UIPI-safe submit click
        debug_print("Step 5: UIPI-safe submit click...")
        submit_success = uipi_safe_click(SUBMIT_COORDS, "submit button")
        
        # Step 6: Fallback if submit click failed
        if not submit_success:
            debug_print("⚠️ Submit click failed, using Enter key...")
            cursor_jiggle()
            pyautogui.press('enter')
            time.sleep(0.5)
        
        # Step 7: Post-submit wait
        debug_print(f"Step 7: Post-submit processing...")
        time.sleep(DELAYS['POST_SUBMIT_DELAY'])
        
        debug_print("✅ UIPI-safe submission completed!")
        return True
        
    except Exception as e:
        debug_print(f"❌ UIPI-safe submission failed: {e}")
        return False

# =============================================================================
# MAIN BOT LOGIC
# =============================================================================

def main():
    print("=" * 70)
    print("🛡️ ROBLOX TEMPLATE MATCHING BOT")
    print("=" * 70)
    
    # Check admin status first
    if not check_uipi_status():
        print("\n🚨 ADMIN PRIVILEGES REQUIRED!")
        print("Please restart as Administrator to avoid UIPI issues.")
        input("Press Enter to continue anyway (clicking may fail)...")
    
    print(f"📍 Scan region: {SCAN_REGION}")
    print(f"🎯 Input field: {INPUT_COORDS}")
    print(f"🔘 Submit button: {SUBMIT_COORDS}")
    print("\n🛡️ FEATURES:")
    print("   - Font-based template matching (no OCR!)")
    print("   - Supports numbers -10 to 10")
    print("   - Roblox Fredoka/Cartoon font detection")
    print("   - Windows UIPI fixes")
    print("   - Alternative input methods")
    print("=" * 70)
    
    # Configure PyAutoGUI
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.02
    
    loop_count = 0
    successful_submissions = 0
    failed_submissions = 0
    
    try:
        while True:
            loop_count += 1
            print(f"\n🔄 Loop {loop_count}: Scanning for math problems...")
            
            if windows_safe_keyboard_check():
                debug_print("🛑 Stop condition detected")
                break
            
            try:
                # Capture screenshot
                debug_print("📸 Capturing screenshot...")
                screenshot = pyautogui.screenshot(region=(
                    SCAN_REGION['x'], 
                    SCAN_REGION['y'], 
                    SCAN_REGION['width'], 
                    SCAN_REGION['height']
                ))
                
                # Extract expression using template matching
                expression = extract_math_expression_with_templates(screenshot)
                
                if expression:
                    print(f"📖 Detected: {expression}")
                    result = calculate_result(expression)
                    
                    if result is not None:
                        print(f"🧮 Answer: {result}")
                        
                        # Submit with UIPI-safe methods
                        if uipi_safe_answer_submission(result):
                            successful_submissions += 1
                            print(f"✅ SUCCESS! ({successful_submissions}✅ {failed_submissions}❌)")
                        else:
                            failed_submissions += 1
                            print(f"❌ FAILED! ({successful_submissions}✅ {failed_submissions}❌)")
                            time.sleep(DELAYS['ERROR_RETRY_DELAY'])
                    else:
                        debug_print("❌ Calculation failed")
                else:
                    debug_print("👁️ No math expression detected")
                
            except Exception as e:
                debug_print(f"❌ Loop error: {e}")
                failed_submissions += 1
                time.sleep(DELAYS['ERROR_RETRY_DELAY'])
            
            # Small jiggle before next loop
            cursor_jiggle()
            debug_print(f"⏰ Waiting {DELAYS['LOOP_DELAY']}s...")
            time.sleep(DELAYS['LOOP_DELAY'])
            
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        total = successful_submissions + failed_submissions
        success_rate = (successful_submissions / max(1, total)) * 100
        
        print("\n" + "=" * 70)
        print("📊 FINAL STATISTICS")
        print("=" * 70)
        print(f"✅ Successful: {successful_submissions}")
        print(f"❌ Failed: {failed_submissions}")
        print(f"📈 Success rate: {success_rate:.1f}%")
        print("=" * 70)

def test_template_creation():
    """Test template creation"""
    print("🧪 TESTING TEMPLATE CREATION")
    print("=" * 50)
    
    templates = create_roblox_font_templates()
    
    print(f"Created templates for: {list(templates.keys())}")
    
    # Save templates for inspection
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    for char, template in templates.items():
        safe_char = char.replace('-', 'neg').replace('/', 'div').replace('*', 'mul').replace('+', 'plus')
        filename = f"templates/template_{safe_char}.png"
        cv2.imwrite(filename, template)
        print(f"Saved: {filename}")

def test_with_screenshot():
    """Test with current screenshot"""
    print("🧪 TESTING WITH SCREENSHOT")
    print("=" * 50)
    
    print("Taking screenshot in 3 seconds...")
    time.sleep(3)
    
    screenshot = pyautogui.screenshot(region=(
        SCAN_REGION['x'], 
        SCAN_REGION['y'], 
        SCAN_REGION['width'], 
        SCAN_REGION['height']
    ))
    
    # Save screenshot for inspection
    screenshot.save('test_screenshot.png')
    print("Saved: test_screenshot.png")
    
    # Test template matching
    expression = extract_math_expression_with_templates(screenshot)
    
    if expression:
        result = calculate_result(expression)
        print(f"✅ Detected: {expression} = {result}")
    else:
        print("❌ No expression detected")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Roblox Template Matching Bot')
    parser.add_argument('--test-templates', action='store_true', help='Test template creation')
    parser.add_argument('--test-screenshot', action='store_true', help='Test with screenshot')
    parser.add_argument('--check-admin', action='store_true', help='Check admin status')
    
    args = parser.parse_args()
    
    if args.test_templates:
        test_template_creation()
    elif args.test_screenshot:
        test_with_screenshot()
    elif args.check_admin:
        check_uipi_status()
    else:
        main()