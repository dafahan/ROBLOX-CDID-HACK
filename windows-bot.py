"""
FIXED CLICKING BOT - WINDOWS VERSION

This version fixes the clicking issues where the bot only hovers instead of clicking.

FIXES APPLIED:
- Multiple click methods with retries
- Proper mouse button specification
- Click verification
- Alternative clicking approaches
- Better timing and delays
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import re

# Windows-specific imports and setup
import pyautogui
import cv2
import pytesseract

# Configure Tesseract path for Windows
if sys.platform.startswith('win'):
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
        r'C:\tesseract\tesseract.exe'
    ]
    
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
        input("Press Enter to continue anyway (might fail)...")

# Windows keyboard handling
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("⚠️ Keyboard library not available. Using file-based stopping only.")
    KEYBOARD_AVAILABLE = False

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

SCAN_REGION = {
    'x': 792,
    'y': 484,
    'width': 133,
    'height': 50
}

INPUT_COORDS = (1043, 519)
SUBMIT_COORDS = (950, 604)

OCR_CONFIG = '--psm 6 -c tessedit_char_whitelist=0123456789+-*/'

# Enhanced timing for better clicking
DELAYS = {
    'LOOP_DELAY': 1.0,
    'INPUT_CLICK_DELAY': 0.5,    # Longer delay after input click
    'FOCUS_DELAY': 0.3,          # More time for focus
    'TYPE_DELAY': 0.1,
    'CLEAR_DELAY': 0.2,
    'SUBMIT_CLICK_DELAY': 0.8,   # Longer delay before submit
    'POST_SUBMIT_DELAY': 3.0,    # Longer wait after submit
    'ERROR_RETRY_DELAY': 1.0,
    'CLICK_VERIFICATION_DELAY': 0.2,  # Time to verify click worked
}

DEBUG_MODE = True
SAVE_SCREENSHOTS = False
FAILSAFE_KEY = 'q'
STOP_FILE = 'stop_bot.txt'

# =============================================================================
# ENHANCED CLICKING FUNCTIONS
# =============================================================================

def debug_print(message):
    if DEBUG_MODE:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

def robust_click(coords, description="element", retries=5):
    """
    Ultra-robust clicking function with multiple methods and verification
    """
    x, y = coords
    debug_print(f"🎯 Attempting to click {description} at ({x}, {y})")
    
    for attempt in range(retries):
        try:
            debug_print(f"Click attempt {attempt + 1}/{retries}")
            
            # Method 1: Standard click with explicit button specification
            debug_print(f"Method 1: Standard left click")
            pyautogui.moveTo(x, y, duration=0.3)
            time.sleep(0.2)
            pyautogui.click(x, y, button='left')
            time.sleep(DELAYS['CLICK_VERIFICATION_DELAY'])
            
            # Verify the click worked by checking mouse position
            final_pos = pyautogui.position()
            if abs(final_pos.x - x) <= 3 and abs(final_pos.y - y) <= 3:
                debug_print(f"✅ Click successful using Method 1")
                return True
            
            # Method 2: Double click
            debug_print(f"Method 2: Double click")
            pyautogui.doubleClick(x, y)
            time.sleep(DELAYS['CLICK_VERIFICATION_DELAY'])
            
            # Method 3: Mouse down/up explicitly
            debug_print(f"Method 3: Explicit mouse down/up")
            pyautogui.moveTo(x, y, duration=0.2)
            time.sleep(0.1)
            pyautogui.mouseDown(x, y, button='left')
            time.sleep(0.05)
            pyautogui.mouseUp(x, y, button='left')
            time.sleep(DELAYS['CLICK_VERIFICATION_DELAY'])
            
            # Method 4: Right click then left click (sometimes helps)
            if attempt >= 2:
                debug_print(f"Method 4: Right click + left click")
                pyautogui.rightClick(x, y)
                time.sleep(0.1)
                pyautogui.click(x, y, button='left')
                time.sleep(DELAYS['CLICK_VERIFICATION_DELAY'])
            
            # Method 5: Multiple rapid clicks (last resort)
            if attempt >= 3:
                debug_print(f"Method 5: Multiple rapid clicks")
                for i in range(3):
                    pyautogui.click(x, y, button='left')
                    time.sleep(0.05)
                time.sleep(DELAYS['CLICK_VERIFICATION_DELAY'])
            
        except Exception as e:
            debug_print(f"❌ Click attempt {attempt + 1} failed: {e}")
        
        # Wait before next attempt
        time.sleep(0.3)
    
    debug_print(f"❌ All {retries} click attempts failed for {description}")
    return False

def alternative_submit_methods(answer):
    """
    Alternative methods to submit the answer if clicking fails
    """
    debug_print("🔄 Trying alternative submit methods...")
    
    try:
        # Method 1: Press Enter key
        debug_print("Alternative Method 1: Press Enter")
        pyautogui.press('enter')
        time.sleep(0.5)
        
        # Method 2: Press Tab then Enter (in case focus is wrong)
        debug_print("Alternative Method 2: Tab + Enter")
        pyautogui.press('tab')
        time.sleep(0.2)
        pyautogui.press('enter')
        time.sleep(0.5)
        
        # Method 3: Re-type answer and press Enter
        debug_print("Alternative Method 3: Re-type + Enter")
        pyautogui.typewrite(str(answer))
        time.sleep(0.2)
        pyautogui.press('enter')
        time.sleep(0.5)
        
        debug_print("✅ Alternative submit methods completed")
        return True
        
    except Exception as e:
        debug_print(f"❌ Alternative submit methods failed: {e}")
        return False

def windows_safe_keyboard_check():
    try:
        if os.path.exists(STOP_FILE):
            return True
        if KEYBOARD_AVAILABLE:
            return keyboard.is_pressed(FAILSAFE_KEY)
        return False
    except Exception:
        return os.path.exists(STOP_FILE)

def enhanced_text_input(text, clear_first=True):
    """Enhanced text input with better clearing"""
    try:
        text_str = str(text)
        debug_print(f"⌨️ Typing text: '{text_str}'")
        
        if clear_first:
            # Multiple clearing methods
            debug_print("Clearing input field...")
            
            # Method 1: Ctrl+A and Delete
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(DELAYS['CLEAR_DELAY'])
            pyautogui.press('delete')
            time.sleep(DELAYS['CLEAR_DELAY'])
            
            # Method 2: Multiple backspaces
            for _ in range(10):
                pyautogui.press('backspace')
                time.sleep(0.02)
            
            time.sleep(DELAYS['CLEAR_DELAY'])
        
        # Type each character slowly
        for i, char in enumerate(text_str):
            debug_print(f"  Typing char {i+1}/{len(text_str)}: '{char}'")
            pyautogui.typewrite(char)
            time.sleep(DELAYS['TYPE_DELAY'])
        
        debug_print(f"✅ Successfully typed: '{text_str}'")
        return True
        
    except Exception as e:
        debug_print(f"❌ Failed to type text: {e}")
        return False

# =============================================================================
# OCR FUNCTIONS
# =============================================================================

def preprocess_image(image):
    debug_print("🔍 Processing image for OCR...")
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    debug_print("✅ Image processing completed")
    return cleaned

def extract_math_expression(processed_image):
    try:
        debug_print("🔍 Running OCR...")
        text = pytesseract.image_to_string(processed_image, config=OCR_CONFIG)
        text = text.strip().replace(' ', '').replace('\n', '')
        debug_print(f"OCR result: '{text}'")
        
        if text and any(op in text for op in ['+', '-', '*', '/']):
            debug_print(f"✅ Valid math expression: '{text}'")
            return text
        else:
            debug_print("❌ No valid math expression found")
            return None
    except Exception as e:
        debug_print(f"❌ OCR failed: {e}")
        return None

def calculate_result(expression):
    try:
        debug_print(f"🧮 Calculating: {expression}")
        if not re.match(r'^[0-9+\-*/\s]+$', expression):
            debug_print(f"❌ Invalid characters: {expression}")
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
# ENHANCED ANSWER SUBMISSION
# =============================================================================

def enhanced_answer_submission(answer):
    """
    Enhanced answer submission with multiple clicking methods
    """
    debug_print(f"🎯 Starting enhanced answer submission for: {answer}")
    
    try:
        # Step 1: Click input field with robust clicking
        debug_print("Step 1: Clicking input field...")
        if not robust_click(INPUT_COORDS, "input field"):
            debug_print("❌ Failed to click input field")
            return False
        
        time.sleep(DELAYS['INPUT_CLICK_DELAY'])
        
        # Step 2: Focus verification
        debug_print(f"Step 2: Waiting {DELAYS['FOCUS_DELAY']}s for focus...")
        time.sleep(DELAYS['FOCUS_DELAY'])
        
        # Step 3: Clear and type answer
        debug_print("Step 3: Typing answer...")
        if not enhanced_text_input(answer, clear_first=True):
            debug_print("❌ Failed to type answer")
            return False
        
        # Step 4: Wait before submit
        debug_print(f"Step 4: Waiting {DELAYS['SUBMIT_CLICK_DELAY']}s before submit...")
        time.sleep(DELAYS['SUBMIT_CLICK_DELAY'])
        
        # Step 5: Try robust clicking on submit button
        debug_print("Step 5: Clicking submit button...")
        submit_success = robust_click(SUBMIT_COORDS, "submit button")
        
        # Step 6: If clicking failed, try alternative methods
        if not submit_success:
            debug_print("⚠️ Submit button click failed, trying alternatives...")
            alternative_submit_methods(answer)
        
        # Step 7: Wait for processing
        debug_print(f"Step 7: Waiting {DELAYS['POST_SUBMIT_DELAY']}s for processing...")
        time.sleep(DELAYS['POST_SUBMIT_DELAY'])
        
        debug_print("✅ Answer submission sequence completed!")
        return True
        
    except Exception as e:
        debug_print(f"❌ Answer submission failed: {e}")
        return False

# =============================================================================
# MAIN BOT LOGIC
# =============================================================================

def main():
    print("=" * 70)
    print("🔧 FIXED CLICKING BOT - WINDOWS VERSION")
    print("=" * 70)
    print(f"📍 Scan region: {SCAN_REGION}")
    print(f"🎯 Input field: {INPUT_COORDS}")
    print(f"🔘 Submit button: {SUBMIT_COORDS}")
    print(f"🐛 Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print("\n🔧 CLICK FIXES APPLIED:")
    print("   - Multiple click methods with retries")
    print("   - Explicit button specification")
    print("   - Click verification")
    print("   - Alternative submit methods")
    print("   - Better timing and delays")
    print("=" * 70)
    
    # Configure PyAutoGUI
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    
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
                
                if SAVE_SCREENSHOTS:
                    timestamp = int(time.time())
                    screenshot.save(f"screenshot_{timestamp}.png")
                
                # Process and extract
                processed_image = preprocess_image(screenshot)
                expression = extract_math_expression(processed_image)
                
                if expression:
                    print(f"📖 Detected: {expression}")
                    result = calculate_result(expression)
                    
                    if result is not None:
                        print(f"🧮 Answer: {result}")
                        
                        # Submit with enhanced clicking
                        if enhanced_answer_submission(result):
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
            
            debug_print(f"⏰ Waiting {DELAYS['LOOP_DELAY']}s...")
            time.sleep(DELAYS['LOOP_DELAY'])
            
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except pyautogui.FailSafeException:
        print("\n🛑 Bot stopped by failsafe")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        total = successful_submissions + failed_submissions
        success_rate = (successful_submissions / max(1, total)) * 100
        
        print("\n" + "=" * 70)
        print("📊 FINAL STATISTICS")
        print("=" * 70)
        print(f"🔄 Total loops: {loop_count}")
        print(f"✅ Successful: {successful_submissions}")
        print(f"❌ Failed: {failed_submissions}")
        print(f"📈 Success rate: {success_rate:.1f}%")
        print("=" * 70)

def test_clicking():
    """Test the clicking functionality"""
    print("🧪 TESTING CLICKING FUNCTIONALITY")
    print("=" * 50)
    
    print("This will test both input field and submit button clicking")
    print("Make sure your target application is visible!")
    
    input("\nPress Enter to test INPUT FIELD clicking...")
    success1 = robust_click(INPUT_COORDS, "input field")
    print(f"Input field click: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    
    input("\nPress Enter to test SUBMIT BUTTON clicking...")
    success2 = robust_click(SUBMIT_COORDS, "submit button")
    print(f"Submit button click: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    
    if not success1 or not success2:
        print("\n⚠️ CLICKING ISSUES DETECTED")
        print("Possible solutions:")
        print("1. Run as Administrator")
        print("2. Check coordinates are correct")
        print("3. Ensure target window is active")
        print("4. Try different coordinates")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed Clicking Bot')
    parser.add_argument('--test-click', action='store_true', help='Test clicking functionality')
    parser.add_argument('--coords', action='store_true', help='Find coordinates')
    
    args = parser.parse_args()
    
    if args.test_click:
        test_clicking()
    elif args.coords:
        print("Move mouse to get coordinates, Ctrl+C to stop:")
        try:
            while True:
                x, y = pyautogui.position()
                print(f"\rX: {x}, Y: {y}", end="", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print(f"\nLast position: ({x}, {y})")
    else:
        # Check dependencies
        try:
            print("Checking dependencies...")
            import pyautogui, cv2, pytesseract
            print("✓ All libraries available")
        except ImportError as e:
            print(f"❌ Missing: {e}")
            exit(1)
        
        try:
            pytesseract.get_tesseract_version()
            print("✓ Tesseract available")
        except Exception as e:
            print(f"⚠️ Tesseract issue: {e}")
        
        main()