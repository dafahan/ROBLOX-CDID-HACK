"""
ENHANCED ROBLOX MATH PROBLEM SOLVER BOT - LINUX VERSION

IMPORTANT DISCLAIMER:
Using automation bots in Roblox violates their Terms of Service and can result in 
permanent account suspension or ban. This script is provided for educational 
purposes only. Use at your own risk.

LINUX SETUP:
sudo apt install tesseract-ocr tesseract-ocr-eng imagemagick
pip install pyautogui opencv-python pytesseract pillow numpy

IMPROVEMENTS:
- Linux-compatible failsafe system
- Better delays and timing
- Robust clicking with verification
- Debug mode to see what's happening
- Multiple retry attempts
- Mouse movement simulation
- X11 display fixes
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import re
import signal
import subprocess
import tempfile

# Fix X11 display issues on Linux
if sys.platform.startswith('linux'):
    os.environ['DISPLAY'] = ':0'
    try:
        os.system('xhost +local: 2>/dev/null')
    except:
        pass

import pyautogui
import cv2
import pytesseract

# Linux-safe keyboard handling
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("⚠️ Keyboard library not available. Using file-based stopping only.")
    KEYBOARD_AVAILABLE = False

# =============================================================================
# ENHANCED CONFIGURATION
# =============================================================================

# Screen region to monitor
SCAN_REGION = {
    'x': 831,
    'y': 585,
    'width': 254,
    'height': 60
}

# Click coordinates
INPUT_COORDS = (997, 758)
SUBMIT_COORDS = (1134, 758)

# OCR Configuration
OCR_CONFIG = '--psm 6 -c tessedit_char_whitelist=0123456789+-*/'

# Enhanced timing settings
TIMING = {
    'LOOP_DELAY': 1.0,          # Delay between scan loops
    'CLICK_DELAY': 0.5,         # Delay after clicking
    'TYPE_DELAY': 0.15,         # Delay between characters
    'SUBMIT_DELAY': 0.8,        # Delay before clicking submit
    'POST_SUBMIT_DELAY': 3.0,   # Delay after submitting
    'RETRY_DELAY': 0.3,         # Delay between retries
    'FOCUS_DELAY': 0.2,         # Additional delay for input focus
}

# Debug settings
DEBUG_MODE = True  # Set to True for detailed logging
SAVE_SCREENSHOTS = True  # Save screenshots for debugging

# Linux-safe failsafe
FAILSAFE_KEY = 'q'
STOP_FILE = 'stop_bot.txt'
MAX_RUNTIME_MINUTES = 30  # Auto-stop after 30 minutes

# Global stop flag
stop_bot = False

# =============================================================================
# LINUX-SAFE SCREENSHOT FUNCTIONS
# =============================================================================

def linux_safe_screenshot(region=None):
    """
    Linux-safe screenshot function that tries multiple methods
    """
    # Method 1: Try ImageMagick import
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            if region:
                x, y, width, height = region
                cmd = ['import', '-window', 'root', '-crop', f'{width}x{height}+{x}+{y}', tmp.name]
            else:
                cmd = ['import', '-window', 'root', tmp.name]
            
            result = subprocess.run(cmd, capture_output=True, check=True)
            if result.returncode == 0:
                image = Image.open(tmp.name)
                os.unlink(tmp.name)
                return image
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Method 2: Try scrot
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            if region:
                x, y, width, height = region
                cmd = ['scrot', '-a', f'{x},{y},{width},{height}', tmp.name]
            else:
                cmd = ['scrot', tmp.name]
            
            result = subprocess.run(cmd, capture_output=True, check=True)
            if result.returncode == 0:
                image = Image.open(tmp.name)
                os.unlink(tmp.name)
                return image
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Method 3: Fallback to pyautogui
    try:
        if region:
            return pyautogui.screenshot(region=region)
        else:
            return pyautogui.screenshot()
    except Exception as e:
        raise Exception(f"All screenshot methods failed. Install: sudo apt install imagemagick OR sudo apt install scrot")

# =============================================================================
# LINUX-SAFE FAILSAFE SYSTEM
# =============================================================================

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global stop_bot
    print("\n\nReceived stop signal (Ctrl+C)")
    stop_bot = True

def setup_failsafe():
    """Set up Linux-safe failsafe mechanisms"""
    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Remove any existing stop file
    try:
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
    except:
        pass

def safe_keyboard_check():
    """Linux-safe keyboard checking"""
    global stop_bot
    
    # Check global stop flag
    if stop_bot:
        return True
    
    # Check stop file
    if os.path.exists(STOP_FILE):
        print(f"\nStop file '{STOP_FILE}' found")
        stop_bot = True
        try:
            os.remove(STOP_FILE)  # Clean up
        except:
            pass
        return True
    
    # Try keyboard if available
    try:
        if KEYBOARD_AVAILABLE:
            if keyboard.is_pressed(FAILSAFE_KEY):
                stop_bot = True
                return True
    except Exception:
        pass
    
    return False

# =============================================================================
# ENHANCED UTILITY FUNCTIONS
# =============================================================================

def debug_log(message, level="INFO"):
    """Enhanced logging function"""
    if DEBUG_MODE:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

def move_mouse_naturally(start_pos, end_pos, duration=0.3):
    """Move mouse in a natural curve"""
    try:
        debug_log(f"Moving mouse from {start_pos} to {end_pos}")
        
        # Move to start position first
        pyautogui.moveTo(start_pos[0], start_pos[1], duration=duration/2)
        time.sleep(0.1)
        
        # Move to end position
        pyautogui.moveTo(end_pos[0], end_pos[1], duration=duration)
        time.sleep(0.1)
        
        return True
    except Exception as e:
        debug_log(f"Mouse movement failed: {e}", "ERROR")
        return False

def enhanced_click(coords, click_type="left", retries=3):
    """Enhanced clicking with retries and verification"""
    for attempt in range(retries):
        try:
            debug_log(f"Clicking at {coords} (attempt {attempt + 1})")
            
            # Move mouse naturally to position
            current_pos = pyautogui.position()
            if not move_mouse_naturally(current_pos, coords):
                continue
            
            # Perform click
            pyautogui.click(coords[0], coords[1], button=click_type)
            time.sleep(TIMING['CLICK_DELAY'])
            
            # Verify click worked (mouse should be at target position)
            final_pos = pyautogui.position()
            if abs(final_pos.x - coords[0]) <= 5 and abs(final_pos.y - coords[1]) <= 5:
                debug_log(f"✓ Click successful at {coords}")
                return True
            else:
                debug_log(f"Click verification failed. Expected {coords}, got {final_pos}")
                
        except Exception as e:
            debug_log(f"Click attempt {attempt + 1} failed: {e}", "ERROR")
            
        time.sleep(TIMING['RETRY_DELAY'])
    
    debug_log(f"❌ All click attempts failed for {coords}", "ERROR")
    return False

def robust_text_input(text, clear_first=True, retries=3):
    """Robust text input with multiple methods"""
    text_str = str(text)
    
    for attempt in range(retries):
        try:
            debug_log(f"Typing text: '{text_str}' (attempt {attempt + 1})")
            
            # Method 1: Clear field first if requested
            if clear_first:
                # Try multiple clear methods
                pyautogui.hotkey('ctrl', 'a')
                time.sleep(0.1)
                pyautogui.press('delete')
                time.sleep(0.1)
                pyautogui.press('backspace')
                time.sleep(0.1)
            
            # Method 2: Type character by character with delays
            for char in text_str:
                pyautogui.typewrite(char)
                time.sleep(TIMING['TYPE_DELAY'])
            
            debug_log(f"✓ Text input successful: '{text_str}'")
            return True
            
        except Exception as e:
            debug_log(f"Text input attempt {attempt + 1} failed: {e}", "ERROR")
            
        time.sleep(TIMING['RETRY_DELAY'])
    
    debug_log(f"❌ All text input attempts failed for '{text_str}'", "ERROR")
    return False

def preprocess_image(image):
    """Image preprocessing for OCR"""
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    return cleaned

def extract_math_expression(processed_image):
    """OCR text extraction"""
    try:
        text = pytesseract.image_to_string(processed_image, config=OCR_CONFIG)
        text = text.strip().replace(' ', '').replace('\n', '')
        
        if text and any(op in text for op in ['+', '-', '*', '/']):
            debug_log(f"OCR detected: '{text}'")
            return text
        else:
            debug_log(f"OCR result not a math expression: '{text}'")
            return None
            
    except Exception as e:
        debug_log(f"OCR extraction failed: {e}", "ERROR")
        return None

def calculate_result(expression):
    """Safe math calculation"""
    try:
        if not re.match(r'^[0-9+\-*/\s]+$', expression):
            debug_log(f"Invalid characters in expression: {expression}", "ERROR")
            return None
        
        result = eval(expression)
        
        if isinstance(result, (int, float)):
            debug_log(f"Calculation: {expression} = {int(result)}")
            return int(result)
        else:
            return None
            
    except Exception as e:
        debug_log(f"Calculation failed for '{expression}': {e}", "ERROR")
        return None

def enhanced_answer_submission(answer):
    """Enhanced answer submission with full debugging"""
    debug_log(f"🎯 Starting answer submission for: {answer}")
    
    try:
        # Step 1: Click input field with enhanced clicking
        debug_log("Step 1: Clicking input field...")
        if not enhanced_click(INPUT_COORDS):
            debug_log("❌ Failed to click input field", "ERROR")
            return False
        
        # Step 2: Wait for focus
        debug_log(f"Step 2: Waiting {TIMING['FOCUS_DELAY']}s for input focus...")
        time.sleep(TIMING['FOCUS_DELAY'])
        
        # Step 3: Type answer with robust input
        debug_log("Step 3: Typing answer...")
        if not robust_text_input(answer):
            debug_log("❌ Failed to type answer", "ERROR")
            return False
        
        # Step 4: Wait before submitting
        debug_log(f"Step 4: Waiting {TIMING['SUBMIT_DELAY']}s before submit...")
        time.sleep(TIMING['SUBMIT_DELAY'])
        
        # Step 5: Click submit button
        debug_log("Step 5: Clicking submit button...")
        if not enhanced_click(SUBMIT_COORDS):
            debug_log("❌ Failed to click submit button", "ERROR")
            return False
        
        # Step 6: Wait for form processing
        debug_log(f"Step 6: Waiting {TIMING['POST_SUBMIT_DELAY']}s for form processing...")
        time.sleep(TIMING['POST_SUBMIT_DELAY'])
        
        debug_log("✅ Answer submission completed successfully!")
        return True
        
    except Exception as e:
        debug_log(f"❌ Answer submission failed: {e}", "ERROR")
        return False

def save_debug_screenshot(image, filename_prefix):
    """Save screenshots for debugging"""
    if SAVE_SCREENSHOTS:
        try:
            timestamp = int(time.time())
            filename = f"{filename_prefix}_{timestamp}.png"
            image.save(filename)
            debug_log(f"💾 Screenshot saved: {filename}")
        except Exception as e:
            debug_log(f"Failed to save screenshot: {e}", "ERROR")

# =============================================================================
# LINUX COORDINATE HELPER
# =============================================================================

def linux_coordinate_helper():
    """Linux coordinate finder"""
    print("\n🎯 LINUX COORDINATE HELPER")
    print("Move your mouse over elements and note coordinates:")
    print("- Math problem area (for OCR)")
    print("- Input field center")
    print("- Submit button center")
    print("Press Ctrl+C when done")
    
    try:
        while True:
            x, y = pyautogui.position()
            print(f"\rMouse: X={x:4d}, Y={y:4d} | Use: ({x}, {y})", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n✅ Last position: X={x}, Y={y}")

def test_screenshot_methods():
    """Test which screenshot methods work"""
    print("🔍 Testing screenshot methods...")
    
    methods = [
        ("ImageMagick", lambda: subprocess.run(['import', '--version'], capture_output=True)),
        ("Scrot", lambda: subprocess.run(['scrot', '--version'], capture_output=True)),
        ("PyAutoGUI", lambda: pyautogui.screenshot())
    ]
    
    for name, test_func in methods:
        try:
            test_func()
            print(f"✓ {name} available")
        except:
            print(f"❌ {name} not available")

# =============================================================================
# MAIN BOT LOGIC
# =============================================================================

def main():
    """Enhanced main bot loop for Linux"""
    print("=" * 70)
    print("🐧 ENHANCED LINUX ROBLOX MATH PROBLEM SOLVER BOT")
    print("=" * 70)
    print(f"📍 Scan region: {SCAN_REGION}")
    print(f"🎯 Input coords: {INPUT_COORDS}")
    print(f"🔘 Submit coords: {SUBMIT_COORDS}")
    print(f"🐛 Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print(f"📸 Save screenshots: {'ON' if SAVE_SCREENSHOTS else 'OFF'}")
    
    print(f"\n🛑 STOP METHODS:")
    if KEYBOARD_AVAILABLE:
        print(f"   1. Press '{FAILSAFE_KEY}' key")
    print(f"   2. Create file '{STOP_FILE}'")
    print(f"   3. Press Ctrl+C")
    print(f"   4. Auto-stop after {MAX_RUNTIME_MINUTES} minutes")
    print("=" * 70)
    
    # Set up failsafe
    setup_failsafe()
    
    # Configure PyAutoGUI
    pyautogui.FAILSAFE = False  # We handle our own failsafe
    pyautogui.PAUSE = 0.1       # Small pause between actions
    
    # Calculate stop time
    from datetime import datetime, timedelta
    start_time = datetime.now()
    stop_time = start_time + timedelta(minutes=MAX_RUNTIME_MINUTES)
    
    loop_count = 0
    successful_submissions = 0
    failed_submissions = 0
    
    try:
        while not stop_bot:
            loop_count += 1
            current_time = datetime.now()
            
            debug_log(f"🔄 Starting loop {loop_count} (Runtime: {current_time - start_time})")
            
            # Check stop conditions
            if safe_keyboard_check():
                debug_log("🛑 Stop condition detected")
                break
            
            # Check time limit
            if current_time >= stop_time:
                print(f"\n⏰ Time limit reached ({MAX_RUNTIME_MINUTES} minutes)")
                break
            
            try:
                # Capture screenshot using Linux-safe method
                debug_log("📸 Capturing screenshot...")
                screenshot = linux_safe_screenshot(region=(
                    SCAN_REGION['x'], 
                    SCAN_REGION['y'], 
                    SCAN_REGION['width'], 
                    SCAN_REGION['height']
                ))
                
                save_debug_screenshot(screenshot, "scan_region")
                
                # Process image
                debug_log("🔍 Processing image for OCR...")
                processed_image = preprocess_image(screenshot)
                
                # Extract math expression
                expression = extract_math_expression(processed_image)
                
                if expression:
                    print(f"\n📖 Detected: {expression}")
                    
                    # Calculate result
                    result = calculate_result(expression)
                    
                    if result is not None:
                        print(f"🧮 Result: {expression} = {result}")
                        
                        # Submit answer
                        if enhanced_answer_submission(result):
                            successful_submissions += 1
                            print(f"✅ SUCCESS! (✓{successful_submissions} ❌{failed_submissions})")
                        else:
                            failed_submissions += 1
                            print(f"❌ FAILED! (✓{successful_submissions} ❌{failed_submissions})")
                            
                    else:
                        debug_log("❌ Failed to calculate result")
                else:
                    debug_log("👁️ No math expression detected")
                
            except Exception as e:
                debug_log(f"❌ Loop error: {e}", "ERROR")
                failed_submissions += 1
            
            # Wait before next iteration
            debug_log(f"⏰ Waiting {TIMING['LOOP_DELAY']}s before next scan...")
            time.sleep(TIMING['LOOP_DELAY'])
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        success_rate = (successful_submissions / max(1, successful_submissions + failed_submissions)) * 100
        print("\n" + "=" * 70)
        print("📊 FINAL STATISTICS")
        print("=" * 70)
        print(f"🔄 Total loops: {loop_count}")
        print(f"✅ Successful submissions: {successful_submissions}")
        print(f"❌ Failed submissions: {failed_submissions}")
        print(f"📈 Success rate: {success_rate:.1f}%")
        print(f"⏰ Runtime: {datetime.now() - start_time}")
        print("=" * 70)
        print("🐧 LINUX BOT STOPPED")
        print("=" * 70)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Linux Roblox Math Bot')
    parser.add_argument('--coords', action='store_true', help='Find coordinates')
    parser.add_argument('--test-screenshot', action='store_true', help='Test screenshot methods')
    parser.add_argument('--test-region', action='store_true', help='Test scan region')
    parser.add_argument('--stop', action='store_true', help='Create stop file')
    
    args = parser.parse_args()
    
    if args.coords:
        linux_coordinate_helper()
    elif args.test_screenshot:
        test_screenshot_methods()
    elif args.test_region:
        try:
            screenshot = linux_safe_screenshot(region=(
                SCAN_REGION['x'], SCAN_REGION['y'], 
                SCAN_REGION['width'], SCAN_REGION['height']
            ))
            screenshot.save("test_region.png")
            print("✓ Test screenshot saved as 'test_region.png'")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    elif args.stop:
        try:
            with open(STOP_FILE, 'w') as f:
                f.write('stop')
            print(f"✓ Created stop file '{STOP_FILE}'")
        except Exception as e:
            print(f"❌ Failed to create stop file: {e}")
    else:
        # Check dependencies
        print("🔍 Checking dependencies...")
        try:
            import pyautogui, cv2, pytesseract
            print("✓ Python libraries installed")
        except ImportError as e:
            print(f"❌ Missing Python library: {e}")
            exit(1)
        
        # Check Tesseract
        try:
            pytesseract.get_tesseract_version()
            print("✓ Tesseract OCR available")
        except Exception as e:
            print(f"❌ Tesseract issue: {e}")
            print("Install with: sudo apt install tesseract-ocr")
        
        # Test screenshot methods
        test_screenshot_methods()
        
        # Display mouse position helper
        print("\n🎯 MOUSE POSITION HELPER")
        print("Move your mouse to note coordinates...")
        for i in range(5):
            try:
                x, y = pyautogui.position()
                print(f"\rMouse position: X={x}, Y={y}", end="", flush=True)
            except:
                print(f"\rMouse position: (unable to detect)", end="", flush=True)
            time.sleep(1)
        print("\n")
        
        # Start bot
        input("Press Enter to start the Linux bot (or Ctrl+C to cancel)...")
        main()