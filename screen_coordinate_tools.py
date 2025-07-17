"""
SCREEN COORDINATE FINDER TOOL

This script helps you find the exact x, y coordinates and dimensions 
for the screen region where math problems appear.

INSTALLATION:
pip install pyautogui

USAGE:
Run this script and move your mouse to the desired positions.
The coordinates will be displayed in real-time.
"""

import pyautogui
import time

def method1_real_time_coordinates():
    """
    Method 1: Real-time coordinate display
    Move your mouse around and see coordinates in real-time
    """
    print("=" * 60)
    print("METHOD 1: REAL-TIME COORDINATE TRACKER")
    print("=" * 60)
    print("Move your mouse to see coordinates in real-time")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        while True:
            x, y = pyautogui.position()
            print(f"\rMouse position: X={x:4d}, Y={y:4d}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped tracking")

def method2_click_to_record():
    """
    Method 2: Click to record specific coordinates
    Click at corners of your desired region
    """
    print("\n" + "=" * 60)
    print("METHOD 2: CLICK TO RECORD COORDINATES")
    print("=" * 60)
    print("Instructions:")
    print("1. Click on the TOP-LEFT corner of where math problems appear")
    print("2. Then click on the BOTTOM-RIGHT corner")
    print("3. The script will calculate the region automatically")
    print()
    print("Waiting for first click (top-left corner)...")
    
    # Wait for first click (top-left)
    try:
        while True:
            if pyautogui.mouseDown():
                x1, y1 = pyautogui.position()
                print(f"✓ Top-left corner recorded: ({x1}, {y1})")
                
                # Wait for mouse release
                while pyautogui.mouseDown():
                    time.sleep(0.01)
                
                break
            time.sleep(0.01)
        
        print("Now click on the BOTTOM-RIGHT corner...")
        time.sleep(1)  # Brief pause
        
        # Wait for second click (bottom-right)
        while True:
            if pyautogui.mouseDown():
                x2, y2 = pyautogui.position()
                print(f"✓ Bottom-right corner recorded: ({x2}, {y2})")
                
                # Calculate region
                x = min(x1, x2)
                y = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                
                print("\n" + "=" * 40)
                print("CALCULATED REGION:")
                print("=" * 40)
                print(f"x = {x}")
                print(f"y = {y}")
                print(f"width = {width}")
                print(f"height = {height}")
                print("\nCopy these values to your bot script!")
                print("=" * 40)
                
                break
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nCancelled by user")

def method3_screenshot_with_overlay():
    """
    Method 3: Take screenshot and show coordinates
    """
    print("\n" + "=" * 60)
    print("METHOD 3: SCREENSHOT WITH GRID")
    print("=" * 60)
    print("Taking screenshot in 3 seconds...")
    print("This will save a screenshot with coordinate grid overlay")
    
    for i in range(3, 0, -1):
        print(f"Screenshot in {i}...")
        time.sleep(1)
    
    # Take screenshot
    screenshot = pyautogui.screenshot()
    
    # Save with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screen_coordinates_{timestamp}.png"
    screenshot.save(filename)
    
    print(f"✓ Screenshot saved as: {filename}")
    print("Open this image and note the coordinates of your target area")

def method4_interactive_selector():
    """
    Method 4: Interactive region selector with live preview
    """
    print("\n" + "=" * 60)
    print("METHOD 4: INTERACTIVE REGION SELECTOR")
    print("=" * 60)
    print("Instructions:")
    print("1. Position your mouse at the TOP-LEFT of the math problem area")
    print("2. Press SPACE to mark the top-left corner")
    print("3. Move mouse to BOTTOM-RIGHT of the area")
    print("4. Press SPACE again to mark bottom-right corner")
    print("5. Press ESC to cancel anytime")
    print()
    
    import keyboard
    
    # Disable PyAutoGUI failsafe for this method
    pyautogui.FAILSAFE = False
    
    print("Move mouse to TOP-LEFT corner and press SPACE...")
    
    top_left = None
    bottom_right = None
    
    try:
        while True:
            if keyboard.is_pressed('space'):
                if top_left is None:
                    top_left = pyautogui.position()
                    print(f"✓ Top-left set: {top_left}")
                    print("Now move to BOTTOM-RIGHT corner and press SPACE...")
                    time.sleep(0.5)  # Prevent double-trigger
                elif bottom_right is None:
                    bottom_right = pyautogui.position()
                    print(f"✓ Bottom-right set: {bottom_right}")
                    
                    # Calculate and display results
                    x = min(top_left[0], bottom_right[0])
                    y = min(top_left[1], bottom_right[1])
                    width = abs(bottom_right[0] - top_left[0])
                    height = abs(bottom_right[1] - top_left[1])
                    
                    print("\n" + "🎯 FINAL COORDINATES 🎯")
                    print("=" * 30)
                    print(f"x = {x}")
                    print(f"y = {y}")
                    print(f"width = {width}")
                    print(f"height = {height}")
                    print("=" * 30)
                    
                    # Show code snippet
                    print("\nCopy this to your bot script:")
                    print("-" * 30)
                    print("SCAN_REGION = {")
                    print(f"    'x': {x},")
                    print(f"    'y': {y},")
                    print(f"    'width': {width},")
                    print(f"    'height': {height}")
                    print("}")
                    print("-" * 30)
                    break
                    
            elif keyboard.is_pressed('esc'):
                print("Cancelled by user")
                break
                
            time.sleep(0.01)
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    """
    Main menu to choose coordinate finding method
    """
    print("SCREEN COORDINATE FINDER")
    print("Choose a method to find your screen coordinates:")
    print()
    print("1. Real-time coordinate tracker (move mouse around)")
    print("2. Click to record region corners")
    print("3. Take screenshot with grid")
    print("4. Interactive region selector (recommended)")
    print("5. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                method1_real_time_coordinates()
            elif choice == '2':
                method2_click_to_record()
            elif choice == '3':
                method3_screenshot_with_overlay()
            elif choice == '4':
                method4_interactive_selector()
            elif choice == '5':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
                continue
                
            # Ask if user wants to try another method
            if choice in ['1', '2', '3', '4']:
                again = input("\nTry another method? (y/n): ").strip().lower()
                if again != 'y':
                    break
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()