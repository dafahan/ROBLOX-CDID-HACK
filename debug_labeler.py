"""
DEBUG VERSION - CHARACTER REGION DETECTION

This version adds extensive debugging to help identify why character regions
aren't being detected or clicked properly.
"""

import os
import sys
import time
import json
import numpy as np
import cv2
from PIL import Image, ImageTk
import pyautogui
import threading
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog, filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    print("❌ tkinter not available")
    TKINTER_AVAILABLE = False

class DebugRobloxLabeler:
    def __init__(self):
        if not TKINTER_AVAILABLE:
            print("❌ Cannot start GUI - tkinter not available")
            return
            
        # Configuration
        self.scan_region = {
            'x': 792,
            'y': 484, 
            'width': 133,
            'height': 50
        }
        
        # Data storage
        self.data_dir = "roblox_training_data"
        self.screenshots_dir = f"{self.data_dir}/screenshots"
        self.characters_dir = f"{self.data_dir}/characters"
        self.debug_dir = f"{self.data_dir}/debug"
        
        # Create directories
        for dir_path in [self.data_dir, self.screenshots_dir, self.characters_dir, self.debug_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Character classes
        self.character_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-']
        
        # Tracking
        self.captured_screenshots = []
        self.current_screenshot = None
        self.character_regions = []
        self.selected_region = None
        self.debug_images = {}
        
        # Setup UI
        self.setup_ui()
        
        print("🐛 DEBUG MODE ENABLED")
        print("This version will show detailed information about character detection")
    
    def setup_ui(self):
        """Setup the debugging interface"""
        self.root = tk.Tk()
        self.root.title("DEBUG: Roblox Character Region Detector")
        self.root.geometry("1200x900")
        
        # Main frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Control buttons
        ttk.Button(control_frame, text="📸 Take Screenshot", 
                  command=self.debug_capture).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="🔍 Analyze Image", 
                  command=self.debug_analyze).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="💾 Save Debug Images", 
                  command=self.save_debug_images).pack(side='left', padx=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready for debugging")
        self.status_label.pack(side='right', padx=5)
        
        # Notebook for different views
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Original image tab
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="Original")
        
        self.original_canvas = tk.Canvas(self.original_frame, bg='white')
        self.original_canvas.pack(fill='both', expand=True)
        
        # Processed image tab
        self.processed_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.processed_frame, text="Processed")
        
        self.processed_canvas = tk.Canvas(self.processed_frame, bg='white')
        self.processed_canvas.pack(fill='both', expand=True)
        
        # Regions tab
        self.regions_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.regions_frame, text="Detected Regions")
        
        self.regions_canvas = tk.Canvas(self.regions_frame, bg='white')
        self.regions_canvas.pack(fill='both', expand=True)
        
        # Debug info tab
        self.info_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.info_frame, text="Debug Info")
        
        # Debug text area
        self.debug_text = tk.Text(self.info_frame, wrap='word')
        debug_scroll = ttk.Scrollbar(self.info_frame, orient='vertical', command=self.debug_text.yview)
        self.debug_text.configure(yscrollcommand=debug_scroll.set)
        
        debug_scroll.pack(side='right', fill='y')
        self.debug_text.pack(side='left', fill='both', expand=True)
        
        # Labeling frame
        label_frame = ttk.Frame(self.root)
        label_frame.pack(fill='x', padx=10, pady=5)
        
        self.create_label_buttons(label_frame)
        
        # Bind click events to all canvases
        self.original_canvas.bind("<Button-1>", self.on_image_click)
        self.regions_canvas.bind("<Button-1>", self.on_image_click)
    
    def create_label_buttons(self, parent):
        """Create labeling buttons"""
        ttk.Label(parent, text="Click on character regions above, then select label:").pack()
        
        button_frame = ttk.Frame(parent)
        button_frame.pack()
        
        # Number buttons
        for i in range(10):
            btn = ttk.Button(button_frame, text=str(i), width=3,
                           command=lambda x=str(i): self.label_selected_region(x))
            btn.pack(side='left', padx=1)
        
        # Operator buttons
        for op in ['+', '-']:
            btn = ttk.Button(button_frame, text=op, width=3,
                           command=lambda x=op: self.label_selected_region(x))
            btn.pack(side='left', padx=1)
        
        # Skip button
        ttk.Button(button_frame, text="❌ Skip", width=5,
                  command=lambda: self.label_selected_region('skip')).pack(side='left', padx=5)
    
    def debug_log(self, message):
        """Add message to debug log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.debug_text.insert(tk.END, log_message)
        self.debug_text.see(tk.END)
        print(log_message.strip())
    
    def debug_capture(self):
        """Take a screenshot and analyze it"""
        try:
            self.debug_log("📸 Taking screenshot...")
            
            screenshot = pyautogui.screenshot(region=(
                self.scan_region['x'],
                self.scan_region['y'],
                self.scan_region['width'],
                self.scan_region['height']
            ))
            
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debug_{timestamp}.png"
            filepath = f"{self.screenshots_dir}/{filename}"
            screenshot.save(filepath)
            
            self.current_screenshot = filepath
            self.debug_log(f"✅ Screenshot saved: {filename}")
            self.debug_log(f"Image size: {screenshot.size}")
            
            # Automatically analyze
            self.debug_analyze()
            
        except Exception as e:
            self.debug_log(f"❌ Screenshot failed: {e}")
            messagebox.showerror("Error", f"Screenshot failed: {e}")
    
    def debug_analyze(self):
        """Analyze the current screenshot"""
        if not self.current_screenshot or not os.path.exists(self.current_screenshot):
            messagebox.showwarning("Warning", "No screenshot to analyze. Take a screenshot first.")
            return
        
        try:
            self.debug_log("🔍 Starting image analysis...")
            
            # Load image
            img_color = cv2.imread(self.current_screenshot)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            
            self.debug_log(f"Image shape: {img_gray.shape}")
            self.debug_log(f"Image dtype: {img_gray.dtype}")
            self.debug_log(f"Pixel value range: {img_gray.min()} - {img_gray.max()}")
            
            # Show original image
            self.show_image_on_canvas(img_color, self.original_canvas, "original")
            
            # Process image step by step
            self.debug_process_image(img_gray)
            
            # Find character regions
            self.character_regions = self.debug_find_regions(img_gray)
            
            # Show regions
            self.show_detected_regions(img_color)
            
            self.status_label.config(text=f"Analysis complete - Found {len(self.character_regions)} regions")
            
        except Exception as e:
            self.debug_log(f"❌ Analysis failed: {e}")
            messagebox.showerror("Error", f"Analysis failed: {e}")
    
    def debug_process_image(self, img_gray):
        """Debug image processing steps"""
        self.debug_log("🔄 Processing image...")
        
        # Step 1: Original grayscale
        self.debug_images['gray'] = img_gray.copy()
        self.debug_log(f"Grayscale conversion complete")
        
        # Step 2: Apply different thresholds
        self.debug_log("Trying different threshold methods...")
        
        # OTSU threshold
        thresh_val, binary_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.debug_images['binary_otsu'] = binary_otsu
        self.debug_log(f"OTSU threshold value: {thresh_val}")
        
        # Manual thresholds
        _, binary_120 = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)
        self.debug_images['binary_120'] = binary_120
        
        _, binary_150 = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
        self.debug_images['binary_150'] = binary_150
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        self.debug_images['adaptive'] = adaptive
        
        # Show processed image (using OTSU for now)
        self.show_image_on_canvas(binary_otsu, self.processed_canvas, "processed")
        
        self.debug_log("✅ Image processing complete")
    
    def debug_find_regions(self, img_gray):
        """Debug character region detection"""
        self.debug_log("🔍 Finding character regions...")
        
        # Use OTSU binary image
        binary = self.debug_images['binary_otsu']
        
        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.debug_log(f"Found {len(contours)} contours")
        
        regions = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            self.debug_log(f"Contour {i}: pos=({x},{y}), size=({w}x{h}), area={area}")
            
            # Check if it meets size criteria
            min_width, max_width = 5, 80  # More lenient
            min_height, max_height = 8, 80  # More lenient
            min_area = 20
            
            size_ok = min_width <= w <= max_width and min_height <= h <= max_height
            area_ok = area >= min_area
            
            if size_ok and area_ok:
                regions.append({
                    'id': i,
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'area': area,
                    'labeled': False,
                    'label': None
                })
                self.debug_log(f"✅ Region {i} accepted")
            else:
                self.debug_log(f"❌ Region {i} rejected - size_ok:{size_ok}, area_ok:{area_ok}")
        
        # Sort by x position
        regions.sort(key=lambda r: r['bbox'][0])
        
        self.debug_log(f"✅ Found {len(regions)} valid character regions")
        return regions
    
    def show_detected_regions(self, img_color):
        """Show image with detected regions highlighted"""
        self.debug_log("🎨 Drawing detected regions...")
        
        # Create copy for drawing
        display_img = img_color.copy()
        
        # Draw all detected regions
        for i, region in enumerate(self.character_regions):
            x, y, w, h = region['bbox']
            
            # Draw rectangle
            color = (0, 255, 0) if region['labeled'] else (0, 0, 255)  # Green if labeled, Red if not
            cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)
            
            # Draw region ID
            cv2.putText(display_img, str(i), (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw dimensions
            cv2.putText(display_img, f"{w}x{h}", (x, y + h + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Show on canvas
        self.show_image_on_canvas(display_img, self.regions_canvas, "regions")
        
        self.debug_log(f"✅ Displayed {len(self.character_regions)} regions")
    
    def show_image_on_canvas(self, img, canvas, img_type):
        """Display image on specified canvas"""
        try:
            # Convert BGR to RGB if needed
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Scale up for better visibility
            scale_factor = 6
            h, w = img_rgb.shape[:2]
            img_scaled = cv2.resize(img_rgb, (w * scale_factor, h * scale_factor), 
                                  interpolation=cv2.INTER_NEAREST)
            
            # Convert to PIL and display
            pil_img = Image.fromarray(img_scaled)
            photo = ImageTk.PhotoImage(pil_img)
            
            # Clear canvas and add image
            canvas.delete("all")
            canvas.create_image(0, 0, anchor='nw', image=photo)
            canvas.image = photo  # Keep reference
            
            # Update scroll region
            canvas.configure(scrollregion=canvas.bbox("all"))
            
            self.debug_log(f"✅ Displayed {img_type} image on canvas")
            
        except Exception as e:
            self.debug_log(f"❌ Failed to display {img_type} image: {e}")
    
    def on_image_click(self, event):
        """Handle click on image"""
        if not self.character_regions:
            self.debug_log("❌ No character regions to click on")
            messagebox.showinfo("Info", "No character regions detected. Take a screenshot and analyze first.")
            return
        
        # Convert click coordinates to original image coordinates
        scale_factor = 6
        click_x = event.x // scale_factor
        click_y = event.y // scale_factor
        
        self.debug_log(f"🖱️ Click at: ({event.x}, {event.y}) -> original: ({click_x}, {click_y})")
        
        # Find clicked region
        clicked_region = None
        for region in self.character_regions:
            x, y, w, h = region['bbox']
            if x <= click_x <= x + w and y <= click_y <= y + h:
                clicked_region = region
                break
        
        if clicked_region:
            self.selected_region = clicked_region
            self.debug_log(f"✅ Selected region {clicked_region['id']} at {clicked_region['bbox']}")
            self.status_label.config(text=f"Selected region {clicked_region['id']} - Choose label below")
            
            # Highlight selected region
            self.highlight_selected_region()
        else:
            self.debug_log(f"❌ No region found at click position ({click_x}, {click_y})")
            self.status_label.config(text="No region at that position - try clicking directly on red rectangles")
    
    def highlight_selected_region(self):
        """Highlight the selected region"""
        if not self.selected_region or not self.current_screenshot:
            return
        
        try:
            # Reload image
            img_color = cv2.imread(self.current_screenshot)
            display_img = img_color.copy()
            
            # Draw all regions
            for region in self.character_regions:
                x, y, w, h = region['bbox']
                
                if region == self.selected_region:
                    # Highlight selected region
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 255, 0), 3)  # Yellow
                    cv2.putText(display_img, "SELECTED", (x, y - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                else:
                    # Normal regions
                    color = (0, 255, 0) if region['labeled'] else (0, 0, 255)
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)
                
                cv2.putText(display_img, str(region['id']), (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update display
            self.show_image_on_canvas(display_img, self.regions_canvas, "regions")
            
        except Exception as e:
            self.debug_log(f"❌ Failed to highlight region: {e}")
    
    def label_selected_region(self, label):
        """Label the selected region"""
        if not self.selected_region:
            self.debug_log("❌ No region selected for labeling")
            messagebox.showwarning("Warning", "Please click on a character region first")
            return
        
        self.selected_region['labeled'] = True
        self.selected_region['label'] = label
        
        self.debug_log(f"✅ Labeled region {self.selected_region['id']} as '{label}'")
        
        # Save character image if not skip
        if label != 'skip':
            self.save_character_region(self.selected_region, label)
        
        # Update display
        self.show_detected_regions(cv2.imread(self.current_screenshot))
        
        self.selected_region = None
        self.status_label.config(text=f"Labeled as '{label}' - Click next region")
    
    def save_character_region(self, region, label):
        """Save character region as individual image"""
        try:
            # Load original image
            img = cv2.imread(self.current_screenshot, cv2.IMREAD_GRAYSCALE)
            
            # Extract region
            x, y, w, h = region['bbox']
            char_img = img[y:y+h, x:x+w]
            
            # Resize to 32x32
            char_img_resized = cv2.resize(char_img, (32, 32))
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{label}_{timestamp}.png"
            filepath = f"{self.characters_dir}/{filename}"
            
            cv2.imwrite(filepath, char_img_resized)
            
            self.debug_log(f"💾 Saved character: {filename}")
            
        except Exception as e:
            self.debug_log(f"❌ Failed to save character: {e}")
    
    def save_debug_images(self):
        """Save all debug images for inspection"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for img_name, img_data in self.debug_images.items():
                filename = f"debug_{img_name}_{timestamp}.png"
                filepath = f"{self.debug_dir}/{filename}"
                cv2.imwrite(filepath, img_data)
                self.debug_log(f"💾 Saved debug image: {filename}")
            
            self.debug_log(f"✅ All debug images saved to {self.debug_dir}")
            messagebox.showinfo("Success", f"Debug images saved to {self.debug_dir}")
            
        except Exception as e:
            self.debug_log(f"❌ Failed to save debug images: {e}")
    
    def run(self):
        """Start the debug application"""
        if not TKINTER_AVAILABLE:
            print("❌ Cannot run GUI application")
            return
        
        print("🐛 Debug Roblox Character Region Detector")
        print("=" * 50)
        print("Instructions:")
        print("1. Click 'Take Screenshot' to capture Roblox math problem")
        print("2. Check the different tabs to see processing steps")
        print("3. Look at 'Debug Info' tab for detailed analysis")
        print("4. Click on red rectangles in 'Detected Regions' tab")
        print("5. Use buttons below to label characters")
        print("=" * 50)
        
        self.root.mainloop()

if __name__ == "__main__":
    app = DebugRobloxLabeler()
    app.run()