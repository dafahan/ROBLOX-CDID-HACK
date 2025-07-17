"""
ENHANCED ROBLOX CHARACTER LABELER WITH REVIEW AND DELETE FUNCTIONALITY

Improvements:
- Fixed screenshot loading on startup
- Added screenshot review functionality  
- Added delete buttons for screenshots and individual character labels
- Better navigation and management
- Improved error handling
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
    TKINTER_AVAILABLE = False

class EnhancedRobloxLabeler:
    def __init__(self):
        if not TKINTER_AVAILABLE:
            print("❌ Cannot start GUI")
            return
            
        # Configuration
        self.scan_region = {
            'x': 792,
            'y': 484, 
            'width': 143,
            'height': 50
        }
        
        # Data storage
        self.data_dir = "roblox_training_data"
        self.screenshots_dir = f"{self.data_dir}/screenshots"
        self.characters_dir = f"{self.data_dir}/characters"
        
        for dir_path in [self.data_dir, self.screenshots_dir, self.characters_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Character classes
        self.character_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-']
        
        # Tracking
        self.captured_screenshots = []
        self.labeled_characters = {}
        self.current_screenshot = None
        self.current_screenshot_index = -1
        self.character_regions = []
        self.selected_region = None
        
        # Load existing data
        self.load_existing_data()
        
        # Setup UI
        self.setup_ui()
        
        # Auto-load first screenshot if available
        if self.captured_screenshots:
            self.load_screenshot_by_index(0)
    
    def load_existing_data(self):
        """Load existing screenshots and labels"""
        print("🔍 Loading existing data...")
        
        # Load existing screenshots
        if os.path.exists(self.screenshots_dir):
            screenshot_files = [f for f in os.listdir(self.screenshots_dir) if f.endswith('.png')]
            screenshot_files.sort(reverse=True)  # Most recent first
            
            self.captured_screenshots = [os.path.join(self.screenshots_dir, f) for f in screenshot_files]
            print(f"✅ Found {len(self.captured_screenshots)} existing screenshots")
        
        # Load existing labels
        labels_file = f"{self.data_dir}/labels.json"
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                self.labeled_characters = json.load(f)
            print(f"✅ Loaded {len(self.labeled_characters)} existing labels")
    
    def setup_ui(self):
        """Setup the enhanced interface"""
        self.root = tk.Tk()
        self.root.title("Enhanced Roblox Character Labeler with Review & Delete")
        self.root.geometry("1200x800")
        
        # Main control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Screenshot controls
        screenshot_frame = ttk.LabelFrame(control_frame, text="Screenshot Management")
        screenshot_frame.pack(fill='x', pady=2)
        
        ttk.Button(screenshot_frame, text="📸 Take New", 
                  command=self.take_screenshot).pack(side='left', padx=2)
        
        ttk.Button(screenshot_frame, text="📁 Load File", 
                  command=self.load_screenshot_dialog).pack(side='left', padx=2)
        
        ttk.Button(screenshot_frame, text="⏮️ Previous", 
                  command=self.load_previous_screenshot).pack(side='left', padx=2)
        
        ttk.Button(screenshot_frame, text="⏭️ Next", 
                  command=self.load_next_screenshot).pack(side='left', padx=2)
        
        ttk.Button(screenshot_frame, text="🗑️ Delete Current", 
                  command=self.delete_current_screenshot).pack(side='left', padx=2)
        
        ttk.Button(screenshot_frame, text="📋 Review All", 
                  command=self.open_review_window).pack(side='left', padx=2)
        
        # Screenshot info
        self.screenshot_info = ttk.Label(screenshot_frame, text="No screenshot loaded")
        self.screenshot_info.pack(side='right', padx=5)
        
        # Detection controls
        detection_frame = ttk.LabelFrame(control_frame, text="Character Detection")
        detection_frame.pack(fill='x', pady=2)
        
        ttk.Button(detection_frame, text="🔄 Detect Characters", 
                  command=self.detect_characters).pack(side='left', padx=2)
        
        ttk.Button(detection_frame, text="🧹 Clear Regions", 
                  command=self.clear_regions).pack(side='left', padx=2)
        
        ttk.Button(detection_frame, text="💾 Save Labels", 
                  command=self.save_labels).pack(side='left', padx=2)
        
        ttk.Button(detection_frame, text="🚀 Export Dataset", 
                  command=self.export_dataset).pack(side='left', padx=2)
        
        self.status_label = ttk.Label(detection_frame, text="Ready")
        self.status_label.pack(side='right', padx=5)
        
        # Image display with scrollbars
        image_frame = ttk.Frame(self.root)
        image_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(image_frame, bg='white')
        scroll_v = ttk.Scrollbar(image_frame, orient='vertical', command=self.canvas.yview)
        scroll_h = ttk.Scrollbar(image_frame, orient='horizontal', command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=scroll_v.set, xscrollcommand=scroll_h.set)
        
        scroll_v.pack(side='right', fill='y')
        scroll_h.pack(side='bottom', fill='x')
        self.canvas.pack(side='left', fill='both', expand=True)
        
        # Labeling controls
        self.create_label_buttons()
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_image_click)
        self.root.bind("<Left>", lambda e: self.load_previous_screenshot())
        self.root.bind("<Right>", lambda e: self.load_next_screenshot())
        self.root.bind("<Delete>", lambda e: self.delete_selected_region())
    
    def create_label_buttons(self):
        """Create labeling buttons"""
        label_frame = ttk.LabelFrame(self.root, text="Character Labeling")
        label_frame.pack(fill='x', padx=10, pady=5)
        
        instruction_label = ttk.Label(label_frame, 
            text="Click character regions above, then select label. Use arrow keys to navigate, Delete to remove selected region.")
        instruction_label.pack(pady=2)
        
        button_frame = ttk.Frame(label_frame)
        button_frame.pack(pady=2)
        
        # Numbers
        for i in range(10):
            btn = ttk.Button(button_frame, text=str(i), width=3,
                           command=lambda x=str(i): self.label_region(x))
            btn.pack(side='left', padx=1)
        
        # Operators
        for op in ['+', '-']:
            btn = ttk.Button(button_frame, text=op, width=3,
                           command=lambda x=op: self.label_region(x))
            btn.pack(side='left', padx=1)
        
        # Special actions
        action_frame = ttk.Frame(label_frame)
        action_frame.pack(pady=2)
        
        ttk.Button(action_frame, text="❌ Skip", width=8,
                  command=lambda: self.label_region('skip')).pack(side='left', padx=2)
        
        ttk.Button(action_frame, text="🗑️ Delete Region", width=12,
                  command=self.delete_selected_region).pack(side='left', padx=2)
        
        ttk.Button(action_frame, text="🔍 Show Region Info", width=15,
                  command=self.show_region_info).pack(side='left', padx=2)
        
        # Progress and statistics
        stats_frame = ttk.Frame(label_frame)
        stats_frame.pack(pady=2)
        
        self.progress_label = ttk.Label(stats_frame, text="Progress: 0 characters labeled")
        self.progress_label.pack(side='left')
        
        self.region_info_label = ttk.Label(stats_frame, text="No region selected")
        self.region_info_label.pack(side='right')
    
    def load_screenshot_by_index(self, index):
        """Load screenshot by index in the list"""
        if not self.captured_screenshots or index < 0 or index >= len(self.captured_screenshots):
            return False
        
        try:
            self.current_screenshot = self.captured_screenshots[index]
            self.current_screenshot_index = index
            
            # Update info display
            filename = os.path.basename(self.current_screenshot)
            self.screenshot_info.config(
                text=f"{index + 1}/{len(self.captured_screenshots)}: {filename}")
            
            # Auto-detect characters
            self.detect_characters()
            
            print(f"📸 Loaded screenshot {index + 1}/{len(self.captured_screenshots)}: {filename}")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load screenshot: {e}")
            return False
    
    def load_screenshot_dialog(self):
        """Load screenshot via file dialog"""
        filepath = filedialog.askopenfilename(
            initialdir=self.screenshots_dir,
            title="Select Screenshot to Load",
            filetypes=[("PNG files", "*.png"), ("All images", "*.png *.jpg *.jpeg")]
        )
        if filepath:
            # Add to list if not already present
            if filepath not in self.captured_screenshots:
                self.captured_screenshots.append(filepath)
            
            index = self.captured_screenshots.index(filepath)
            self.load_screenshot_by_index(index)
    
    def load_previous_screenshot(self):
        """Load previous screenshot"""
        if not self.captured_screenshots:
            messagebox.showinfo("Info", "No screenshots available")
            return
        
        if self.current_screenshot_index <= 0:
            # Go to last
            self.load_screenshot_by_index(len(self.captured_screenshots) - 1)
        else:
            self.load_screenshot_by_index(self.current_screenshot_index - 1)
    
    def load_next_screenshot(self):
        """Load next screenshot"""
        if not self.captured_screenshots:
            messagebox.showinfo("Info", "No screenshots available")
            return
        
        if self.current_screenshot_index >= len(self.captured_screenshots) - 1:
            # Go to first
            self.load_screenshot_by_index(0)
        else:
            self.load_screenshot_by_index(self.current_screenshot_index + 1)
    
    def delete_current_screenshot(self):
        """Delete current screenshot"""
        if not self.current_screenshot:
            messagebox.showwarning("Warning", "No screenshot to delete")
            return
        
        filename = os.path.basename(self.current_screenshot)
        result = messagebox.askyesno("Confirm Delete", 
            f"Delete screenshot '{filename}'?\nThis cannot be undone.")
        
        if result:
            try:
                # Remove file
                os.remove(self.current_screenshot)
                
                # Remove from list
                deleted_index = self.current_screenshot_index
                self.captured_screenshots.remove(self.current_screenshot)
                
                print(f"🗑️ Deleted screenshot: {filename}")
                
                # Load next screenshot or clear if none left
                if self.captured_screenshots:
                    # Load screenshot at same index, or previous if at end
                    new_index = min(deleted_index, len(self.captured_screenshots) - 1)
                    self.load_screenshot_by_index(new_index)
                else:
                    self.current_screenshot = None
                    self.current_screenshot_index = -1
                    self.character_regions = []
                    self.canvas.delete("all")
                    self.screenshot_info.config(text="No screenshots")
                    self.status_label.config(text="All screenshots deleted")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete screenshot: {e}")
    
    def clear_regions(self):
        """Clear all detected regions"""
        self.character_regions = []
        self.selected_region = None
        if self.current_screenshot:
            self.display_image_only()
        self.status_label.config(text="Regions cleared")
    
    def delete_selected_region(self):
        """Delete selected region"""
        if not self.selected_region:
            messagebox.showwarning("Warning", "No region selected")
            return
        
        region_id = self.selected_region['id']
        
        # Remove from regions list
        self.character_regions = [r for r in self.character_regions if r['id'] != region_id]
        
        # Renumber remaining regions
        for i, region in enumerate(self.character_regions):
            region['id'] = i
        
        self.selected_region = None
        self.display_regions()
        
        self.status_label.config(text=f"Deleted region {region_id}")
        print(f"🗑️ Deleted region {region_id}")
    
    def show_region_info(self):
        """Show detailed info about selected region"""
        if not self.selected_region:
            messagebox.showinfo("Info", "No region selected")
            return
        
        region = self.selected_region
        x, y, w, h = region['bbox']
        
        info = f"""Region {region['id']} Information:
        
Position: ({x}, {y})
Size: {w} x {h} pixels
Area: {region.get('area', w*h)} pixels
Detection Method: {region.get('method', 'Unknown')}
Labeled: {'Yes' if region['labeled'] else 'No'}
Label: {region.get('label', 'None')}
        """
        
        messagebox.showinfo("Region Information", info)
    
    def open_review_window(self):
        """Open review window to see all screenshots and their labels"""
        if not self.captured_screenshots:
            messagebox.showinfo("Info", "No screenshots to review")
            return
        
        # Create review window
        review_window = tk.Toplevel(self.root)
        review_window.title("Screenshot Review")
        review_window.geometry("800x600")
        
        # Create listbox with scrollbar
        list_frame = ttk.Frame(review_window)
        list_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=('Courier', 10))
        listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Populate listbox
        for i, screenshot_path in enumerate(self.captured_screenshots):
            filename = os.path.basename(screenshot_path)
            
            # Count labels for this screenshot
            # Note: This is a simplified count - in a real app you'd track which labels belong to which screenshot
            status = "✅ Current" if screenshot_path == self.current_screenshot else "📷"
            
            listbox.insert(tk.END, f"{i+1:3d}. {status} {filename}")
        
        # Buttons
        button_frame = ttk.Frame(review_window)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        def load_selected():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                self.load_screenshot_by_index(index)
                review_window.focus_set()  # Keep review window focused
        
        def delete_selected():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                filename = os.path.basename(self.captured_screenshots[index])
                
                result = messagebox.askyesno("Confirm Delete", 
                    f"Delete screenshot '{filename}'?", parent=review_window)
                
                if result:
                    try:
                        # Delete file
                        os.remove(self.captured_screenshots[index])
                        
                        # Update current screenshot if needed
                        if self.captured_screenshots[index] == self.current_screenshot:
                            if len(self.captured_screenshots) > 1:
                                new_index = 0 if index == 0 else index - 1
                                self.load_screenshot_by_index(new_index)
                            else:
                                self.current_screenshot = None
                                self.current_screenshot_index = -1
                                self.character_regions = []
                                self.canvas.delete("all")
                        
                        # Remove from list
                        del self.captured_screenshots[index]
                        
                        # Update listbox
                        listbox.delete(index)
                        
                        # Renumber listbox items
                        listbox.delete(0, tk.END)
                        for i, screenshot_path in enumerate(self.captured_screenshots):
                            filename = os.path.basename(screenshot_path)
                            status = "✅ Current" if screenshot_path == self.current_screenshot else "📷"
                            listbox.insert(tk.END, f"{i+1:3d}. {status} {filename}")
                        
                        print(f"🗑️ Deleted screenshot: {filename}")
                        
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to delete: {e}", parent=review_window)
        
        ttk.Button(button_frame, text="📂 Load Selected", 
                  command=load_selected).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="🗑️ Delete Selected", 
                  command=delete_selected).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="✅ Close", 
                  command=review_window.destroy).pack(side='right', padx=5)
        
        # Info label
        info_label = ttk.Label(button_frame, 
            text=f"Total: {len(self.captured_screenshots)} screenshots, {len(self.labeled_characters)} characters labeled")
        info_label.pack(side='left', padx=20)
    
    def take_screenshot(self):
        """Take screenshot"""
        try:
            screenshot = pyautogui.screenshot(region=(
                self.scan_region['x'], self.scan_region['y'],
                self.scan_region['width'], self.scan_region['height']
            ))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_{timestamp}.png"
            filepath = f"{self.screenshots_dir}/{filename}"
            screenshot.save(filepath)
            
            # Add to beginning of list (most recent first)
            self.captured_screenshots.insert(0, filepath)
            
            # Load the new screenshot
            self.load_screenshot_by_index(0)
            
            self.status_label.config(text=f"Screenshot taken: {filename}")
            print(f"📸 New screenshot: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Screenshot failed: {e}")
    
    def detect_characters(self):
        """Detect characters using improved method"""
        if not self.current_screenshot:
            messagebox.showwarning("Warning", "Load a screenshot first")
            return
        
        try:
            print("🔍 Starting character detection...")
            
            # Load image
            img = cv2.imread(self.current_screenshot, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise Exception("Failed to load image")
                
            print(f"Image shape: {img.shape}, range: {img.min()}-{img.max()}")
            
            # Multiple detection methods
            methods = [
                ("Normal OTSU", cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                ("Inverted OTSU", cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
                ("Adaptive", cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                 cv2.THRESH_BINARY_INV, 11, 2))
            ]
            
            best_regions = []
            best_method = None
            
            for method_name, binary_img in methods:
                print(f"Trying {method_name}...")
                regions = self.find_regions_in_binary(binary_img, method_name)
                
                if len(regions) > len(best_regions):
                    best_regions = regions
                    best_method = method_name
                    print(f"✅ {method_name} found {len(regions)} regions (new best)")
            
            # Fallback to character segmentation
            if len(best_regions) < 2:
                print("Trying character segmentation...")
                best_regions = self.segment_characters(img)
                best_method = "Character Segmentation"
            
            self.character_regions = best_regions
            print(f"Final result: {len(self.character_regions)} regions using {best_method}")
            
            # Display result
            self.display_regions()
            
            self.status_label.config(text=f"Found {len(self.character_regions)} regions using {best_method}")
            
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            messagebox.showerror("Error", f"Detection failed: {e}")
    
    def find_regions_in_binary(self, binary_img, method_name):
        """Find character regions in binary image"""
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Size filtering
            if (3 <= w <= 40 and 8 <= h <= 45 and area >= 15 and 0.2 <= w/h <= 5.0):
                regions.append({
                    'id': len(regions),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'method': method_name,
                    'labeled': False,
                    'label': None
                })
        
        # Sort by x position
        regions.sort(key=lambda r: r['bbox'][0])
        return regions
    
    def segment_characters(self, img):
        """Segment characters by analyzing spacing"""
        height, width = img.shape
        vertical_projection = np.sum(img < 128, axis=0)
        threshold = max(2, vertical_projection.max() * 0.1)
        
        in_character = False
        char_start = 0
        regions = []
        
        for x in range(width):
            if vertical_projection[x] >= threshold and not in_character:
                char_start = x
                in_character = True
            elif vertical_projection[x] < threshold and in_character:
                char_width = x - char_start
                
                if char_width >= 5:
                    char_segment = img[:, char_start:x]
                    horizontal_projection = np.sum(char_segment < 128, axis=1)
                    nonzero_rows = np.where(horizontal_projection > 0)[0]
                    
                    if len(nonzero_rows) > 0:
                        char_top = nonzero_rows[0]
                        char_height = nonzero_rows[-1] + 1 - char_top
                        
                        if char_height >= 8:
                            regions.append({
                                'id': len(regions),
                                'bbox': (char_start, char_top, char_width, char_height),
                                'area': char_width * char_height,
                                'method': 'Segmentation',
                                'labeled': False,
                                'label': None
                            })
                
                in_character = False
        
        return regions
    
    def display_image_only(self):
        """Display image without regions"""
        if not self.current_screenshot:
            return
        
        try:
            img = cv2.imread(self.current_screenshot)
            if img is None:
                return
            
            # Scale up for visibility
            scale = 8
            h, w = img.shape[:2]
            display_img = cv2.resize(img, (w * scale, h * scale), 
                                   interpolation=cv2.INTER_NEAREST)
            
            # Convert and display
            img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            photo = ImageTk.PhotoImage(pil_img)
            
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor='nw', image=photo)
            self.canvas.image = photo
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def display_regions(self):
        """Display image with detected regions"""
        if not self.current_screenshot:
            return
        
        try:
            img = cv2.imread(self.current_screenshot)
            if img is None:
                return
                
            display_img = img.copy()
            
            # Draw regions
            for region in self.character_regions:
                x, y, w, h = region['bbox']
                
                # Color based on status
                if region['labeled']:
                    color = (0, 255, 0)  # Green
                    thickness = 3
                else:
                    color = (0, 0, 255)  # Red
                    thickness = 2
                
                # Draw rectangle
                cv2.rectangle(display_img, (x, y), (x + w, y + h), color, thickness)
                
                # Draw ID
                cv2.putText(display_img, str(region['id']), (x, y - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw label if available
                if region['label'] and region['label'] != 'skip':
                    cv2.putText(display_img, region['label'], (x, y + h + 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # Scale up for visibility
            scale = 8
            h, w = display_img.shape[:2]
            display_img = cv2.resize(display_img, (w * scale, h * scale), 
                                   interpolation=cv2.INTER_NEAREST)
            
            # Convert and display
            img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            photo = ImageTk.PhotoImage(pil_img)
            
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor='nw', image=photo)
            self.canvas.image = photo
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            # Highlight selected region
            if self.selected_region:
                x, y, w, h = self.selected_region['bbox']
                scale = 8
                canvas_x1, canvas_y1 = x * scale, y * scale
                canvas_x2, canvas_y2 = (x + w) * scale, (y + h) * scale
                self.canvas.create_rectangle(canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                                           outline='yellow', width=4, tags="highlight")
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def on_image_click(self, event):
        """Handle image click"""
        if not self.character_regions:
            messagebox.showinfo("Info", "No regions detected. Detect characters first.")
            return
        
        # Convert click coordinates 
        scale = 8
        click_x = event.x // scale
        click_y = event.y // scale
        
        # Find clicked region
        for region in self.character_regions:
            x, y, w, h = region['bbox']
            if x <= click_x <= x + w and y <= click_y <= y + h:
                self.selected_region = region
                self.status_label.config(text=f"Selected region {region['id']} - Choose label")
                self.region_info_label.config(text=f"Region {region['id']}: {w}x{h} at ({x},{y})")
                
                # Redraw with highlight
                self.display_regions()
                return
        
        # No region clicked
        self.selected_region = None
        self.region_info_label.config(text="No region selected")
        self.status_label.config(text="No region at that position")
        self.display_regions()  # Remove highlight
    
    def label_region(self, label):
        """Label selected region"""
        if not self.selected_region:
            messagebox.showwarning("Warning", "Select a region first")
            return
        
        self.selected_region['labeled'] = True
        self.selected_region['label'] = label
        
        print(f"✅ Labeled region {self.selected_region['id']} as '{label}'")
        
        # Save character if not skip
        if label != 'skip':
            self.save_character(self.selected_region, label)
        
        # Update display
        self.display_regions()
        
        # Update progress
        labeled_count = sum(1 for r in self.character_regions if r['labeled'] and r['label'] != 'skip')
        total_count = len(self.labeled_characters) + labeled_count
        self.progress_label.config(text=f"Progress: {total_count} characters labeled")
        
        self.selected_region = None
        self.region_info_label.config(text="No region selected")
        self.status_label.config(text=f"Labeled as '{label}' - Select next region")
    
    def save_character(self, region, label):
        """Save character image"""
        try:
            # Load original
            img = cv2.imread(self.current_screenshot, cv2.IMREAD_GRAYSCALE)
            
            # Extract region
            x, y, w, h = region['bbox']
            char_img = img[y:y+h, x:x+w]
            
            # Resize to 32x32
            char_img = cv2.resize(char_img, (32, 32))
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{label}_{timestamp}.png"
            filepath = f"{self.characters_dir}/{filename}"
            
            cv2.imwrite(filepath, char_img)
            
            # Store metadata
            self.labeled_characters[filename] = {
                'label': label,
                'bbox': region['bbox'],
                'method': region.get('method', 'unknown'),
                'timestamp': timestamp,
                'screenshot': os.path.basename(self.current_screenshot)
            }
            
            print(f"💾 Saved: {filename}")
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    def save_labels(self):
        """Save labels to file"""
        try:
            labels_file = f"{self.data_dir}/labels.json"
            with open(labels_file, 'w') as f:
                json.dump(self.labeled_characters, f, indent=2)
            
            count = len(self.labeled_characters)
            print(f"💾 Saved {count} labels")
            self.status_label.config(text=f"Saved {count} labels")
            messagebox.showinfo("Success", f"Saved {count} character labels")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save labels: {e}")
    
    def export_dataset(self):
        """Export dataset for training"""
        if not self.labeled_characters:
            messagebox.showwarning("Warning", "No labeled data to export")
            return
        
        try:
            # Collect data
            images, labels = [], []
            label_to_idx = {char: idx for idx, char in enumerate(self.character_classes)}
            skipped_count = 0
            
            for filename, data in self.labeled_characters.items():
                if data['label'] in self.character_classes:
                    img_path = f"{self.characters_dir}/{filename}"
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img_norm = img.astype(np.float32) / 255.0
                            images.append(img_norm)
                            labels.append(label_to_idx[data['label']])
                        else:
                            skipped_count += 1
                            print(f"⚠️ Skipped corrupted image: {filename}")
                    else:
                        skipped_count += 1
                        print(f"⚠️ Skipped missing file: {filename}")
                else:
                    skipped_count += 1
                    print(f"⚠️ Skipped unknown label '{data['label']}': {filename}")
            
            if not images:
                messagebox.showwarning("Warning", "No valid images found for export")
                return
            
            # Convert to arrays
            X = np.array(images).reshape(-1, 32, 32, 1)
            y = np.array(labels)
            
            # Save dataset
            dataset_dir = f"{self.data_dir}/dataset"
            os.makedirs(dataset_dir, exist_ok=True)
            
            np.save(f"{dataset_dir}/X_train.npy", X)
            np.save(f"{dataset_dir}/y_train.npy", y)
            
            # Save metadata
            metadata = {
                'character_classes': self.character_classes,
                'num_samples': len(X),
                'image_shape': (32, 32, 1),
                'skipped_files': skipped_count,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(f"{dataset_dir}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            with open(f"{dataset_dir}/characters.json", 'w') as f:
                json.dump(self.character_classes, f)
            
            # Create summary report
            summary = f"""Dataset Export Summary
========================
Total samples exported: {len(X)}
Character classes: {len(self.character_classes)}
Image dimensions: 32x32x1
Skipped files: {skipped_count}

Class distribution:
"""
            for char in self.character_classes:
                count = sum(1 for label in labels if label == label_to_idx[char])
                summary += f"  '{char}': {count} samples\n"
            
            with open(f"{dataset_dir}/export_summary.txt", 'w') as f:
                f.write(summary)
            
            messagebox.showinfo("Export Complete", 
                f"Successfully exported {len(X)} samples to {dataset_dir}\n"
                f"Skipped {skipped_count} invalid files\n\n"
                f"Files created:\n"
                f"- X_train.npy (image data)\n"
                f"- y_train.npy (labels)\n"
                f"- metadata.json (dataset info)\n"
                f"- characters.json (class list)\n"
                f"- export_summary.txt (detailed report)")
            
            print(f"✅ Exported dataset: {len(X)} samples, skipped {skipped_count}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def delete_character_label(self, filename):
        """Delete a specific character label and its file"""
        try:
            # Remove from labeled_characters
            if filename in self.labeled_characters:
                del self.labeled_characters[filename]
            
            # Remove file
            filepath = f"{self.characters_dir}/{filename}"
            if os.path.exists(filepath):
                os.remove(filepath)
            
            print(f"🗑️ Deleted character label: {filename}")
            
            # Update progress
            self.progress_label.config(text=f"Progress: {len(self.labeled_characters)} characters labeled")
            
        except Exception as e:
            print(f"❌ Failed to delete character label: {e}")
    
    def show_character_labels_window(self):
        """Show window with all character labels for review/deletion"""
        if not self.labeled_characters:
            messagebox.showinfo("Info", "No character labels to review")
            return
        
        # Create review window
        labels_window = tk.Toplevel(self.root)
        labels_window.title("Character Labels Review")
        labels_window.geometry("900x600")
        
        # Create treeview for better display
        tree_frame = ttk.Frame(labels_window)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        columns = ('Filename', 'Label', 'Screenshot', 'Method', 'Timestamp')
        tree = ttk.Treeview(tree_frame, columns=columns, show='tree headings')
        
        # Configure columns
        tree.heading('#0', text='#')
        tree.column('#0', width=50)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Populate tree
        for i, (filename, data) in enumerate(self.labeled_characters.items()):
            tree.insert('', 'end', text=str(i+1), values=(
                filename,
                data['label'],
                data.get('screenshot', 'Unknown'),
                data.get('method', 'Unknown'),
                data.get('timestamp', 'Unknown')
            ))
        
        # Buttons
        button_frame = ttk.Frame(labels_window)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        def delete_selected_label():
            selection = tree.selection()
            if not selection:
                messagebox.showwarning("Warning", "Select a label to delete", parent=labels_window)
                return
            
            item = tree.item(selection[0])
            filename = item['values'][0]
            label = item['values'][1]
            
            result = messagebox.askyesno("Confirm Delete", 
                f"Delete character label?\nFile: {filename}\nLabel: {label}", 
                parent=labels_window)
            
            if result:
                self.delete_character_label(filename)
                tree.delete(selection[0])
                
                # Renumber remaining items
                for i, item_id in enumerate(tree.get_children()):
                    tree.item(item_id, text=str(i+1))
        
        def show_character_image():
            selection = tree.selection()
            if not selection:
                messagebox.showwarning("Warning", "Select a character to view", parent=labels_window)
                return
            
            item = tree.item(selection[0])
            filename = item['values'][0]
            filepath = f"{self.characters_dir}/{filename}"
            
            if os.path.exists(filepath):
                # Create image window
                img_window = tk.Toplevel(labels_window)
                img_window.title(f"Character: {filename}")
                
                try:
                    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Scale up for visibility
                        img_scaled = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
                        img_pil = Image.fromarray(img_scaled)
                        photo = ImageTk.PhotoImage(img_pil)
                        
                        label_widget = tk.Label(img_window, image=photo)
                        label_widget.image = photo  # Keep reference
                        label_widget.pack(padx=10, pady=10)
                        
                        info_label = tk.Label(img_window, text=f"Label: {item['values'][1]}\nOriginal size: 32x32")
                        info_label.pack(pady=5)
                    else:
                        tk.Label(img_window, text="Failed to load image").pack(padx=10, pady=10)
                        
                except Exception as e:
                    tk.Label(img_window, text=f"Error: {e}").pack(padx=10, pady=10)
            else:
                messagebox.showerror("Error", f"Character image not found: {filepath}", parent=labels_window)
        
        ttk.Button(button_frame, text="👁️ View Image", 
                  command=show_character_image).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="🗑️ Delete Selected", 
                  command=delete_selected_label).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="✅ Close", 
                  command=labels_window.destroy).pack(side='right', padx=5)
        
        # Info
        info_label = ttk.Label(button_frame, 
            text=f"Total character labels: {len(self.labeled_characters)}")
        info_label.pack(side='left', padx=20)
    
    def run(self):
        """Start application"""
        if not TKINTER_AVAILABLE:
            print("❌ Cannot run GUI - tkinter not available")
            return
        
        print("🚀 Enhanced Roblox Character Labeler")
        print("=" * 60)
        print("New features in this version:")
        print("✅ Fixed screenshot loading on startup")
        print("✅ Screenshot review window with delete functionality")
        print("✅ Individual region deletion")
        print("✅ Character labels review window")
        print("✅ Better navigation with keyboard shortcuts")
        print("✅ Improved error handling and user feedback")
        print()
        print("Controls:")
        print("📸 Take New - captures math problem from screen")
        print("📁 Load File - browse and load existing screenshots")
        print("⏮️⏭️ Previous/Next - navigate screenshots (or use arrow keys)")
        print("🗑️ Delete Current - remove current screenshot")
        print("📋 Review All - see all screenshots in a list")
        print("🔄 Detect Characters - find text regions automatically")
        print("Click red rectangles to select, then use number/operator buttons")
        print("🗑️ Delete Region - remove unwanted detected regions")
        print("Delete key - quick delete selected region")
        
        # Add character labels review button
        review_frame = ttk.LabelFrame(self.root, text="Data Review")
        review_frame.pack(fill='x', padx=10, pady=2)
        
        ttk.Button(review_frame, text="🏷️ Review Character Labels", 
                  command=self.show_character_labels_window).pack(side='left', padx=5)
        
        ttk.Button(review_frame, text="📊 Dataset Statistics", 
                  command=self.show_dataset_statistics).pack(side='left', padx=5)
        
        # Show startup status
        if self.captured_screenshots:
            print(f"📸 Found {len(self.captured_screenshots)} existing screenshots")
            if self.current_screenshot:
                current_name = os.path.basename(self.current_screenshot)
                print(f"📷 Loaded: {current_name}")
        else:
            print("📸 No existing screenshots found")
        
        if self.labeled_characters:
            print(f"🏷️ Found {len(self.labeled_characters)} existing character labels")
        else:
            print("🏷️ No existing character labels found")
        
        print("=" * 60)
        
        self.root.mainloop()
    
    def show_dataset_statistics(self):
        """Show dataset statistics window"""
        if not self.labeled_characters:
            messagebox.showinfo("Info", "No labeled data for statistics")
            return
        
        # Calculate statistics
        label_counts = {}
        method_counts = {}
        screenshot_counts = {}
        
        for filename, data in self.labeled_characters.items():
            label = data['label']
            method = data.get('method', 'Unknown')
            screenshot = data.get('screenshot', 'Unknown')
            
            label_counts[label] = label_counts.get(label, 0) + 1
            method_counts[method] = method_counts.get(method, 0) + 1
            screenshot_counts[screenshot] = screenshot_counts.get(screenshot, 0) + 1
        
        # Create statistics window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Dataset Statistics")
        stats_window.geometry("600x500")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(stats_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Label distribution tab
        label_frame = ttk.Frame(notebook)
        notebook.add(label_frame, text="Label Distribution")
        
        label_text = tk.Text(label_frame, wrap='word', font=('Courier', 10))
        label_scroll = ttk.Scrollbar(label_frame, command=label_text.yview)
        label_text.configure(yscrollcommand=label_scroll.set)
        
        label_text.pack(side='left', fill='both', expand=True)
        label_scroll.pack(side='right', fill='y')
        
        label_stats = f"Label Distribution ({len(self.labeled_characters)} total samples)\n"
        label_stats += "=" * 50 + "\n\n"
        
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            percentage = (count / len(self.labeled_characters)) * 100
            label_stats += f"'{label}': {count:3d} samples ({percentage:5.1f}%)\n"
        
        label_text.insert('1.0', label_stats)
        label_text.configure(state='disabled')
        
        # Method distribution tab
        method_frame = ttk.Frame(notebook)
        notebook.add(method_frame, text="Detection Methods")
        
        method_text = tk.Text(method_frame, wrap='word', font=('Courier', 10))
        method_scroll = ttk.Scrollbar(method_frame, command=method_text.yview)
        method_text.configure(yscrollcommand=method_scroll.set)
        
        method_text.pack(side='left', fill='both', expand=True)
        method_scroll.pack(side='right', fill='y')
        
        method_stats = f"Detection Method Distribution\n"
        method_stats += "=" * 50 + "\n\n"
        
        for method in sorted(method_counts.keys()):
            count = method_counts[method]
            percentage = (count / len(self.labeled_characters)) * 100
            method_stats += f"{method}: {count:3d} samples ({percentage:5.1f}%)\n"
        
        method_text.insert('1.0', method_stats)
        method_text.configure(state='disabled')
        
        # Screenshot distribution tab
        screenshot_frame = ttk.Frame(notebook)
        notebook.add(screenshot_frame, text="Screenshots")
        
        screenshot_text = tk.Text(screenshot_frame, wrap='word', font=('Courier', 10))
        screenshot_scroll = ttk.Scrollbar(screenshot_frame, command=screenshot_text.yview)
        screenshot_text.configure(yscrollcommand=screenshot_scroll.set)
        
        screenshot_text.pack(side='left', fill='both', expand=True)
        screenshot_scroll.pack(side='right', fill='y')
        
        screenshot_stats = f"Characters per Screenshot\n"
        screenshot_stats += "=" * 50 + "\n\n"
        
        for screenshot in sorted(screenshot_counts.keys()):
            count = screenshot_counts[screenshot]
            screenshot_stats += f"{screenshot}: {count} characters\n"
        
        screenshot_text.insert('1.0', screenshot_stats)
        screenshot_text.configure(state='disabled')
        
        # Close button
        ttk.Button(stats_window, text="✅ Close", 
                  command=stats_window.destroy).pack(pady=10)

if __name__ == "__main__":
    app = EnhancedRobloxLabeler()
    app.run()