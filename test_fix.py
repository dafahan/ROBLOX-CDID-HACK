"""
OCR MATH EXPRESSION DETECTION FIX

This fixes issues with detecting similar-looking characters like:
- 5-5 (might be read as S-S, s-s, or other variations)
- 1-5 (might be read as l-S, I-5, etc.)

IMPROVEMENTS:
- Enhanced character cleaning and normalization
- Multiple OCR attempts with different configurations
- Pattern matching for common misreadings
- Fallback detection methods
"""

import re
import pytesseract
from PIL import Image
import cv2
import numpy as np

# =============================================================================
# CHARACTER NORMALIZATION AND CLEANING
# =============================================================================

def normalize_ocr_text(text):
    """
    Normalize OCR text to fix common character recognition errors
    """
    if not text:
        return ""
    
    # Convert to string and clean
    text = str(text).strip()
    
    # Common OCR character fixes
    char_fixes = {
        # Letters that should be numbers
        'S': '5', 's': '5', 'B': '8', 'o': '0', 'O': '0', 
        'l': '1', 'I': '1', 'i': '1', '|': '1',
        'Z': '2', 'z': '2', 'g': '9', 'G': '6',
        
        # Common operator fixes
        '—': '-', '–': '-', '_': '-', '=': '-',
        'x': '*', 'X': '*', '×': '*',
        '÷': '/', ':': '/',
        
        # Remove unwanted characters
        ' ': '', '\n': '', '\t': '', '\r': '',
        '.': '', ',': '', ';': '', '!': '',
        '(': '', ')': '', '[': ']', '{': '}',
        '"': '', "'": '', '`': '',
    }
    
    # Apply character fixes
    for old_char, new_char in char_fixes.items():
        text = text.replace(old_char, new_char)
    
    return text

def extract_math_pattern(text):
    """
    Extract math patterns using regex, accounting for OCR errors
    """
    if not text:
        return None
    
    # List of possible math patterns (accounting for OCR errors)
    patterns = [
        # Standard patterns
        r'(\d+)\s*([+\-*/])\s*(\d+)',           # 5 - 5
        r'(\d+)\s*([+\-*/])\s*(\d+)',           # 5-5
        
        # Patterns with potential OCR errors
        r'([0-9SsOol|I]+)\s*([+\-*/—–_=xX×÷:])\s*([0-9SsOol|I]+)',  # Mixed chars
        r'([0-9]{1,2})\s*[-—–_=]\s*([0-9]{1,2})',                    # Various dash types
        r'([0-9]{1,2})\s*[+]\s*([0-9]{1,2})',                       # Plus operations
        r'([0-9]{1,2})\s*[*/xX×÷:]\s*([0-9]{1,2})',                 # Multiply/divide
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            if len(match.groups()) == 3:
                # Standard format: num1 op num2
                num1, op, num2 = match.groups()
            elif len(match.groups()) == 2:
                # Assume subtraction if only 2 groups
                num1, num2 = match.groups()
                op = '-'
            else:
                continue
            
            # Normalize the extracted parts
            num1 = normalize_ocr_text(num1)
            num2 = normalize_ocr_text(num2)
            op = normalize_ocr_text(op)
            
            # Validate that we have valid numbers
            try:
                int(num1)
                int(num2)
                if op in ['+', '-', '*', '/']:
                    return f"{num1}{op}{num2}"
            except ValueError:
                continue
    
    return None

def enhanced_extract_math_expression(processed_image):
    """
    Enhanced math expression extraction with multiple OCR attempts
    """
    
    # OCR configurations to try (in order of preference)
    ocr_configs = [
        # Standard config
        '--psm 6 -c tessedit_char_whitelist=0123456789+-*/',
        
        # More permissive configs for difficult cases
        '--psm 7 -c tessedit_char_whitelist=0123456789+-*/SsOolI',
        '--psm 8 -c tessedit_char_whitelist=0123456789+-*/',
        '--psm 13 -c tessedit_char_whitelist=0123456789+-*/',
        
        # Very permissive (allows more characters)
        '--psm 6 -c tessedit_char_whitelist=0123456789+-*/SsOolI—–_=xX×÷:',
        '--psm 7',  # No character restriction
        '--psm 8',  # Single word, no restriction
    ]
    
    debug_results = []
    
    for i, config in enumerate(ocr_configs):
        try:
            # Convert processed image to PIL format
            pil_image = Image.fromarray(processed_image)
            
            # Run OCR with current config
            raw_text = pytesseract.image_to_string(pil_image, config=config)
            
            # Clean and normalize
            cleaned_text = normalize_ocr_text(raw_text)
            
            # Try to extract math pattern
            math_expression = extract_math_pattern(cleaned_text)
            
            debug_results.append({
                'config': f'Config {i+1}',
                'raw': raw_text.strip(),
                'cleaned': cleaned_text,
                'math': math_expression
            })
            
            # If we found a valid math expression, return it
            if math_expression:
                print(f"✅ Math detected with Config {i+1}: '{math_expression}'")
                print(f"   Raw OCR: '{raw_text.strip()}'")
                return math_expression
                
        except Exception as e:
            debug_results.append({
                'config': f'Config {i+1}',
                'error': str(e)
            })
            continue
    
    # If no config worked, show debug info
    print("❌ No math expression detected. Debug info:")
    for result in debug_results:
        if 'error' in result:
            print(f"  {result['config']}: ERROR - {result['error']}")
        else:
            print(f"  {result['config']}: '{result.get('raw', '')}' → '{result.get('cleaned', '')}' → {result.get('math', 'None')}")
    
    return None

# =============================================================================
# IMAGE PREPROCESSING IMPROVEMENTS
# =============================================================================

def create_multiple_image_variants(image):
    """
    Create multiple processed variants of the image to improve OCR success
    """
    # Convert PIL to OpenCV if needed
    if isinstance(image, Image.Image):
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        opencv_image = image
    
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    variants = {}
    
    # Variant 1: Standard threshold
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['standard'] = thresh1
    
    # Variant 2: Inverted (white text on black background)
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants['inverted'] = thresh2
    
    # Variant 3: Higher contrast
    contrast = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
    _, thresh3 = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants['high_contrast'] = thresh3
    
    # Variant 4: Morphological operations
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    variants['morphological'] = morph
    
    # Variant 5: Dilated (thicker text)
    dilated = cv2.dilate(thresh1, kernel, iterations=1)
    variants['dilated'] = dilated
    
    # Variant 6: Eroded (thinner text)
    eroded = cv2.erode(thresh1, kernel, iterations=1)
    variants['eroded'] = eroded
    
    return variants

def multi_variant_ocr_extraction(image):
    """
    Try OCR on multiple image variants
    """
    print("🔍 Trying multiple image variants for OCR...")
    
    variants = create_multiple_image_variants(image)
    
    for variant_name, variant_image in variants.items():
        print(f"  Testing variant: {variant_name}")
        
        try:
            result = enhanced_extract_math_expression(variant_image)
            if result:
                print(f"✅ Success with {variant_name} variant: '{result}'")
                return result
        except Exception as e:
            print(f"  {variant_name} failed: {e}")
            continue
    
    print("❌ All variants failed")
    return None

# =============================================================================
# MANUAL PATTERN DETECTION (FALLBACK)
# =============================================================================

def manual_digit_detection(image):
    """
    Fallback method: Manual digit detection using image analysis
    """
    print("🔍 Trying manual digit detection...")
    
    try:
        # This is a simplified approach - you could implement more sophisticated
        # digit recognition using template matching or machine learning
        
        gray = cv2.cvtColor(np.array(image) if isinstance(image, Image.Image) else image, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count significant contours (might indicate characters)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 10]
        
        print(f"  Found {len(significant_contours)} potential characters")
        
        # If we find 3 contours, it might be "X-Y" pattern
        if len(significant_contours) == 3:
            print("  Pattern suggests: digit-operator-digit")
            # This would need more sophisticated implementation
            return "manual_detection_needed"
        
    except Exception as e:
        print(f"  Manual detection failed: {e}")
    
    return None

# =============================================================================
# MAIN ENHANCED DETECTION FUNCTION
# =============================================================================

def detect_math_expression_robust(image):
    """
    Main function that tries multiple methods to detect math expressions
    """
    print("🎯 Starting robust math expression detection...")
    
    # Method 1: Multi-variant OCR
    result = multi_variant_ocr_extraction(image)
    if result:
        return result
    
    # Method 2: Manual pattern detection (fallback)
    manual_result = manual_digit_detection(image)
    if manual_result and manual_result != "manual_detection_needed":
        return manual_result
    
    print("❌ All detection methods failed")
    return None

# =============================================================================
# TESTING FUNCTION
# =============================================================================

def test_ocr_with_sample():
    """
    Test function to debug OCR issues
    """
    print("🧪 OCR Testing Function")
    print("=" * 40)
    
    # Test cases that commonly fail
    test_cases = [
        "5-5", "1-5", "5 - 5", "1 - 5",
        "S-S", "s-s", "l-S", "I-5",
        "5—5", "5–5", "5_5"
    ]
    
    print("Testing normalization:")
    for test in test_cases:
        normalized = normalize_ocr_text(test)
        pattern = extract_math_pattern(normalized)
        print(f"  '{test}' → '{normalized}' → {pattern}")
    
    print("\n✅ Test complete!")

# =============================================================================
# INTEGRATION WITH EXISTING BOT
# =============================================================================

def replace_in_existing_bot():
    """
    Instructions for integrating this fix into your existing bot
    """
    print("""
    🔧 TO FIX YOUR EXISTING BOT:
    
    Replace your extract_math_expression function with:
    
    def extract_math_expression(processed_image):
        return detect_math_expression_robust(processed_image)
    
    This will use the enhanced detection with multiple methods.
    """)

if __name__ == "__main__":
    test_ocr_with_sample()
    replace_in_existing_bot()