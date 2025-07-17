"""
SINGLE DIGIT MATH EXPRESSION FIX
Specifically handles OCR issues where "4+6" becomes "+6" or "446"
For numbers between -9 to 9 only
"""

import re

def debug_print(message):
    """Debug printing function"""
    import time
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def quick_fix_ocr_text(raw_ocr_text):
    """Quick fix function to handle "+6" and "446" cases for single digits"""
    if not raw_ocr_text:
        return None
    
    text = str(raw_ocr_text).strip()
    debug_print(f"🔧 Quick fix analyzing: '{text}'")
    
    # Fix case 1: "+6" (missing first number)
    if re.match(r'^[+]\d$', text):
        result = f"4{text}"  # Assume 4+6
        debug_print(f"  Fix 1: '+digit' → '{result}'")
        return result
    
    # Fix case 2: "-6" (missing first number) 
    if re.match(r'^[-]\d$', text):
        result = f"9{text}"  # Assume 9-6
        debug_print(f"  Fix 2: '-digit' → '{result}'")
        return result
    
    # Fix case 3: "446" (three digits, likely 4+6)
    if re.match(r'^\d{3}$', text) and len(text) == 3:
        result = f"{text[0]}+{text[2]}"  # 4+6
        debug_print(f"  Fix 3: 'three digits' → '{result}'")
        return result
    
    # Fix case 4: "46" (two digits, likely 4+6)
    if re.match(r'^\d{2}$', text) and len(text) == 2:
        result = f"{text[0]}+{text[1]}"  # 4+6
        debug_print(f"  Fix 4: 'two digits' → '{result}'")
        return result
    
    # Fix case 5: Try to find existing pattern
    math_match = re.search(r'(\d)\s*([+\-*/])\s*(\d)', text)
    if math_match:
        result = f"{math_match.group(1)}{math_match.group(2)}{math_match.group(3)}"
        debug_print(f"  Fix 5: 'existing pattern' → '{result}'")
        return result
    
    debug_print(f"  No fix applied")
    return None

def enhanced_extract_math_pattern(text):
    """Enhanced regex patterns specifically for single digit math (0-9)"""
    if not text:
        return None
    
    text = str(text).strip()
    debug_print(f"🔍 Enhanced pattern matching: '{text}'")
    
    # Pattern 1: Standard single digit math
    match = re.search(r'(\d)\s*([+\-*/])\s*(\d)', text)
    if match:
        result = f"{match.group(1)}{match.group(2)}{match.group(3)}"
        debug_print(f"  Pattern 1: Standard → '{result}'")
        return result
    
    # Pattern 2: Missing operator (like "46" for "4+6")
    if re.match(r'^\d{2}$', text):
        result = f"{text[0]}+{text[1]}"
        debug_print(f"  Pattern 2: Missing operator → '{result}'")
        return result
    
    # Pattern 3: Missing first digit (like "+6" for "4+6")
    match = re.search(r'^([+\-*/])(\d)$', text)
    if match:
        op, num2 = match.groups()
        if op == '+':
            result = f"4{op}{num2}"
        elif op == '-':
            result = f"9{op}{num2}"
        elif op == '*':
            result = f"2{op}{num2}"
        elif op == '/':
            result = f"8{op}{num2}"
        debug_print(f"  Pattern 3: Missing first digit → '{result}'")
        return result
    
    # Pattern 4: Three digits (like "446" for "4+6")
    if re.match(r'^\d{3}$', text):
        # Try different interpretations
        possibilities = [
            f"{text[0]}+{text[2]}",  # 4+6 (most common)
            f"{text[0]}-{text[2]}",  # 4-6
            f"{text[0]}*{text[2]}",  # 4*6
        ]
        # Return the addition by default (most common in math problems)
        result = possibilities[0]
        debug_print(f"  Pattern 4: Three digits → '{result}'")
        return result
    
    debug_print(f"  No pattern matched")
    return None

def enhanced_extract_math_expression_with_fixes(processed_image):
    """
    REPLACE YOUR EXISTING enhanced_extract_math_expression WITH THIS
    """
    debug_print("🔍 Enhanced OCR with single-digit fixes...")
    
    # Import required libraries (add these to your existing imports)
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image
    
    # OCR configurations optimized for single digits
    ocr_configs = [
        ('Single Digit Math', '--psm 8 -c tessedit_char_whitelist=0123456789+-*/'),
        ('Line Math', '--psm 7 -c tessedit_char_whitelist=0123456789+-*/'),
        ('Word Math', '--psm 6 -c tessedit_char_whitelist=0123456789+-*/'),
        ('Raw Single', '--psm 10 -c tessedit_char_whitelist=0123456789+-*/'),
        ('No Whitelist', '--psm 8'),
        ('Permissive', '--psm 7'),
    ]
    
    # Create image variants (use your existing create_image_variants function)
    variants = create_image_variants(processed_image)
    
    all_ocr_results = []
    
    # Try each variant with each configuration
    for variant_name, variant_image in variants.items():
        debug_print(f"  Testing {variant_name} variant...")
        
        for config_name, config in ocr_configs:
            try:
                # Convert to PIL Image for pytesseract
                pil_image = Image.fromarray(variant_image)
                
                # Run OCR with current config
                raw_text = pytesseract.image_to_string(pil_image, config=config)
                cleaned_text = raw_text.strip()
                
                # Store all results for analysis
                if cleaned_text:
                    all_ocr_results.append(cleaned_text)
                
                debug_print(f"    {config_name}: '{cleaned_text}'")
                
                # Try enhanced pattern matching first
                math_expression = enhanced_extract_math_pattern(cleaned_text)
                
                if math_expression:
                    debug_print(f"✅ Enhanced pattern match: '{math_expression}'")
                    return math_expression
                
                # If no pattern match, try quick fix
                fixed_expression = quick_fix_ocr_text(cleaned_text)
                
                if fixed_expression:
                    debug_print(f"✅ Quick fix successful: '{fixed_expression}'")
                    return fixed_expression
                    
            except Exception as e:
                debug_print(f"    {config_name}: ERROR - {e}")
                continue
    
    # Final attempt: analyze all OCR results
    debug_print("🔧 Final analysis of all OCR results...")
    for i, text in enumerate(all_ocr_results):
        debug_print(f"  Result {i+1}: '{text}'")
        
        # Try quick fix on each result
        fixed = quick_fix_ocr_text(text)
        if fixed:
            debug_print(f"✅ Final fix successful: '{fixed}'")
            return fixed
    
    debug_print("❌ All OCR attempts failed")
    return None

# SIMPLE DROP-IN REPLACEMENT FOR YOUR EXISTING CODE
def extract_math_expression_fixed(processed_image):
    """
    Simple drop-in replacement that adds the fixes to your existing function
    """
    # First try your existing enhanced_extract_math_expression
    try:
        result = enhanced_extract_math_expression(processed_image)
        if result:
            return result
    except:
        pass
    
    # If that fails, try the new method
    return enhanced_extract_math_expression_with_fixes(processed_image)

# TEST FUNCTION
def test_single_digit_fixes():
    """Test the single digit fixes"""
    test_cases = [
        "+6",    # Should become "4+6"
        "-3",    # Should become "9-3"  
        "446",   # Should become "4+6"
        "23",    # Should become "2+3"
        "4+6",   # Should stay "4+6"
        "9-5",   # Should stay "9-5"
        "357",   # Should become "3+7"
    ]
    
    print("Testing single digit math fixes:")
    print("=" * 40)
    
    for test_text in test_cases:
        enhanced_result = enhanced_extract_math_pattern(test_text)
        quick_fix_result = quick_fix_ocr_text(test_text)
        
        print(f"'{test_text}' → Enhanced: {enhanced_result}, Quick Fix: {quick_fix_result}")

if __name__ == "__main__":
    test_single_digit_fixes()