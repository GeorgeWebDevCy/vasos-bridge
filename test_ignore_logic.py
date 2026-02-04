
import unittest
import re

# Logic to be tested (copied/adapted from plan)
def mask_ignored_text(text, ignore_list):
    if not ignore_list:
        return text, {}
    
    # Sort by length descending to handle overlapping phrases (e.g. "Super Man" vs "Man")
    sorted_ignore = sorted(ignore_list, key=len, reverse=True)
    
    mask_map = {}
    masked_text = text
    
    for i, phrase in enumerate(sorted_ignore):
        if not phrase.strip():
            continue
        # Use a placeholder that is unlikely to be translated
        placeholder = f"__IGNORE_{i}__"
        
        # We need to be careful not to double-mask or mask parts of placeholders
        # Simple string replacement might be enough if tokens are unique enough
        # Using regex to ensure we match the exact phrase case-sensitively or insensitively?
        # User requirement implies specific words, assume case-sensitive for controls, 
        # but often people want case-insensitive. Let's stick to literal replacement for now 
        # as it's safer for "code" or specific terms.
        
        if phrase in masked_text:
            mask_map[placeholder] = phrase
            masked_text = masked_text.replace(phrase, placeholder)
            
    return masked_text, mask_map

def unmask_ignored_text(masked_text, mask_map):
    text = masked_text
    # Reverse order doesn't strictly matter if placeholders are unique, 
    # but good practice to allow nested if we ever supported it. 
    # Here placeholders are unique keys.
    for placeholder, original in mask_map.items():
        text = text.replace(placeholder, original)
    return text

class TestIgnoreLogic(unittest.TestCase):
    def test_basic_masking(self):
        text = "Hello world, this is a test."
        ignore = ["world", "test"]
        masked, mapping = mask_ignored_text(text, ignore)
        
        self.assertIn("__IGNORE_", masked)
        self.assertNotIn("world", masked)
        self.assertNotIn("test", masked)
        
        unmasked = unmask_ignored_text(masked, mapping)
        self.assertEqual(unmasked, text)

    def test_overlap(self):
        text = "I like Super Man and Man."
        ignore = ["Man", "Super Man"]
        # Should mask "Super Man" first
        masked, mapping = mask_ignored_text(text, ignore)
        
        self.assertNotIn("Super Man", masked)
        # The second "Man" should also be masked
        self.assertNotIn(" Man.", masked.replace("__IGNORE_", "")) # removing placeholder to check text content
        
        unmasked = unmask_ignored_text(masked, mapping)
        self.assertEqual(unmasked, text)

    def test_empty_list(self):
        text = "Nothing to ignore."
        ignore = []
        masked, mapping = mask_ignored_text(text, ignore)
        self.assertEqual(masked, text)
        self.assertEqual(mapping, {})

if __name__ == '__main__':
    unittest.main()
