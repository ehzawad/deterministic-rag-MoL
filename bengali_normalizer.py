#!/usr/bin/env python3
"""
Bengali Text Normalization Module

This module provides comprehensive Bengali text normalization to ensure
consistent Unicode representation between FAQ data and input queries.
"""

import unicodedata
import re

class BengaliNormalizer:
    """
    Bengali text normalizer to handle Unicode variations and ensure consistent encoding
    """
    
    def __init__(self):
        # Bengali Unicode character mappings for common variations
        self.char_mappings = {
            # Ya-phala variations (য-ফলা)
            '\u09af\u09bc': '\u09df',  # য় -> য়
            '\u09df': '\u09df',        # য় (keep as is)
            
            # Ra-phala variations (র-ফলা)
            '\u09b0\u09cd': '\u09b0\u09cd',  # র্ (keep as is)
            
            # Common conjunct variations
            '\u0995\u09cd\u09b7': '\u0995\u09cd\u09b7',  # ক্ষ
            '\u099c\u09cd\u099e': '\u099c\u09cd\u099e',  # জ্ঞ
            
            # Vowel diacritic variations
            '\u09c7\u09be': '\u09cb',  # ে + া = ো
            '\u09c7\u09d7': '\u09cb',  # ে + ৗ = ো (alternative)
            
            # Common digit variations (if any)
            '০': '০', '১': '১', '২': '২', '৩': '৩', '৪': '৪',
            '৫': '৫', '৬': '৬', '৭': '৭', '৮': '৮', '৯': '৯',
        }
        
        # Specific problematic character sequences we found
        self.problem_sequences = {
            # The "উন্নয়ন" variation we found in debugging
            'উন্নয়ন': self._normalize_unnayan,
            'দেয়া': self._normalize_deya,
            'নেয়া': self._normalize_neya,
            'হয়ে': self._normalize_hoye,
            'গেয়ে': self._normalize_geye,
            'দেয়': self._normalize_dey,
            'নেয়': self._normalize_ney,
        }
    
    def _normalize_unnayan(self, text):
        """Normalize উন্নয়ন variations"""
        # Handle different representations of উন্নয়ন
        patterns = [
            r'উন্নয়ন',
            r'উন্নযন',
            r'উন্নয্ন',
        ]
        normalized = 'উন্নয়ন'
        for pattern in patterns:
            text = re.sub(pattern, normalized, text)
        return text
    
    def _normalize_deya(self, text):
        """Normalize দেয়া variations"""
        patterns = [r'দেয়া', r'দেওয়া']
        normalized = 'দেয়া'
        for pattern in patterns:
            text = re.sub(pattern, normalized, text)
        return text
    
    def _normalize_neya(self, text):
        """Normalize নেয়া variations"""
        patterns = [r'নেয়া', r'নেওয়া']
        normalized = 'নেয়া'
        for pattern in patterns:
            text = re.sub(pattern, normalized, text)
        return text
    
    def _normalize_hoye(self, text):
        """Normalize হয়ে variations"""
        patterns = [r'হয়ে', r'হওয়ে']
        normalized = 'হয়ে'
        for pattern in patterns:
            text = re.sub(pattern, normalized, text)
        return text
    
    def _normalize_geye(self, text):
        """Normalize গেয়ে variations"""
        patterns = [r'গেয়ে', r'গেওয়ে']
        normalized = 'গেয়ে'
        for pattern in patterns:
            text = re.sub(pattern, normalized, text)
        return text
    
    def _normalize_dey(self, text):
        """Normalize দেয় variations"""
        patterns = [r'দেয়', r'দেও']
        normalized = 'দেয়'
        for pattern in patterns:
            text = re.sub(pattern, normalized, text)
        return text
    
    def _normalize_ney(self, text):
        """Normalize নেয় variations"""
        patterns = [r'নেয়', r'নেও']
        normalized = 'নেয়'
        for pattern in patterns:
            text = re.sub(pattern, normalized, text)
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """
        Apply Unicode normalization to Bengali text
        """
        # Step 1: Unicode NFC normalization (Canonical Decomposition followed by Canonical Composition)
        text = unicodedata.normalize('NFC', text)
        
        # Step 2: Apply character mappings for known variations
        for original, replacement in self.char_mappings.items():
            text = text.replace(original, replacement)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in Bengali text
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Handle common punctuation spacing
        text = re.sub(r'\s*([।?!])\s*', r'\1', text)  # Remove spaces around Bengali punctuation
        text = re.sub(r'\s*([.?!])\s*', r'\1', text)   # Remove spaces around English punctuation
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize punctuation marks
        """
        # Standardize question marks
        text = text.replace('?', '?')  # Ensure ASCII question mark
        
        # Standardize periods
        text = text.replace('।', '।')  # Bengali period (dari)
        
        # Handle mixed punctuation
        text = re.sub(r'[.।]\s*\?', '?', text)  # Remove period before question mark
        
        return text
    
    def normalize_digits(self, text: str) -> str:
        """
        Normalize digit representations (Bengali vs ASCII)
        """
        # Convert Bengali digits to ASCII (optional, depending on requirements)
        bengali_to_ascii = {
            '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
            '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'
        }
        
        # Keep Bengali digits for now (comment out if you want ASCII)
        # for bengali, ascii in bengali_to_ascii.items():
        #     text = text.replace(bengali, ascii)
        
        return text
    
    def normalize_common_words(self, text: str) -> str:
        """
        Apply word-level normalizations for common Bengali words
        """
        for word, normalizer_func in self.problem_sequences.items():
            if word in text:
                text = normalizer_func(text)
        
        return text
    
    def normalize(self, text: str) -> str:
        """
        Apply complete Bengali text normalization
        
        Args:
            text (str): Input Bengali text to normalize
            
        Returns:
            str: Normalized Bengali text
        """
        if not text:
            return text
        
        # Apply all normalization steps in order
        text = self.normalize_unicode(text)
        text = self.normalize_common_words(text)
        text = self.normalize_whitespace(text)
        text = self.normalize_punctuation(text)
        text = self.normalize_digits(text)
        
        return text
    
    def normalize_for_matching(self, text: str) -> str:
        """
        Special normalization for exact matching purposes
        """
        # Apply standard normalization
        text = self.normalize(text)
        
        # Convert to lowercase for case-insensitive matching
        text = text.lower()
        
        # Additional aggressive normalizations for matching
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\u2060\ufeff]', '', text)
        
        # Normalize multiple consecutive whitespace to single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


# Global instance
bengali_normalizer = BengaliNormalizer()

# Convenience functions
def normalize_bengali(text: str) -> str:
    """
    Normalize Bengali text using the global normalizer
    
    Args:
        text (str): Input Bengali text
        
    Returns:
        str: Normalized Bengali text
    """
    return bengali_normalizer.normalize(text)

def normalize_for_matching(text: str) -> str:
    """
    Normalize Bengali text for exact matching purposes
    
    Args:
        text (str): Input Bengali text
        
    Returns:
        str: Normalized Bengali text optimized for matching
    """
    return bengali_normalizer.normalize_for_matching(text)


if __name__ == "__main__":
    # Test the normalizer
    test_cases = [
        "মৃত ব্যক্তির জমির খাজনা/ভূমি উন্নয়ন কর কি অনলাইনের মাধ্যমে দেয়া যাবে?",
        "মৃত ব্যক্তির ভূমি উন্নয়ন কর কি অনলাইনের মাধ্যমে দেয়া যাবে?",
        "নামজারি প্রয়োজনীয় ফি কত?",
        "অনলাইনে কি নক্সা/ম্যাপ দেখা যায়?",
    ]
    
    print("Bengali Text Normalization Test:")
    print("=" * 60)
    
    for text in test_cases:
        normalized = normalize_bengali(text)
        match_ready = normalize_for_matching(text)
        
        print(f"Original: {text}")
        print(f"Normalized: {normalized}")
        print(f"Match-ready: {match_ready}")
        print(f"Bytes: {match_ready.encode('utf-8')}")
        print("-" * 40)