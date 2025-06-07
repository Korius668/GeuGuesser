import os
import easyocr
from collections import Counter
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

def detect_languages(frame_dir):
    """
    Detects all languages present in text within PNG image frames in a given directory
    using multiple EasyOCR readers for text extraction and langdetect for language identification.
    Addresses EasyOCR's compatibility requirement for Chinese models.

    Args:
        frame_dir (str): The path to the directory containing the image frames (PNG files).

    Returns:
        list: A list of unique languages detected across all frames.
              Language names are human-readable (e.g., "English", "Japanese").
    """
    # Set seed for reproducibility of langdetect (optional, but good practice)
    DetectorFactory.seed = 0

    # Initialize multiple EasyOCR readers to handle language compatibility requirements.
    # Each reader will attempt to use GPU, falling back to CPU if not available or error.

    # Reader for Traditional Chinese and English (required by EasyOCR)
    try:
        reader_ch_tra = easyocr.Reader(['ch_tra', 'en'], gpu=True)
        print("EasyOCR reader for Traditional Chinese initialized with GPU.")
    except Exception as e:
        print(f"Warning: GPU error for ch_tra_en reader. Falling back to CPU. Error: {e}")
        reader_ch_tra = easyocr.Reader(['ch_tra', 'en'], gpu=False)
        print("EasyOCR reader for Traditional Chinese initialized with CPU.")

    # Reader for Simplified Chinese and English (required by EasyOCR)
    try:
        reader_ch_sim = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        print("EasyOCR reader for Simplified Chinese initialized with GPU.")
    except Exception as e:
        print(f"Warning: GPU error for ch_sim_en reader. Falling back to CPU. Error: {e}")
        reader_ch_sim = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        print("EasyOCR reader for Simplified Chinese initialized with CPU.")
    
    try:
        reader_ja = easyocr.Reader(["ja","en"], gpu=True)
        print("EasyOCR reader for Japanese initialized with GPU.")
    except Exception as e:
        print(f"Warning: GPU error for ja_en reader. Falling back to CPU. Error: {e}")
        reader_ja = easyocr.Reader(["ja","en"], gpu=False)
        print("EasyOCR reader for Japanese initialized with CPU.")

    try:
        reader_ko = easyocr.Reader(['ko', 'en'], gpu=True)
        print("EasyOCR reader for Korean initialized with GPU.")
    except Exception as e:
        print(f"Warning: GPU error for Korean reader. Falling back to CPU. Error: {e}")
        reader_ko = easyocr.Reader(['ko', 'en'], gpu=False)
        print("EasyOCR reader for Korean initialized with CPU.")

    try:
        reader_ru = easyocr.Reader(['ru', 'en'], gpu=True)
        print("EasyOCR reader for Russian initialized with GPU.")
    except Exception as e:
        print(f"Warning: GPU error for Korean reader. Falling back to CPU. Error: {e}")
        reader_ru = easyocr.Reader(['ru', 'en'], gpu=False)
        print("EasyOCR reader for Russian initialized with CPU.")

    try:
        reader_ar = easyocr.Reader(['ar', "en"], gpu=True)
        print("EasyOCR reader for Arabic initialized with GPU.")
    except Exception as e:
        print(f"Warning: GPU error for Korean reader. Falling back to CPU. Error: {e}")
        reader_ar = easyocr.Reader(['ar', 'en'], gpu=False)
        print("EasyOCR reader for Arabic initialized with CPU.")   

    # Reader for other non-Chinese, non-English languages
    # Note: 'en' is not included here to avoid redundancy and potential conflicts,
    # as it's already part of the Chinese readers. If a text is purely English,
    # one of the Chinese readers will likely pick it up.
    other_langs = ['fr', 'de', 'en']
    try:
        reader_others = easyocr.Reader(other_langs, gpu=True)
        print(f"EasyOCR reader for {', '.join(other_langs)} initialized with GPU.")
    except Exception as e:
        print(f"Warning: GPU error for other languages reader. Falling back to CPU. Error: {e}")
        reader_others = easyocr.Reader(other_langs, gpu=False)
        print(f"EasyOCR reader for {', '.join(other_langs)} initialized with CPU.")

    # List of all readers to iterate through
    #all_readers = [reader_ch_tra, reader_ch_sim, reader_ja, reader_ko, reader_ru, reader_others]
    all_readers = [reader_ch_tra, reader_ch_sim, reader_ja, reader_ko, reader_ru, reader_ar, reader_others]

    detected_languages_list = [] # This list will store all detected languages (can have duplicates)

    # Iterate through all files in the specified directory, sorted by name
    for file_name in sorted(os.listdir(frame_dir)):
        # Process only files that end with '.png' (case-insensitive)
        if file_name.lower().endswith(".png"):
            file_path = os.path.join(frame_dir, file_name)
            print(f"\nProcessing file: {file_path}") # Print current file being processed

            # Store all text results from different readers for the current image
            image_text_results = []

            # Attempt to read text using each configured reader
            for i, reader in enumerate(all_readers):
                try:
                    # Use EasyOCR to read text from the image.
                    # The 'readtext' method returns a list of tuples:
                    # (bounding_box_coordinates, detected_text, confidence_score)
                    results = reader.readtext(file_path)
                    if results:
                        image_text_results.extend(results)
                        print(f"  Reader {i+1} found {len(results)} text snippets.")
                except Exception as e:
                    print(f"  Error with Reader {i+1} on file {file_name}: {e}")
                    # Continue to the next reader even if one fails

            if not image_text_results:
                print(f"  No text detected in {file_name} by any reader.")
                continue # Move to the next file if no text is found

            # Process all collected text snippets for the current image
            for (bbox, text, confidence) in image_text_results:
                # Only attempt language detection if the extracted text is not empty or just whitespace
                if text.strip():
                    try:
                        # Use the 'langdetect.detect' function to identify the language of the text snippet.
                        # It returns the ISO 639-1 language code (e.g., 'en', 'fr', 'zh-cn').
                        lang_code = detect(text)

                        # Map langdetect's ISO 639-1 codes to more descriptive, human-readable names.
                        # This mapping can be extended for more languages as needed.
                        if lang_code == 'en':
                            detected_languages_list.append("English")
                        elif lang_code == 'fr':
                            detected_languages_list.append("French")
                        elif lang_code == 'de':
                            detected_languages_list.append("German")
                        elif lang_code == 'ja':
                            detected_languages_list.append("Japanese")
                        elif lang_code == 'ko':
                            detected_languages_list.append("Korean")
                        elif lang_code == 'ru':
                            detected_languages_list.append("Russian")
                        elif lang_code == 'ar':
                            detected_languages_list.append("Arabic")
                        elif lang_code == 'zh-cn': # ISO code for Simplified Chinese
                            detected_languages_list.append("Simplified Chinese")
                        elif lang_code == 'zh-tw': # ISO code for Traditional Chinese
                            detected_languages_list.append("Traditional Chinese")
                        else:
                            # For any other languages detected by langdetect, append their ISO code
                            detected_languages_list.append(f"Other ({lang_code})")

                        print(f"  Detected text: '{text}' (Confidence: {confidence:.2f}) -> Language: {lang_code}")

                    except LangDetectException:
                        # Catch specific exception if langdetect cannot determine the language
                        # (e.g., for very short or ambiguous text snippets)
                        print(f"  Could not detect language for text: '{text}'")
                    except Exception as lang_e:
                        # Catch any other unexpected errors during language detection
                        print(f"  Error during language detection for text '{text}': {lang_e}")
                else:
                    print(f"  Skipping empty text snippet.")

    # Use collections.Counter to count occurrences and then get a list of unique languages.
    # This provides the distinct languages found across all processed frames.
    unique_languages = list(Counter(detected_languages_list).keys())
    return unique_languages
