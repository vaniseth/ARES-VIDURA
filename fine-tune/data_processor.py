import os
from unstructured.partition.pdf import partition_pdf

def process_pdfs_with_unstructured(folder_path, cache_folder):
    """
    Processes all PDFs in a folder using the 'unstructured' library,
    returning a dictionary of {filename: text}. Caches results to text files.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Input PDF folder '{folder_path}' not found.")
        return {}
    if not os.path.isdir(cache_folder):
        print(f"Creating cache directory: {cache_folder}")
        os.makedirs(cache_folder)
        
    processed_docs = {}
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    print(f"\nFound {len(pdf_files)} PDF(s) to process with 'unstructured'.")
    
    for filename in pdf_files:
        pdf_path = os.path.join(folder_path, filename)
        # Use a more robust cache filename
        output_txt_path = os.path.join(cache_folder, os.path.basename(filename) + '.txt')
        
        file_text = ""
        if os.path.exists(output_txt_path):
            print(f"  -> Found cached output for '{filename}'. Loading from cache.")
            with open(output_txt_path, 'r', encoding='utf-8') as f:
                file_text = f.read()
        else:
            print(f"  -> Processing '{filename}' with unstructured (fast strategy)...")
            try:
                # Use strategy="fast" for speed, which is great for this use case.
                # 'pymupdf' is a fast and reliable PDF parsing library.
                elements = partition_pdf(filename=pdf_path, strategy="fast", pdf_library="pymupdf")
                file_text = "\n\n".join([str(el) for el in elements])
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(file_text)
                print(f"  -> Successfully processed and cached '{filename}'.")
            except Exception as e:
                print(f"  -> Error processing '{filename}' with unstructured. Error: {e}")
                continue
        
        if file_text:
            processed_docs[filename] = file_text
            
    return processed_docs