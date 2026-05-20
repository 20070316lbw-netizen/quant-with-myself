import zipfile
import xml.etree.ElementTree as ET
import sys
import os

sys.path.insert(0, os.getcwd())

def read_docx(file_path):
    # docx files are zip files containing xml document structure.
    # The main text content resides in word/document.xml.
    try:
        with zipfile.ZipFile(file_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # Namespaces are important in docx xml format.
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            paragraphs = []
            for para in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                texts = [node.text for node in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                if texts:
                    paragraphs.append("".join(texts))
                else:
                    paragraphs.append("")
            
            return "\n".join(paragraphs)
    except Exception as e:
        return f"Error reading docx: {e}"

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    doc_path = "Alpha158_因子实现手册.docx"
    if os.path.exists(doc_path):
        text = read_docx(doc_path)
        with open("play/Alpha158_manual.md", "w", encoding="utf-8") as f:
            f.write(text)
        print("=== CONTENT OF Alpha158_因子实现手册.docx ===")
        print(text)
    else:
        print(f"File not found: {doc_path}")
