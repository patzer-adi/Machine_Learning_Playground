import os
import json
import re
import ast
from fpdf import FPDF

# Configuration
SOURCE_DIR = "/Users/adityagowari/Programs/Machine_Learning_Playground"
OUTPUT_DIR = os.path.join(SOURCE_DIR, "simplified_code_pdfs")
EXTENSIONS_TO_CONVERT = {'.py', '.ipynb', '.md', '.txt', '.cpp', '.cu', '.h', '.c'}
EXCLUDE_DIRS = {'.git', '.venv', '.conda', '__pycache__', '.ipynb_checkpoints', 'all_code_pdfs', 'simplified_code_pdfs'}

# Color Palettes (R, G, B)
COLORS = {
    'BLUE': (30, 144, 255),    # Explanatory Comments
    'BLACK': (0, 0, 0),        # Code
    'RED': (220, 20, 60),      # Definitions (class, def)
    'GREEN': (34, 139, 34),    # Section Headers
    'ORANGE': (255, 140, 0),   # Logic Focus / Study Tips
    'GRAY': (128, 128, 128)    # Footers
}

class CodeSimplifier:
    """Refactors and simplifies code for educational purposes."""
    
    ML_NAMING_MAP = {
        'x': 'features',
        'y': 'target',
        'lr': 'learning_rate',
        'clf': 'classifier',
        'acc': 'accuracy',
        'df': 'dataframe',
        'plt': 'plot_tool',
        'np': 'numpy_lib',
        'pd': 'pandas_lib'
    }

    @staticmethod
    def simplify_python(code):
        try:
            tree = ast.parse(code)
            # 1. Basic cleaning: Remove docstrings that are just placeholder strings
            # and prepare for educational injections.
            
            # Since full AST refactoring into source is complex without libraries like 'astor' or 'black',
            # we will use a hybrid approach: AST for structure analysis + Regex for precise replacement.
            
            simplified_lines = []
            lines = code.split('\n')
            
            # Identify function/class positions
            meta = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    meta.append({'type': 'function', 'name': node.name, 'line': node.lineno, 'args': [a.arg for a in node.args.args]})
                elif isinstance(node, ast.ClassDef):
                    meta.append({'type': 'class', 'name': node.name, 'line': node.lineno})

            # Process lines
            for i, line in enumerate(lines):
                line_idx = i + 1
                stripped = line.strip()
                
                # Skip original comments that look like junk (too many hashes, empty ones)
                if stripped.startswith('#') and len(stripped) < 3:
                    continue
                
                # Check for metadata injections
                for m in meta:
                    if m['line'] == line_idx:
                        if m['type'] == 'function':
                            simplified_lines.append(f"\n# [STUDY] Function: {m['name']}")
                            simplified_lines.append(f"# Inputs: {', '.join(m['args']) if m['args'] else 'None'}")
                            simplified_lines.append(f"# Returns: Calculated output for {m['name']}")
                        elif m['type'] == 'class':
                            simplified_lines.append(f"\n# [STRUCTURE] Class: {m['name']}")
                
                # Rename confusing variables via regex (simple implementation)
                new_line = line
                for old, new in CodeSimplifier.ML_NAMING_MAP.items():
                    # Match variable with word boundaries to avoid renaming 'extra' to 'e_features_tra'
                    new_line = re.sub(rf'\b{old}\b', new, new_line)
                
                simplified_lines.append(new_line)
            
            return "\n".join(simplified_lines)
        except:
            return code # Fallback to original if AST fails

    @staticmethod
    def simplify_generic(code, ext):
        # Remove massive comment blocks and add educational placeholders
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL) # Remove C-style blocks
        lines = code.split('\n')
        processed = []
        for line in lines:
            if line.strip() == "": continue
            # Add labels for C++/CUDA specific blocks
            if "__global__" in line:
                processed.append("// KEY LOGIC: CUDA Kernel running on Device")
            processed.append(line)
        return "\n".join(processed)

    @staticmethod
    def get_educational_header(content, rel_path):
        purpose = "Machine Learning Implementation Strategy"
        if "regression" in content.lower(): purpose = "Continuous variable prediction using optimization"
        if "bayes" in content.lower(): purpose = "Probabilistic classification using feature independence"
        if "knn" in content.lower(): purpose = "Instance-based learning using distance metrics"
        
        return (
            f"--------------------------------------------------\n"
            f"FILE: {rel_path}\n"
            f"PURPOSE: {purpose}\n"
            f"KEY COMPONENTS: Processed Logic & Simplified Structure\n"
            f"--------------------------------------------------\n\n"
        )

class VisualStudyPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.current_file = ""

    def header(self):
        if self.page_no() > 1:
            self.set_font('Arial', 'I', 8)
            self.set_text_color(*COLORS['GRAY'])
            self.cell(0, 10, f"Simplified Track: {self.current_file}", 0, 1, 'R')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(*COLORS['GRAY'])
        self.cell(0, 10, f'Simplified Study Note | Page {self.page_no()}', 0, 0, 'C')

    def add_section(self, title):
        self.ln(5)
        self.set_font("Arial", 'B', 12)
        self.set_text_color(*COLORS['GREEN'])
        self.cell(0, 10, f"=== {title} ===", 0, 1, 'L')
        self.ln(2)

    def write_line(self, line, color_key='BLACK'):
        self.set_font("Courier", size=8)
        self.set_text_color(*COLORS[color_key])
        safe_line = line.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 4, safe_line)

def render_simplified_content(pdf, code, ext):
    lines = code.split('\n')
    current_section = None
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            pdf.ln(1)
            continue

        # Section Detection
        if (line.startswith('import ') or line.startswith('from ')) and current_section != 'Imports':
            pdf.add_section("Imports")
            current_section = 'Imports'
        elif (line.startswith('class ')) and current_section != 'Classes':
            pdf.add_section("Classes")
            current_section = 'Classes'
        elif (line.startswith('def ')) and current_section != 'Functions':
            pdf.add_section("Functions")
            current_section = 'Functions'
        elif 'if __name__ == "__main__":' in line:
            pdf.add_section("Main Execution")
            current_section = 'Main'

        # Color Formatting
        color = 'BLACK'
        if stripped.startswith('#') or stripped.startswith('//'):
            color = 'BLUE'
            if "STUDY" in stripped or "LOGIC" in stripped:
                color = 'ORANGE'
        elif line.startswith('def ') or line.startswith('class '):
            color = 'RED'
        elif line.startswith('import ') or line.startswith('from '):
            color = 'RED'

        pdf.write_line(line, color)

def process_notebook(pdf, filepath, rel_path):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Simplified Notebook: {rel_path}", 1, 1, 'C')
        pdf.ln(5)

        for cell in nb.get('cells', []):
            cell_type = cell.get('cell_type')
            source = "".join(cell.get('source', []))
            
            if cell_type == 'markdown':
                pdf.set_font("Arial", 'B', 10)
                pdf.set_text_color(*COLORS['GREEN'])
                pdf.cell(0, 8, "[ --- MARKDOWN CELL --- ]", 0, 1, 'L')
                pdf.set_font("Arial", '', 10)
                pdf.set_text_color(*COLORS['BLACK'])
                pdf.multi_cell(0, 5, source.encode('latin-1', 'replace').decode('latin-1'))
                pdf.ln(2)
            
            elif cell_type == 'code':
                pdf.set_font("Arial", 'B', 10)
                pdf.set_text_color(*COLORS['ORANGE'])
                pdf.cell(0, 8, "[ --- SIMPLIFIED CODE CELL --- ]", 0, 1, 'L')
                simplified = CodeSimplifier.simplify_python(source)
                render_simplified_content(pdf, simplified, '.py')
                pdf.ln(2)
        return True
    except Exception as e:
        pdf.write_line(f"Error processing notebook: {e}", 'RED')
        return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    files_processed = 0
    for root, dirs, files in os.walk(SOURCE_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            name, ext = os.path.splitext(file)
            ext = ext.lower()
            if ext in EXTENSIONS_TO_CONVERT:
                if "generate_pdfs.py" in file: continue
                
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, SOURCE_DIR)
                safe_name = rel_path.replace(os.sep, '_') + ".pdf"
                output_path = os.path.join(OUTPUT_DIR, safe_name)
                
                print(f"📖 Simplifying & Generating: {rel_path}")
                
                pdf = VisualStudyPDF()
                pdf.current_file = rel_path
                
                try:
                    if ext == '.ipynb':
                        success = process_notebook(pdf, full_path, rel_path)
                    else:
                        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                        
                        # Step 1: Simplify Code
                        if ext == '.py':
                            simplified = CodeSimplifier.simplify_python(content)
                        else:
                            simplified = CodeSimplifier.simplify_generic(content, ext)
                        
                        # Step 2: Add Educational Header
                        header = CodeSimplifier.get_educational_header(simplified, rel_path)
                        final_content = header + simplified
                        
                        # Step 3: Render to PDF
                        pdf.add_page()
                        render_simplified_content(pdf, final_content, ext)
                        success = True
                except Exception as e:
                    pdf.add_page()
                    pdf.write_line(f"Error processing file: {e}", 'RED')
                    success = True # Still save the error report PDF
                
                if success:
                    pdf.output(output_path)
                    files_processed += 1

    print(f"\n✅ All simplified study notes generated!")
    print(f"📂 Total files: {files_processed}")
    print(f"📍 Directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
