import subprocess
import py_compile
import sys

def run():
    restored = False
    for i in range(15):
        try:
            content = subprocess.check_output(["git", "show", f"HEAD~{i}:dictation_engine.py"]).decode("utf-8")
            with open("temp_test.py", "w", encoding="utf-8") as f:
                f.write(content)
            
            # Check syntax
            py_compile.compile("temp_test.py", doraise=True)
            
            # Restore the file
            subprocess.check_call(["git", "checkout", f"HEAD~{i}", "dictation_engine.py"])
            restored = True
            break
        except Exception:
            continue
            
    if not restored:
        sys.exit(1)

    # Now we have a clean file. Let's patch it.
    with open("dictation_engine.py", "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    in_toggle = False
    patched = False

    for i, line in enumerate(lines):
        if "def toggle_recording" in line:
            in_toggle = True
        
        if in_toggle and not patched and line.strip() == "if text:" and i + 1 < len(lines) and "type_text" in lines[i+1]:
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(indent + "t_lower = text.strip().lower()\n")
            new_lines.append(indent + "hallucinations = {\"obrigado.\", \"obrigada.\", \"obrigado\", \"obrigada\", \"obrigado!\", \"obrigada!\", \"obrigado por assistir.\", \"obrigada por assistir.\", \"thank you.\", \"thank you\", \"thanks for watching.\", \"obrigado por assistir\"}\n")
            new_lines.append(indent + "if t_lower in hallucinations:\n")
            new_lines.append(indent + "    log(f\"Ignoring hallucination: '{text}'\")\n")
            new_lines.append(indent + "    text = \"\"\n\n")
            new_lines.append(line)
            patched = True
        else:
            new_lines.append(line)

    with open("dictation_engine.py", "w", encoding="utf-8") as f:
        f.writelines(new_lines)
        
    # Check syntax after patching
    py_compile.compile("dictation_engine.py", doraise=True)
    
    subprocess.check_call(["git", "add", "dictation_engine.py"])
    subprocess.check_call(["git", "commit", "-m", "Restore clean file and apply hallucination filter reliably"])

if __name__ == "__main__":
    run()
