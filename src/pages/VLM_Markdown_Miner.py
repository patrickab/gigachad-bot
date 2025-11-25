import builtins
from collections import deque  # <--- Added missing import
import contextlib
import os
import pty
import select
import subprocess
import time

from ansi2html import Ansi2HTMLConverter
import streamlit as st

from config import DIRECTORY_VLM_OUTPUT
from src.lib.streamlit_helper import nyan_cat_spinner


class ConsoleBuffer:
    def __init__(self, max_lines: int = 1000) -> None:
        self.lines = deque(maxlen=max_lines)
        self.current_line = ""

    def write(self, text: str) -> None:
        # 1. Normalize line endings to handle Windows-style \r\n correctly.
        text = text.replace('\r\n', '\n')
        
        for char in text:
            if char == '\n':
                # Commit current line to history
                self.lines.append(self.current_line)
                self.current_line = ""
            elif char == '\r':
                # This is the "Magic": \r moves cursor to start. 
                # In this buffer logic, its treated as "wiping" the current line
                # so the next characters overwrite it (creating loading bar effect).
                self.current_line = ""
            else:
                self.current_line += char

    def get_content(self) -> str:
        # Join history + current active line
        return "\n".join([*list(self.lines), self.current_line])

def run_command_with_output(command_list: list[str]) -> None:
    output_container = st.empty()
    conv = Ansi2HTMLConverter(dark_bg=False)
    buffer = ConsoleBuffer(max_lines=500)

    master_fd, slave_fd = pty.openpty()

    process = subprocess.Popen(
        command_list,
        stdout=slave_fd,
        stderr=slave_fd, 
        close_fds=True,
        text=True
    )

    os.close(slave_fd)

    last_update_time = 0
    last_content_hash = None
    UPDATE_INTERVAL = 0.1 

    try:
        while True:
            r, _, _ = select.select([master_fd], [], [], 0.1)

            if master_fd in r:
                try:
                    data = os.read(master_fd, 1024).decode('utf-8', errors='replace')
                    if not data:
                        break 

                    buffer.write(data)

                    current_time = time.time()
                    if current_time - last_update_time > UPDATE_INTERVAL:
                        raw_content = buffer.get_content()

                        # Only render if something changed
                        if raw_content != last_content_hash:
                            html_content = conv.convert(raw_content, full=False)

                            # Cleaned up CSS:
                            # 1. Removed flex-reverse (it causes text ordering bugs)
                            # 2. Added overflow-anchor (helps browser handle scrolling)
                            styled_html = f"""
                            <div style="
                                background-color: #1e1e1e; 
                                color: #d4d4d4; 
                                padding: 15px; 
                                border-radius: 5px; 
                                font-family: 'Courier New', Courier, monospace; 
                                font-size: 14px; 
                                line-height: 1.5; 
                                height: 400px; 
                                overflow-y: auto; 
                                overflow-anchor: auto;
                                white-space: pre-wrap;">
                                {html_content}
                            </div>
                            """
                            output_container.markdown(styled_html, unsafe_allow_html=True)
                            last_content_hash = raw_content
                            last_update_time = current_time

                except OSError:
                    break

            if process.poll() is not None and master_fd not in r:
                break

    finally:
        with contextlib.suppress(builtins.BaseException):
            os.close(master_fd)
        process.wait()

    # Render Console
    html_content = conv.convert(buffer.get_content(), full=False)
    styled_html = f"""
    <div style="
        background-color: #1e1e1e; 
        color: #d4d4d4; 
        padding: 15px; 
        border-radius: 5px; 
        font-family: 'Courier New', Courier, monospace; 
        font-size: 14px; 
        line-height: 1.5; 
        height: 400px; 
        overflow-y: auto;
        white-space: pre-wrap;">
        {html_content}
    </div>
    """
    output_container.markdown(styled_html, unsafe_allow_html=True)


def vlm_markdown_miner() -> None:
    """
    VLM-powered Markdown Miner for Obsidian Notes.
    """
    st.title("VLM Markdown Miner")
    
    st.markdown("""
    **Hardware Requirements:**
    - GPU with at least 8GB VRAM.
    - CPU with at least 16GB RAM.
    """)

    if st.button("Run VLM Markdown Miner", key="run_vlm_markdown_miner"):
        with nyan_cat_spinner(), st.spinner("Installing mineru package..."):
            subprocess.run(["uv", "pip", "install", "mineru[core]"])    

        with nyan_cat_spinner():
            run_command_with_output(["bash", "src/static/pdf-minerU.sh"])
    subprocess.run(["bash", "mv", "converted*", DIRECTORY_VLM_OUTPUT])

if __name__ == "__main__":
    vlm_markdown_miner()
