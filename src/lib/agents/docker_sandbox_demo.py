# %% [markdown]
# # Docker Sandbox Verification
# Ensure you have the `DockerSandbox` class defined or imported before running this.
from src.lib.agents.docker_sandbox import DockerSandbox

# %%
# 1. Initialize the Sandbox
# This will build the image (if missing) and start the container.
try:
    sandbox = DockerSandbox()
    print(f"✅ Sandbox initialized. Container ID: {sandbox.container_id}")
except Exception as e:
    print(f"❌ Initialization failed: {e}")

# %%
# 2. Define the Fibonacci Task
# We calculate f_0 through f_10 and print them to stdout.
# We also write the result to a file in the shared workspace to test persistence.

fib_script = """
import os

def get_fibonacci_sequence(n):
    sequence = []
    a, b = 0, 1
    for _ in range(n):
        sequence.append(a)
        a, b = b, a + b
    return sequence

# Calculate f_0 -> f_10 (11 numbers)
count = 11
fib_seq = get_fibonacci_sequence(count)

# 1. Print to STDOUT (captured by logs)
print(f"Fibonacci Sequence (f0-f10): {fib_seq}")

# 2. Write to Workspace (captured by host volume)
output_path = "/app/workspace/fib_output.txt"
with open(output_path, "w") as f:
    f.write(str(fib_seq))
    
print(f"Saved result to {output_path}")
"""

# %%
# 3. Execute the Code
print("⏳ Executing Fibonacci script inside container...")
execution = sandbox.run_code(fib_script)

# %%
# 4. Verify Results (Logs)
print("--- Execution Logs ---")
if execution.logs.stderr:
    print(f"❌ STDERR:\n{execution.logs.stderr}")
else:
    print(f"✅ STDOUT:\n{execution.logs.stdout}")

# %%
# 5. Verify Persistence (Host Volume)
# The container wrote to /app/workspace/fib_output.txt
# Because of the volume bind, this should exist on your Host machine in ./workspace/fib_output.txt

import os

host_file_path = os.path.join(sandbox.workspace_path, "fib_output.txt")

if os.path.exists(host_file_path):
    with open(host_file_path, "r") as f:
        content = f.read()
    print(f"✅ Host File Check: Found 'fib_output.txt' in {sandbox.workspace_path}")
    print(f"📂 File Content: {content}")
else:
    print(f"❌ Host File Check: File not found at {host_file_path}")

# %%
# 6. Cleanup
# Stop and remove the container.
sandbox.stop()
print("🛑 Sandbox stopped.")