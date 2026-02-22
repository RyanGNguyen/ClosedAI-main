#!/usr/bin/env python3
"""
Compute-sanitizer runner for next_token_generation
Usage: python run_sanitizer.py <tool> <input_file>
Where tool is one of: memcheck, initcheck, synccheck, or all
"""

import subprocess
import sys

# Modifiable command parameters
SANITIZER_FLAGS = "--show-backtrace yes --print-limit 100 --error-exitcode 1 --destroy-on-device-error kernel"
PROGRAM = "./next_token_generation"
DEFAULT_INPUT = "input.txt"

def run_sanitizer_tool(tool, input_file):
    """Run compute-sanitizer with specified tool"""
    # Build the command as a modifiable string
    cmd = f"compute-sanitizer --tool {tool} {SANITIZER_FLAGS} {PROGRAM} {input_file}"
    
    print("=" * 60)
    print(f"Running {tool.upper()}")
    print("=" * 60)
    print(f"Command: {cmd}")
    print()
    
    # Execute the command
    result = subprocess.run(cmd, shell=True)
    return result.returncode

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_sanitizer.py <tool> <input_file>")
        print("  tool: memcheck | initcheck | synccheck | all")
        print("  input_file: path to input file (default: input.txt)")
        print()
        print("Examples:")
        print("  python run_sanitizer.py initcheck input.txt    # Check for uninitialized memory")
        print("  python run_sanitizer.py memcheck input.txt     # Check for memory errors")
        print("  python run_sanitizer.py all input.txt          # Run all checks")
        print()
        print("Direct command format:")
        print(f"  compute-sanitizer --tool <tool> {SANITIZER_FLAGS} {PROGRAM} <input_file>")
        sys.exit(1)
    
    tool = sys.argv[1]
    input_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_INPUT
    
    tools_to_run = []
    if tool == "all":
        tools_to_run = ["memcheck", "initcheck", "synccheck"]
    elif tool in ["memcheck", "initcheck", "synccheck"]:
        tools_to_run = [tool]
    else:
        print(f"Error: Unknown tool '{tool}'")
        print("Valid tools: memcheck, initcheck, synccheck, all")
        sys.exit(1)
    
    # Run each tool
    exit_codes = []
    for t in tools_to_run:
        exit_code = run_sanitizer_tool(t, input_file)
        exit_codes.append(exit_code)
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for i, t in enumerate(tools_to_run):
        status = "PASSED" if exit_codes[i] == 0 else "FAILED"
        print(f"{t}: {status} (exit code: {exit_codes[i]})")
    
    # Exit with error if any tool failed
    sys.exit(max(exit_codes))

if __name__ == "__main__":
    main()


# ============================================================================
# Alternative: Direct command strings for manual execution
# ============================================================================
# You can copy-paste these commands directly into your terminal:

"""
# Check for uninitialized memory (most relevant for your issue):
compute-sanitizer --tool initcheck --show-backtrace yes --print-limit 100 --error-exitcode 1 --destroy-on-device-error kernel ./next_token_generation input.txt

# Check for memory errors (out-of-bounds access, leaks, etc.):
compute-sanitizer --tool memcheck --show-backtrace yes --print-limit 100 --error-exitcode 1 --destroy-on-device-error kernel ./next_token_generation input.txt

# Check for synchronization issues:
compute-sanitizer --tool synccheck --show-backtrace yes --print-limit 100 --error-exitcode 1 --destroy-on-device-error kernel ./next_token_generation input.txt

# Simplified version with fewer flags:
compute-sanitizer --tool initcheck ./next_token_generation input.txt
"""
