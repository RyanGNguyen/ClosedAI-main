#!/bin/bash

# Compute sanitizer commands for next_token_generation
# Usage: ./run_sanitizer.sh <tool> <input_file>
# Where tool is one of: memcheck, initcheck, synccheck, or all

SANITIZER_FLAGS="--show-backtrace yes --print-limit 100 --error-exitcode 1 --destroy-on-device-error kernel"
PROGRAM="./next_token_generation"
INPUT_FILE="${2:-input.txt}"

if [ "$1" == "memcheck" ] || [ "$1" == "all" ]; then
    echo "========================================="
    echo "Running MEMCHECK (memory errors)"
    echo "========================================="
    compute-sanitizer --tool memcheck $SANITIZER_FLAGS $PROGRAM $INPUT_FILE
fi

if [ "$1" == "initcheck" ] || [ "$1" == "all" ]; then
    echo "========================================="
    echo "Running INITCHECK (uninitialized memory)"
    echo "========================================="
    compute-sanitizer --tool initcheck $SANITIZER_FLAGS $PROGRAM $INPUT_FILE
fi

if [ "$1" == "synccheck" ] || [ "$1" == "all" ]; then
    echo "========================================="
    echo "Running SYNCCHECK (synchronization)"
    echo "========================================="
    compute-sanitizer --tool synccheck $SANITIZER_FLAGS $PROGRAM $INPUT_FILE
fi

if [ "$1" != "memcheck" ] && [ "$1" != "initcheck" ] && [ "$1" != "synccheck" ] && [ "$1" != "all" ]; then
    echo "Usage: $0 <tool> <input_file>"
    echo "  tool: memcheck | initcheck | synccheck | all"
    echo "  input_file: path to input file (default: input.txt)"
    echo ""
    echo "Examples:"
    echo "  $0 initcheck input.txt    # Check for uninitialized memory"
    echo "  $0 memcheck input.txt     # Check for memory errors"
    echo "  $0 all input.txt          # Run all checks"
fi
