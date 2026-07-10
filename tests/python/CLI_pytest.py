#!/usr/bin/env python3
"""
Pytest for running CLI_inference (DPU, DMA, or SHAVE) on a CSV and comparing vpunn_cycles and inference_cycles/cli_result.
Fails if any differences are found, and prints the CSV line where a difference is spotted.

CSV Column Requirements:
    - All modules: 'vpunn_cycles' (ground truth values)
    - DPU: 'inference_cycles' (CLI results)
    - DMA/SHAVE: 'cli_result' (CLI results)

Usage:
    pytest CLI_pytest.py                                    # Run DPU tests (default)
    pytest CLI_pytest.py --hw_module DPU                    # Run DPU tests
    pytest CLI_pytest.py --hw_module DMA_2_7                # Run DMA tests (NPU 2.7 interface)
    pytest CLI_pytest.py --hw_module DMA_4_0                # Run DMA tests (NPU 4.0+ interface)
    pytest CLI_pytest.py --hw_module SHAVE                  # Run SHAVE tests
    pytest CLI_pytest.py --dpu-csv my_dpu_workloads.csv     # Specify DPU CSV file
    pytest CLI_pytest.py --dma-csv my_dma_workloads.csv     # Specify DMA CSV file
    pytest CLI_pytest.py --shave-csv my_shave_workloads.csv # Specify SHAVE CSV file
    pytest CLI_pytest.py --hw_module DMA_4_0 --dma-csv dma.csv # DMA 4.0 with custom file
"""
import pytest
import pandas as pd
import numpy as np
import os
import subprocess
import sys
from pathlib import Path

def run_cli_inference(input_csv, output_csv=None, base_dir=None, cli_executable=None, hw_module="DPU"):
    """
    Runs CLI_inference_dpu.py, CLI_inference_dma.py, or CLI_inference_shave.py on the input CSV and returns the path to the output CSV.
    If output_csv is not provided, a temporary file is created.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path for output CSV file (optional, auto-generated if None)
        base_dir: Base directory for the project (defaults to project root)
        cli_executable: Path to cost_model_cli executable (auto-detected if None)
        hw_module: Type of test - "DPU", "DMA_2_7", "DMA_4_0", or "SHAVE"
    """
    if output_csv is None:
        output_csv = str(Path(input_csv).with_name("cli_inference_output.csv"))
    
    # Map hw_module to script name (DMA_2_7 and DMA_4_0 both use CLI_inference_dma.py)
    script_hw_module = hw_module.lower()
    if hw_module in ["DMA_2_7", "DMA_4_0"]:
        script_hw_module = "dma"
    
    # Find the appropriate CLI_inference script based on test type
    script_name = f"CLI_inference_{script_hw_module}.py"
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        pytest.fail(f"{script_name} not found at {script_path}")
    
    # Set default base directory to project root
    if base_dir is None:
        base_dir = str(Path(__file__).parent.parent.parent)  # Go up to project root
    
    # Auto-detect CLI executable if not provided
    if cli_executable is None:
        project_root = Path(base_dir)
        # Common build locations for the CLI executable
        # Current build configurations (searched first)
        cli_candidates = [
            project_root / "build" / "bin" / "Debug" / "cost_model_cli.exe",
            project_root / "build" / "bin" / "Release" / "cost_model_cli.exe",
            project_root / "build" / "bin" / "Debug" / "cost_model_cli",
            project_root / "build" / "bin" / "Release" / "cost_model_cli",
            project_root / "out" / "build" / "x64-Debug" / "bin" / "cost_model_cli.exe",
            project_root / "out" / "build" / "x64-Release" / "bin" / "cost_model_cli.exe",
            project_root / "out" / "build" / "x64-Debug" / "bin" / "cost_model_cli",
            project_root / "out" / "build" / "x64-Release" / "bin" / "cost_model_cli",
            # Legacy build configurations (for backward compatibility)
            project_root / "out" / "build" / "x64-Debug" / "apps" / "cost_model_cli" / "cost_model_cli",
            project_root / "out" / "build" / "x64-Debug" / "apps" / "cost_model_cli" / "cost_model_cli.exe",
            project_root / "out" / "build" / "x64-Release" / "apps" / "cost_model_cli" / "cost_model_cli",
            project_root / "out" / "build" / "x64-Release" / "apps" / "cost_model_cli" / "cost_model_cli.exe",
            project_root / "build" / "apps" / "cost_model_cli" / "Debug" / "cost_model_cli",
            project_root / "build" / "apps" / "cost_model_cli" / "Debug" / "cost_model_cli.exe",
            project_root / "build" / "apps" / "cost_model_cli" / "Release" / "cost_model_cli",
            project_root / "build" / "apps" / "cost_model_cli" / "Release" / "cost_model_cli.exe",
            project_root / "build" / "apps" / "cost_model_cli" / "cost_model_cli",
            project_root / "build" / "apps" / "cost_model_cli" / "cost_model_cli.exe",
            project_root / "apps" / "cost_model_cli" / "cost_model_cli",
            project_root / "apps" / "cost_model_cli" / "cost_model_cli.exe",
        ]
        
        for candidate in cli_candidates:
            if candidate.exists():
                cli_executable = str(candidate)
                break
        
        if cli_executable is None:
            pytest.fail(f"Could not find cost_model_cli executable. Checked: {[str(c) for c in cli_candidates]}")
    
    # CLI_inference script expects: <csv_filename> <base_dir> <cli_executable_path>
    cmd = [sys.executable, str(script_path), input_csv, base_dir, cli_executable]
    
    print(f"Running {hw_module.upper()} inference with command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"CLI_inference_{hw_module} failed: {result.stderr}\n{result.stdout}")
    
    # CLI_inference script generates output with pattern: input_with_results.csv
    expected_output = str(Path(input_csv).parent / f"{Path(input_csv).stem}_with_results.csv")
    
    if not os.path.exists(expected_output):
        pytest.fail(f"CLI_inference_{hw_module} did not produce expected output CSV: {expected_output}")
    
    return expected_output

def compare_cycles(csv_path, hw_module="DPU"):
    """
    Compares vpunn_cycles and inference_cycles (or cycles and cli_result for DMA/SHAVE) in the given CSV.
    Fails if any differences are found, and prints the CSV line where a difference is spotted.
    
    Args:
        csv_path: Path to the CSV file to analyze
        hw_module: Type of test - "DPU", "DMA_2_7", "DMA_4_0", or "SHAVE" (for error messaging)
    """
    df = pd.read_csv(csv_path)
    
    # Determine which columns to compare based on hw_module
    # DPU uses 'vpunn_cycles' and 'inference_cycles'
    # DMA uses 'vpunn_cycles' and 'cli_result'
    # SHAVE uses 'cycles' and 'cli_result'
    if hw_module == "DPU":
        vpunn_col = 'vpunn_cycles'
        inference_col = 'inference_cycles'
    elif hw_module == "SHAVE":
        vpunn_col = 'cycles'
        inference_col = 'cli_result'
    else:  # DMA_2_7, DMA_4_0
        vpunn_col = 'vpunn_cycles'
        inference_col = 'cli_result'
    
    if vpunn_col not in df.columns or inference_col not in df.columns:
        pytest.fail(f"CSV missing required columns. Expected '{vpunn_col}' and '{inference_col}'. Columns found: {list(df.columns)}")
    
    differences = []
    for idx, row in df.iterrows():
        v = row[vpunn_col]
        i = row[inference_col]
        if pd.isna(v) and pd.isna(i):
            continue
        if pd.isna(v) or pd.isna(i):
            differences.append(idx)
            continue
        try:
            if not np.isclose(float(v), float(i), rtol=1e-10, atol=1e-10):
                differences.append(idx)
        except Exception:
            if str(v) != str(i):
                differences.append(idx)
    if differences:
        lines = [f"[{hw_module.upper()}] Difference at CSV line {d+2}: {vpunn_col}={df.iloc[d][vpunn_col]}, {inference_col}={df.iloc[d][inference_col]}" for d in differences]
        pytest.fail("\n".join(lines))

def test_cli_inference_cycles(request, tmp_path):
    """
    Pytest: takes a CSV, runs CLI_inference (DPU, DMA, or SHAVE based on --hw_module), and compares cycles.
    """
    # Get hw_module from command line option
    hw_module = request.config.getoption("--hw_module")
    
    # Get CSV filename based on hw_module (DMA_2_7 and DMA_4_0 both use --dma-csv)
    if hw_module == "DPU":
        csv_filename = request.config.getoption("--dpu-csv")
    elif hw_module in ["DMA_2_7", "DMA_4_0"]:
        csv_filename = request.config.getoption("--dma-csv")
    else:  # SHAVE
        csv_filename = request.config.getoption("--shave-csv")
    
    # Check if path is absolute or relative
    csv_path = Path(csv_filename)
    if not csv_path.is_absolute():
        # Search order: 1) test directory, 2) current working directory, 3) project root
        test_dir_path = Path(__file__).parent / csv_filename
        cwd_path = Path.cwd() / csv_filename
        project_root_path = Path(__file__).parent.parent.parent / csv_filename
        
        if test_dir_path.exists():
            input_csv = str(test_dir_path)
        elif cwd_path.exists():
            input_csv = str(cwd_path)
        elif project_root_path.exists():
            input_csv = str(project_root_path)
        else:
            # File not found in any location
            pytest.fail(f"{hw_module.upper()} CSV file not found: {csv_filename}\n" +
                       f"Searched locations:\n" +
                       f"  1. Test directory: {test_dir_path}\n" +
                       f"  2. Current directory: {cwd_path}\n" +
                       f"  3. Project root: {project_root_path}\n" +
                       f"Please provide the full path or place the file in one of these locations.")
    else:
        input_csv = str(csv_path)
        # Check if the absolute path exists
        if not os.path.exists(input_csv):
            pytest.fail(f"{hw_module.upper()} CSV file not found: {input_csv}")
    # Optional: Customize these parameters if needed
    base_dir = None  # Will auto-detect project root
    cli_executable = None  # Will auto-detect cost_model_cli executable
    
    # Run CLI_inference and get the output CSV path
    output_csv = run_cli_inference(input_csv, base_dir=base_dir, cli_executable=cli_executable, hw_module=hw_module)
    
    # Compare the cycles columns
    compare_cycles(output_csv, hw_module=hw_module)
