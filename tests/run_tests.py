#!/usr/bin/env python3
"""
SpectralLLM Test Runner
======================

Comprehensive test runner with multiple testing modes and detailed reporting.
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    print(f"Duration: {duration:.2f}s")
    print(f"Exit code: {result.returncode}")
    
    if result.stdout:
        print(f"\nSTDOUT:\n{result.stdout}")
    
    if result.stderr:
        print(f"\nSTDERR:\n{result.stderr}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run SpectralLLM tests")
    parser.add_argument("--mode", choices=[
        "quick", "unit", "integration", "validation", "performance", "all"
    ], default="quick", help="Test mode to run")
    
    parser.add_argument("--coverage", action="store_true", 
                       help="Run with coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, default=1,
                       help="Number of parallel workers")
    parser.add_argument("--gpu", action="store_true",
                       help="Run GPU tests (requires CUDA/MPS)")
    parser.add_argument("--slow", action="store_true",
                       help="Include slow tests")
    parser.add_argument("--report-dir", default="test_reports",
                       help="Directory for test reports")
    
    args = parser.parse_args()
    
    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        pytest_cmd.extend(["-v", "-s"])
    
    # Add parallel execution
    if args.parallel > 1:
        pytest_cmd.extend(["-n", str(args.parallel)])
    
    # Add coverage
    if args.coverage:
        pytest_cmd.extend([
            "--cov=spectralllm",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=term-missing"
        ])
    
    # Create report directory
    report_dir = Path(args.report_dir)
    report_dir.mkdir(exist_ok=True)
    
    # Add JUnit XML reporting
    pytest_cmd.extend([
        "--junitxml", str(report_dir / "junit.xml")
    ])
    
    # Configure test selection based on mode
    test_results = []
    
    if args.mode == "quick":
        # Quick smoke tests
        cmd = pytest_cmd + [
            "-m", "unit and not slow",
            "tests/unit/test_core_config.py",
            "tests/unit/test_core_model.py::TestSpectralLLMModel::test_model_creation",
            "--maxfail=3"
        ]
        result = run_command(cmd, "Quick Smoke Tests")
        test_results.append(("Quick Tests", result.returncode == 0))
        
    elif args.mode == "unit":
        # All unit tests
        markers = ["unit"]
        if not args.slow:
            markers.append("not slow")
        
        cmd = pytest_cmd + [
            "-m", " and ".join(markers),
            "tests/unit/"
        ]
        result = run_command(cmd, "Unit Tests")
        test_results.append(("Unit Tests", result.returncode == 0))
        
    elif args.mode == "integration":
        # Integration tests
        markers = ["integration"]
        if not args.slow:
            markers.append("not slow")
        if not args.gpu:
            markers.append("not gpu")
            
        cmd = pytest_cmd + [
            "-m", " and ".join(markers),
            "tests/integration/"
        ]
        result = run_command(cmd, "Integration Tests")
        test_results.append(("Integration Tests", result.returncode == 0))
        
    elif args.mode == "validation":
        # Validation framework tests
        markers = ["validation"]
        if not args.slow:
            markers.append("not slow")
            
        cmd = pytest_cmd + [
            "-m", " and ".join(markers),
            "tests/validation/"
        ]
        result = run_command(cmd, "Validation Tests")
        test_results.append(("Validation Tests", result.returncode == 0))
        
    elif args.mode == "performance":
        # Performance tests
        markers = ["performance"]
        if not args.gpu:
            markers.append("not gpu")
            
        cmd = pytest_cmd + [
            "-m", " and ".join(markers),
            "tests/performance/",
            "--durations=0"  # Show all durations
        ]
        result = run_command(cmd, "Performance Tests")
        test_results.append(("Performance Tests", result.returncode == 0))
        
    elif args.mode == "all":
        # Run all test categories
        categories = [
            ("Unit Tests", "unit", "tests/unit/"),
            ("Integration Tests", "integration", "tests/integration/"),
            ("Validation Tests", "validation", "tests/validation/"),
        ]
        
        if args.gpu:
            categories.append(("Performance Tests", "performance", "tests/performance/"))
        
        for name, marker, path in categories:
            markers = [marker]
            if not args.slow and marker != "performance":
                markers.append("not slow")
            if not args.gpu:
                markers.append("not gpu")
                
            cmd = pytest_cmd + [
                "-m", " and ".join(markers),
                path
            ]
            result = run_command(cmd, name)
            test_results.append((name, result.returncode == 0))
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("TEST SUMMARY REPORT")
    print(f"{'='*60}")
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall Status: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    
    # Additional reports
    if args.coverage:
        print(f"\n📊 Coverage report available at: htmlcov/index.html")
    
    print(f"📋 JUnit XML report: {report_dir / 'junit.xml'}")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main() 