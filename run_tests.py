#!/usr/bin/env python3
"""
Test runner for the Color Wheel project test suite.

This script provides a convenient way to run tests with different configurations
and options. It wraps pytest with sensible defaults and additional functionality.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --fast             # Run only fast tests
    python run_tests.py --slow             # Run only slow tests  
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --gpu              # Run GPU-specific tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --performance      # Run performance tests
    python run_tests.py --help             # Show all options
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def check_dependencies():
    """Check if required test dependencies are available."""
    required_packages = [
        'pytest',
        'numpy',
        'opencv-python',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required dependencies are available")
    return True


def check_optional_dependencies():
    """Check for optional dependencies and show status."""
    optional_packages = {
        'pytest-cov': 'Coverage reporting',
        'pytest-xdist': 'Parallel test execution',
        'pytest-benchmark': 'Performance benchmarking',
        'psutil': 'Memory usage monitoring',
        'scikit-learn': 'KDTree optimization tests',
        'numba': 'JIT compilation tests',
        'cupy': 'GPU acceleration tests',
        'colour-science': 'Color space conversion tests',
    }
    
    print("\n📋 Optional dependencies status:")
    for package, description in optional_packages.items():
        try:
            pkg_name = package.replace('-', '_')
            if package == 'opencv-python':
                pkg_name = 'cv2'
            elif package == 'scikit-learn':
                pkg_name = 'sklearn'
            elif package == 'colour-science':
                pkg_name = 'colour'
            
            __import__(pkg_name)
            print(f"   ✅ {package:<18} - {description}")
        except ImportError:
            print(f"   ❌ {package:<18} - {description}")


def run_tests(args):
    """Run pytest with specified arguments."""
    # Base pytest command
    cmd = ['python', '-m', 'pytest']
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(['--cov=color_wheel', '--cov-report=term-missing', '--cov-report=html'])
    
    # Test selection options
    if args.fast:
        cmd.extend(['-m', 'not slow'])
    elif args.slow:
        cmd.extend(['-m', 'slow'])
    elif args.gpu:
        cmd.extend(['-m', 'gpu'])
    elif args.unit:
        cmd.extend(['-k', 'not integration'])
    elif args.integration:
        cmd.extend(['-m', 'integration'])
    elif args.performance:
        cmd.extend(['tests/test_performance_edge_cases.py'])
    
    # Verbosity options
    if args.verbose:
        cmd.append('-v')
    elif args.quiet:
        cmd.append('-q')
    
    # Parallel execution
    if args.parallel and 'pytest-xdist' in sys.modules:
        cmd.extend(['-n', str(args.parallel)])
    
    # Stop on first failure
    if args.fail_fast:
        cmd.append('-x')
    
    # Show output from print statements
    if args.capture == 'no':
        cmd.append('-s')
    
    # Add any additional pytest arguments
    if args.pytest_args:
        cmd.extend(args.pytest_args)
    
    # Add test paths if specified
    if args.test_paths:
        cmd.extend(args.test_paths)
    else:
        cmd.append('tests/')
    
    print(f"🚀 Running: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    print("-" * 60)
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code {result.returncode}")
    
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Color Wheel test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Test selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--fast', action='store_true',
                           help='Run only fast tests (exclude slow tests)')
    test_group.add_argument('--slow', action='store_true',
                           help='Run only slow tests')
    test_group.add_argument('--gpu', action='store_true',
                           help='Run only GPU-specific tests')
    test_group.add_argument('--unit', action='store_true',
                           help='Run only unit tests')
    test_group.add_argument('--integration', action='store_true',
                           help='Run only integration tests')
    test_group.add_argument('--performance', action='store_true',
                           help='Run only performance tests')
    
    # Coverage and reporting
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage report')
    
    # Execution options
    parser.add_argument('--parallel', type=int, metavar='N',
                       help='Run tests in parallel using N processes')
    parser.add_argument('--fail-fast', '-x', action='store_true',
                       help='Stop on first test failure')
    
    # Output options
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output')
    output_group.add_argument('--quiet', '-q', action='store_true',
                             help='Quiet output')
    
    parser.add_argument('--capture', choices=['yes', 'no'], default='yes',
                       help='Capture stdout/stderr (default: yes)')
    
    # Dependency checking
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies and exit')
    
    # Pass through additional pytest arguments
    parser.add_argument('--pytest-args', nargs='*',
                       help='Additional arguments to pass to pytest')
    
    # Test paths
    parser.add_argument('test_paths', nargs='*',
                       help='Specific test files or directories to run')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    check_optional_dependencies()
    
    if args.check_deps:
        return 0
    
    # Run the tests
    return run_tests(args)


if __name__ == '__main__':
    sys.exit(main())