#!/usr/bin/env python
"""
V2 Ablation Study - Quick Runner

This script runs the minimum necessary tests to complete the ablation study:
1. V2-baseline: Frame-independent only
2. V2-improved: Frame-independent + PositionalEncoding + TimestepEmbedder

Combined with existing results:
- V1 (trans_enc): disp_ratio=0.00 (collapse)
- V2-pos: disp_ratio=1.05 (already tested)

This will give you a complete ablation study table for the paper!
"""

import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)
    print(f"Command: {cmd}")
    print()
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        return False
    
    print(f"\n✅ SUCCESS: {description}")
    print(f"Time: {elapsed/60:.1f} minutes")
    return True

def main():
    print("=" * 70)
    print("V2 ABLATION STUDY - QUICK RUNNER")
    print("=" * 70)
    print()
    print("Current status:")
    print("  ✓ V1 (trans_enc): disp_ratio=0.00 (collapse)")
    print("  ✓ V2-pos: disp_ratio=1.05 (already tested)")
    print()
    print("Will test:")
    print("  → V2-baseline: frame-independent only")
    print("  → V2-improved: frame-independent + both components")
    print()
    print("Total estimated time: ~1 hour")
    print("=" * 70)
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    
    if not os.path.exists("test_v2_improved_FIXED.py"):
        print("❌ ERROR: test_v2_improved_FIXED.py not found!")
        print("Please make sure the test script is in the current directory.")
        sys.exit(1)
    
    if not os.path.exists("signwriting_animation/diffusion/core/models.py"):
        print("❌ ERROR: models.py not found!")
        print("Please run:")
        print("  cp models_v2_improved.py signwriting_animation/diffusion/core/models.py")
        sys.exit(1)
    
    print("✅ All prerequisites satisfied!")
    
    # Ask for confirmation
    print("\n" + "=" * 70)
    response = input("Ready to start? This will take ~1 hour. (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Run tests
    tests = [
        ("python test_v2_improved_FIXED.py --version baseline --epochs 500", 
         "V2-baseline (30 min)"),
        ("python test_v2_improved_FIXED.py --version improved --epochs 500", 
         "V2-improved (30 min)"),
    ]
    
    total_start = time.time()
    results = []
    
    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))
        
        if not success:
            print("\n" + "=" * 70)
            print("STOPPING DUE TO ERROR")
            print("=" * 70)
            break
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    for desc, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {desc}")
    
    print()
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print()
    
    if all(success for _, success in results):
        print("=" * 70)
        print("🎉 ABLATION STUDY COMPLETE!")
        print("=" * 70)
        print()
        print("Results saved in:")
        print("  - logs/v2_improved_baseline/")
        print("  - logs/v2_improved_improved/")
        print()
        print("You now have complete data:")
        print("  ✓ V1 (trans_enc): disp_ratio=0.00")
        print("  ✓ V2-baseline: disp_ratio=? (just tested)")
        print("  ✓ V2-pos: disp_ratio=1.05")
        print("  ✓ V2-improved: disp_ratio=? (just tested)")
        print()
        print("Next steps:")
        print("  1. Check displacement ratios in test outputs")
        print("  2. Compare pose files visually")
        print("  3. Create ablation table for paper")
        print("=" * 70)
    else:
        print("=" * 70)
        print("⚠️  SOME TESTS FAILED")
        print("=" * 70)
        print("Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()