"""
Test refactored code on main against reference from FIXED original code.
"""
import torch
import pickle
import numpy as np

# NOTE: torch.set_default_dtype is now automatically set by the model from config.yaml
from sidis.model import TrainableModel

print("="*70)
print("TESTING REFACTORED CODE vs FIXED ORIGINAL REFERENCE")
print("="*70)

# Load reference
with open('reference_clean_outputs.pkl', 'rb') as f:
    ref = pickle.load(f)

events = ref['events']
original_output = ref['outputs']
expt_setup = ref['expt_setup']
rs = ref['rs']

print("\nReference info:", ref['note'])
print(f"\nTest events: {len(events)}")

# Create refactored model
print("\n Creating refactored model...")
model = TrainableModel()
print("✓ Refactored model created")

# Compute outputs
print("\nComputing with refactored code...")
refactored_output = model(events, expt_setup, rs)

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"{'Event':<8} {'Original':<15} {'Refactored':<15} {'Abs Diff':<12} {'Rel Diff':<12} {'Status':<8}")
print("-"*70)

all_match = True
max_abs_diff = 0.0
max_rel_diff = 0.0
tol = 1e-10

for i in range(len(events)):
    orig = original_output[i].item()
    refac = refactored_output[i].item()
    abs_diff = abs(orig - refac)
    rel_diff = abs_diff / abs(orig) if abs(orig) > 1e-15 else 0.0
    
    max_abs_diff = max(max_abs_diff, abs_diff)
    max_rel_diff = max(max_rel_diff, rel_diff)
    
    match = abs_diff < tol
    all_match = all_match and match
    status = "✓" if match else "✗"
    
    print(f"{i:<8} {orig:<15.6e} {refac:<15.6e} {abs_diff:<12.2e} {rel_diff:<12.2e} {status:<8}")

print("-"*70)
print(f"Max absolute diff: {max_abs_diff:.2e}")
print(f"Max relative diff: {max_rel_diff:.2e}")
print(f"Tolerance: {tol:.2e}")

print("\n" + "="*70)
if all_match:
    print("✓✓✓ VALIDATION PASSED ✓✓✓")
    print("="*70)
    print("Refactored code produces IDENTICAL results!")
    print("The refactoring is SUCCESSFUL and CORRECT.")
else:
    print("✗✗✗ VALIDATION FAILED ✗✗✗")
    print("="*70)
    print("Results differ. Investigation needed.")

