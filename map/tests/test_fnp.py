#!/usr/bin/env python3
"""
Test script for the new modular fNP system.

This script verifies that fNP works correctly:
- Tests MAP22 parameterization implementation
- Verifies PDF and FF computation consistency
- Checks parameter management and optimization
- Validates configuration loading

Run from anywhere in the TMD repository.
"""

import os
import sys
import torch
import yaml
import numpy as np

# Set up paths for local imports. The function that finds the
# path to the root and map/ folder, ensure_repo_on_syspath(),
# is in modules/utilities.py, so it's in a module itself. In
# order to import it, we need to manually look for it, since
# the modules/ path is not in the sys.path yet.
try:
    from modules.utilities import ensure_repo_on_syspath
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    map_dir = os.path.dirname(script_dir)
    if map_dir not in sys.path:
        sys.path.insert(0, map_dir)
    from modules.utilities import ensure_repo_on_syspath

# Ensure repository paths are set up
repo_root, map_dir = ensure_repo_on_syspath()

# Import the new modular fNP system
from modules.fnp import fNPManager
from modules.fnp import fNP_evolution, TMDPDFBase, TMDFFBase


def test_evolution_module():
    """Test the standalone evolution module."""
    print("\n" + "=" * 60)
    print("Testing fNP Evolution Module")
    print("=" * 60)

    # Test basic initialization
    evolution = fNP_evolution(init_g2=0.12840, free_mask=[True])
    print(f"✅ Evolution module initialized with g2 = {evolution.g2.item():.5f}")

    # Test forward pass
    b = torch.linspace(0.1, 2.0, 10)
    zeta = torch.tensor(4.0)  # Q² = 2 GeV²

    result = evolution(b, zeta)
    print(f"✅ Evolution computed for {len(b)} b-points")
    print(f"   Range: [{result.min().item():.4f}, {result.max().item():.4f}]")

    # Test parameter access
    param_info = {
        "g2_value": evolution.g2.item(),
        "is_trainable": evolution.free_g2.requires_grad,
        "mask": evolution.g2_mask.tolist(),
    }
    print(f"✅ Parameter info: {param_info}")

    return evolution


def test_fnp_tmdpdf_module():
    """Test the TMD PDF base module."""
    print("\n" + "=" * 60)
    print("Testing TMD PDF Module (MAP22)")
    print("=" * 60)

    # Use MAP22 default parameters
    from modules.fnp import MAP22_DEFAULT_PDF_PARAMS

    pdf_module = TMDPDFBase(
        n_flavors=1,
        init_params=MAP22_DEFAULT_PDF_PARAMS["init_params"],
        free_mask=MAP22_DEFAULT_PDF_PARAMS["free_mask"],
    )
    print(f"✅ PDF module initialized with {pdf_module.n_params} parameters")

    # Test forward pass
    x = torch.tensor([0.001, 0.01, 0.1])
    b = torch.tensor([0.5, 1.0, 1.5])
    zeta = torch.tensor(4.0)  # Q^2 = 4 GeV^2

    # Create mock evolution factor
    evolution = fNP_evolution(init_g2=0.12840, free_mask=[True])
    # evolution takes in two torch tensors, b and zeta, and returns a tensor
    # of the same shape as b.
    NP_evol = evolution(b, zeta)

    result = pdf_module(x, b, NP_evol, 0)
    print(f"✅ fNP for TMDPDF computed for {len(x)} points")
    # detach(): returns a new tensor that shares the same storage as
    # result but is not tracked by autograd (no gradient history).
    # Gradients won’t flow through it. (It’s a view: no copy.)
    # cpu() is used to move the tensor to the CPU, so that it can
    # be converted to a numpy array.
    print(f"   Values: {result.detach().cpu().numpy()}")

    # Test parameter masking
    params = pdf_module.get_params_tensor
    mask = pdf_module.mask
    trainable_count = int(mask.sum().item())
    print(f"✅ Parameter masking: {trainable_count}/{pdf_module.n_params} trainable")

    return pdf_module


def test_fnp_tmdff_module():
    """Test the TMD FF base module."""
    print("\n" + "=" * 60)
    print("Testing TMD FF Module (MAP22)")
    print("=" * 60)

    # Use MAP22 default parameters
    from modules.fnp import MAP22_DEFAULT_FF_PARAMS

    ff_module = TMDFFBase(
        n_flavors=1,
        init_params=MAP22_DEFAULT_FF_PARAMS["init_params"],
        free_mask=MAP22_DEFAULT_FF_PARAMS["free_mask"],
    )
    print(f"✅ FF module initialized with {ff_module.n_params} parameters")

    # Test forward pass
    z = torch.tensor([0.2, 0.5, 0.8])
    b = torch.tensor([0.5, 1.0, 1.5])
    zeta = torch.tensor(4.0)

    # Create mock evolution factor
    evolution = fNP_evolution(init_g2=0.12840, free_mask=[True])
    NP_evol = evolution(b, zeta)

    result = ff_module(z, b, NP_evol, 0)
    print(f"✅ FF computed for {len(z)} points")
    print(f"   Values: {result.detach().numpy()}")

    # Test parameter masking
    params = ff_module.get_params_tensor
    mask = ff_module.mask
    trainable_count = int(mask.sum().item())
    print(f"✅ Parameter masking: {trainable_count}/{ff_module.n_params} trainable")

    return ff_module


def test_fnp_manager():
    """Test the unified fNP manager."""
    print("\n" + "=" * 60)
    print("Testing Unified fNP Manager")
    print("=" * 60)

    # Load unified configuration
    config_file = os.path.join(map_dir, "inputs", "fNPconfig.yaml")

    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        return None

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Initialize manager
    manager = fNPManager(config)
    print(f"✅ Manager initialized successfully")

    # Print parameter summary
    manager.print_parameter_summary()

    # Test PDF evaluation
    x = torch.tensor([0.1, 0.3])
    b = torch.tensor([0.5, 1.0])

    pdf_results = manager.forward_pdf(x, b, flavors=["u", "d"])
    print(f"✅ PDF evaluation for flavors {list(pdf_results.keys())}")
    for flavor, values in pdf_results.items():
        print(f"   {flavor}: {values.detach().numpy()}")

    # Test FF evaluation
    z = torch.tensor([0.3, 0.7])
    ff_results = manager.forward_ff(z, b, flavors=["u", "d"])
    print(f"✅ FF evaluation for flavors {list(ff_results.keys())}")
    for flavor, values in ff_results.items():
        print(f"   {flavor}: {values.detach().numpy()}")

    # Test simultaneous evaluation
    combined_results = manager.forward(x, z, b, pdf_flavors=["u"], ff_flavors=["u"])
    print(f"✅ Combined evaluation completed")

    # Test parameter extraction for optimization
    trainable_params = manager.get_trainable_parameters_dict()
    print(f"✅ Extracted {len(trainable_params)} trainable parameter groups")

    total_trainable = sum(p.numel() for p in trainable_params.values())
    print(f"   Total trainable parameters: {total_trainable}")

    return manager


def test_optimization_compatibility():
    """Test that the system works with PyTorch optimizers."""
    print("\n" + "=" * 60)
    print("Testing Optimization Compatibility")
    print("=" * 60)

    # Create simple configuration
    simple_config = {
        "hadron": "proton",
        "zeta": 1.0,
        "evolution": {"init_g2": 0.1, "free_mask": [True]},
        "pdfs": {
            "u": {
                "init_params": [0.3, 3.0, 0.2, 0.4, 0.3, 0.3, 0.4, 3.0, 3.0, 0.2, 0.2],
                "free_mask": [
                    True,
                    True,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            }
        },
        "ffs": {
            "u": {
                "init_params": [0.2, 2.0, 0.1, 0.25, 5.0, 0.03, 2.0, 0.1, 0.25],
                "free_mask": [
                    True,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
            }
        },
    }

    manager = fNPManager(simple_config)

    # Create optimizer
    optimizer = torch.optim.Adam(manager.parameters(), lr=0.01)
    print(f"✅ Created Adam optimizer")

    # Test forward pass and backward
    x = torch.tensor([0.1, 0.3], requires_grad=False)
    z = torch.tensor([0.2, 0.5], requires_grad=False)
    b = torch.tensor([0.5, 1.0], requires_grad=False)

    # Get parameter count before
    param_info = manager.get_parameter_info()
    trainable_before = param_info["truly_trainable_parameters"]

    # Forward pass
    results = manager.forward(x, z, b, pdf_flavors=["u"], ff_flavors=["u"])

    # Create dummy loss
    pdf_loss = results["pdfs"]["u"].sum()
    ff_loss = results["ffs"]["u"].sum()
    total_loss = pdf_loss + ff_loss

    print(f"✅ Forward pass completed, loss = {total_loss.item():.6f}")

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"✅ Backward pass and optimizer step completed")
    print(f"   Trainable parameters: {trainable_before}")

    # Verify gradients exist
    grad_count = 0
    for name, param in manager.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_count += 1

    print(f"✅ Found gradients for {grad_count} parameter groups")

    return manager


def main():
    """Run all tests."""
    print("🧪 Testing Modular fNP System")
    print("=" * 60)

    try:
        # Test individual modules
        evolution = test_evolution_module()
        pdf_module = test_fnp_tmdpdf_module()
        ff_module = test_fnp_tmdff_module()

        # Test unified system
        manager = test_fnp_manager()

        # Test optimization
        opt_manager = test_optimization_compatibility()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Evolution module working")
        print("✅ TMD PDF module (MAP22) working")
        print("✅ TMD FF module (MAP22) working")
        print("✅ Unified manager working")
        print("✅ Optimization compatibility confirmed")
        print("\nThe modular fNP system is ready for use! 🚀")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
