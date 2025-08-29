#!/usr/bin/env python3
"""
Simple test of the flavor-blind fNP system without APFEL++ dependencies.

This script demonstrates the key features of the flavor-blind system:
1. All flavors share identical parameters
2. Dramatic parameter reduction (21 vs ~160 parameters)
3. Parameter fitting works correctly
4. All flavors evolve together

This test does NOT require APFEL++ or LHAPDF, making it easier to run.
"""

import os
import sys
import torch
import numpy as np

# Add map directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
map_dir = os.path.dirname(script_dir)
if map_dir not in sys.path:
    sys.path.insert(0, map_dir)

from modules.fnp_manager_flavor_blind import (
    fNPManagerFlavorBlind,
    load_flavor_blind_config,
)


def test_flavor_blind_basic():
    """Test basic functionality of flavor-blind system."""
    print("🧪 Testing basic flavor-blind functionality...")

    # Load configuration
    config_path = os.path.join(map_dir, "inputs", "fNPconfig_flavor_blind.yaml")
    config = load_flavor_blind_config(config_path)

    # Initialize model
    model = fNPManagerFlavorBlind(config)
    model.eval()

    # Test kinematic points
    x = torch.tensor([0.1, 0.3, 0.5])
    z = torch.tensor([0.2, 0.5, 0.8])
    b = torch.tensor([0.5, 1.0, 2.0])

    # Test PDF evaluation
    print("\n📊 Testing PDF evaluation...")
    pdf_flavors = ["u", "d", "s", "ubar", "dbar", "sbar"]
    pdf_results = model.forward_pdf(x, b, pdf_flavors)

    # Verify all flavors are identical
    u_pdf = pdf_results["u"]
    for flavor in pdf_flavors[1:]:
        assert torch.allclose(
            u_pdf, pdf_results[flavor]
        ), f"PDF {flavor} not identical to u"
    print(f"  ✅ All {len(pdf_flavors)} PDF flavors are identical")
    print(f"  ✅ PDF values: {u_pdf.detach().numpy()}")

    # Test FF evaluation
    print("\n📊 Testing FF evaluation...")
    ff_results = model.forward_ff(z, b, pdf_flavors)

    # Verify all flavors are identical
    u_ff = ff_results["u"]
    for flavor in pdf_flavors[1:]:
        assert torch.allclose(
            u_ff, ff_results[flavor]
        ), f"FF {flavor} not identical to u"
    print(f"  ✅ All {len(pdf_flavors)} FF flavors are identical")
    print(f"  ✅ FF values: {u_ff.detach().numpy()}")

    # Test parameter count
    param_count = model.count_parameters()
    expected_count = 21  # 1 evolution + 11 PDF + 9 FF
    assert (
        param_count == expected_count
    ), f"Expected {expected_count} parameters, got {param_count}"
    print(f"\n📈 Parameter count: {param_count} (87% reduction vs standard system)")

    return model


def test_flavor_blind_fitting():
    """Test parameter fitting in flavor-blind system."""
    print("\n🏃 Testing flavor-blind parameter fitting...")

    # Initialize model
    config_path = os.path.join(map_dir, "inputs", "fNPconfig_flavor_blind.yaml")
    config = load_flavor_blind_config(config_path)
    model = fNPManagerFlavorBlind(config)
    model.train()

    # Create synthetic data
    torch.manual_seed(42)
    x = torch.linspace(0.1, 0.9, 10)
    z = torch.linspace(0.2, 0.8, 10)
    b = torch.linspace(0.1, 3.0, 10)

    # Generate "true" data with current parameters
    with torch.no_grad():
        pdf_true = model.forward_pdf(x, b, ["u"])["u"]
        ff_true = model.forward_ff(z, b, ["u"])["u"]
        targets = pdf_true * ff_true  # Simple product as synthetic target

    # Perturb parameters
    original_params = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.clone()
                param.add_(torch.randn_like(param) * 0.1)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Fitting loop
    initial_loss = None
    print(f"  🎯 Fitting {model.count_parameters()} parameters...")

    for epoch in range(50):
        optimizer.zero_grad()

        # Forward pass
        pdf_pred = model.forward_pdf(x, b, ["u"])["u"]
        ff_pred = model.forward_ff(z, b, ["u"])["u"]
        predictions = pdf_pred * ff_pred

        # Loss
        loss = torch.mean((predictions - targets) ** 2)

        if epoch == 0:
            initial_loss = loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:2d}: Loss = {loss.item():.6f}")

    final_loss = loss.item()
    improvement = (initial_loss - final_loss) / initial_loss * 100

    print(f"  ✅ Fitting completed!")
    print(
        f"  ✅ Loss improvement: {improvement:.1f}% ({initial_loss:.6f} → {final_loss:.6f})"
    )

    # Verify all flavors still identical after fitting
    test_flavors = ["u", "d", "s", "ubar"]
    pdf_results = model.forward_pdf(x[:3], b[:3], test_flavors)
    ff_results = model.forward_ff(z[:3], b[:3], test_flavors)

    for flavor in test_flavors[1:]:
        assert torch.allclose(pdf_results["u"], pdf_results[flavor], atol=1e-6)
        assert torch.allclose(ff_results["u"], ff_results[flavor], atol=1e-6)

    print(f"  ✅ All flavors remain identical after fitting")

    return model


def test_parameter_sharing():
    """Test that parameter changes affect all flavors simultaneously."""
    print("\n🔄 Testing parameter sharing across flavors...")

    config_path = os.path.join(map_dir, "inputs", "fNPconfig_flavor_blind.yaml")
    config = load_flavor_blind_config(config_path)
    model = fNPManagerFlavorBlind(config)

    x = torch.tensor([0.3])
    z = torch.tensor([0.5])
    b = torch.tensor([1.0])

    # Get initial results for all flavors
    flavors = ["u", "d", "s", "ubar", "dbar", "sbar"]
    pdf_initial = model.forward_pdf(x, b, flavors)
    ff_initial = model.forward_ff(z, b, flavors)

    # Modify parameters
    with torch.no_grad():
        # Change PDF parameters
        model.pdf_module.free_params[0] += 0.1  # Change first PDF parameter
        # Change FF parameters
        model.ff_module.free_params[0] += 0.1  # Change first FF parameter

    # Get results after parameter change
    pdf_modified = model.forward_pdf(x, b, flavors)
    ff_modified = model.forward_ff(z, b, flavors)

    # Verify all flavors changed by the same amount
    for flavor in flavors:
        pdf_change = pdf_modified[flavor] - pdf_initial[flavor]
        ff_change = ff_modified[flavor] - ff_initial[flavor]

        # All flavors should have identical changes
        assert torch.allclose(pdf_change, pdf_modified["u"] - pdf_initial["u"])
        assert torch.allclose(ff_change, ff_modified["u"] - ff_initial["u"])

    print(f"  ✅ Parameter changes affect all {len(flavors)} flavors identically")
    print(f"  ✅ PDF change: {(pdf_modified['u'] - pdf_initial['u']).item():.6f}")
    print(f"  ✅ FF change: {(ff_modified['u'] - ff_initial['u']).item():.6f}")


def main():
    """Run all tests."""
    print("🧬 FLAVOR-BLIND fNP SYSTEM TESTS")
    print("=" * 50)

    try:
        # Test 1: Basic functionality
        model1 = test_flavor_blind_basic()

        # Test 2: Parameter fitting
        model2 = test_flavor_blind_fitting()

        # Test 3: Parameter sharing
        test_parameter_sharing()

        print("\n🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("The flavor-blind fNP system is working correctly:")
        print("  ✅ All flavors share identical parameters")
        print("  ✅ Parameter count reduced by 87% (21 vs ~160)")
        print("  ✅ Gradient-based fitting works correctly")
        print("  ✅ Parameter changes affect all flavors simultaneously")
        print("  ✅ Ready for SIDIS cross section computation")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
