"""
SIDIS TMD Cross-Section Computation with Minimization

This script extends main.py to include parameter minimization capabilities.
It can be used to fit model parameters to data using an optimizer.

Usage:
    python3 sidis/tests/main_with_minimization.py -c fNPconfig_base_flexible.yaml --events mock_events.dat --epochs 100
"""

import torch
import argparse
import pathlib
import sys
from collections import defaultdict

# This ensures the code only runs when the script is executed directly,
# not when imported as a module.
if __name__ == "__main__":
    # Add parent directory to path so sidis can be imported as a package
    parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    from omegaconf import OmegaConf
    from sidis.model import TrainableModel
    from sidis.model.fnp_base_flexible import ParameterLinkParser
    from sidis.utilities.colors import tcolors

    # Set default tensor dtype to float64 for high precision calculations
    torch.set_default_dtype(torch.float64)

    # Get the directory containing this script
    script_dir = pathlib.Path(__file__).resolve().parent
    rootdir = script_dir.parent  # sidis/ directory
    cards_dir = rootdir.joinpath("cards")

    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="SIDIS TMD minimization and linking verification (flexible config).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--fnp_config",
        "-c",
        type=str,
        default="fNPconfig_base_flavor_blind.yaml",
        help="fNP configuration file name (looked up in cards/ directory). "
        "Default: fNPconfig_base_flavor_blind.yaml",
    )

    parser.add_argument(
        "--events",
        "-e",
        type=str,
        default=None,
        help="Path to events file (relative to sidis/ or absolute). "
        "Default: sidis/toy_events.dat",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of optimization epochs. Default: 50",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for optimizer. Default: 0.1",
    )

    parser.add_argument(
        "--target_scale",
        type=float,
        default=10.0,
        help="Scale factor for target (dummy target = initial_output * scale). "
        "Larger values (e.g. 10) make parameters move more. Default: 10.0",
    )

    parser.add_argument(
        "--save_params",
        type=str,
        default=None,
        help="Path to save optimized parameters. If not specified, parameters are not saved.",
    )

    parser.add_argument(
        "--drive_pdfs_u_to",
        type=float,
        default=None,
        metavar="VALUE",
        help="Add a loss term so the minimizer drives all pdfs.u parameters toward this value (e.g. 1.0). "
        "Linked params (e.g. pdfs.d[0]) will move with u. Only applies to flexible config.",
    )

    parser.add_argument(
        "--drive_pdfs_u_weight",
        type=float,
        default=1.0,
        help="Weight for the drive_pdfs_u_to penalty (loss += weight * sum((u_param - target)^2)). Default: 1.0",
    )

    args = parser.parse_args()

    # Check if config file exists in cards/ directory
    config_path = cards_dir.joinpath(args.fnp_config)

    if not config_path.exists():
        print(
            f"{tcolors.FAIL}Error: Configuration file not found: {config_path}{tcolors.ENDC}"
        )
        print(
            f"\n{tcolors.WARNING}Please ensure the config file exists in the cards/ directory.{tcolors.ENDC}"
        )
        print(f"Available config files in {cards_dir}:")
        if cards_dir.exists():
            for f in sorted(cards_dir.glob("*.yaml")):
                print(f"  - {f.name}")
        else:
            print(f"  {tcolors.FAIL}cards/ directory not found!{tcolors.ENDC}")
        print(
            f"\nUsage: python3 sidis/tests/main_with_minimization.py -c <config_file>\n"
        )
        exit(1)

    print(f"{tcolors.GREEN}Using fNP config: {args.fnp_config}{tcolors.ENDC}\n")

    # Initialize the trainable model
    print(f"{tcolors.BOLDWHITE}Initializing model...{tcolors.ENDC}")
    model = TrainableModel(fnp_config=args.fnp_config)
    print(f"{tcolors.GREEN}Model initialized successfully!{tcolors.ENDC}")

    # Load event data
    if args.events is None:
        events_file = rootdir.joinpath("toy_events.dat")
    else:
        events_file = pathlib.Path(args.events)
        if not events_file.is_absolute():
            events_file = rootdir.joinpath(args.events)

    if not events_file.exists():
        print(
            f"{tcolors.FAIL}Error: Events file not found: {events_file}{tcolors.ENDC}"
        )
        exit(1)

    print(f"{tcolors.BOLDWHITE}Loading events from {events_file}{tcolors.ENDC}")
    events_tensor = torch.load(events_file)
    print(f"Event data shape: {events_tensor.shape}")
    print(f"Number of events: {events_tensor.shape[0]}\n")

    # Run initial forward pass
    print(f"{tcolors.BOLDWHITE}Running initial forward pass...{tcolors.ENDC}")
    with torch.no_grad():
        initial_output = model(events_tensor)
    print(f"Initial output shape: {initial_output.shape}")
    print(
        f"Initial output range: [{initial_output.min().item():.6e}, {initial_output.max().item():.6e}]"
    )
    print(f"Initial output mean: {initial_output.mean().item():.6e}\n")

    # Count and display parameter summary
    print(f"{tcolors.BOLDWHITE}Model Parameters Summary:{tcolors.ENDC}")

    def is_fixed_param_buffer(name: str) -> bool:
        if "fixed_param_" in name and not name.endswith("_params"):
            return True
        if name.endswith(".fixed_params") or name == "fixed_params":
            return True
        return False

    total_params = 0
    trainable_params = 0
    fixed_params = 0

    param_groups = {}
    for name, param in model.named_parameters():
        # Extract module name
        parts = name.split(".")
        if len(parts) >= 2:
            module_name = ".".join(parts[:-1])
        else:
            module_name = "other"

        if module_name not in param_groups:
            param_groups[module_name] = []
        param_groups[module_name].append((name, param, True))
    
    # Fixed parameters (buffers)
    for name, buffer in model.named_buffers():
        if is_fixed_param_buffer(name):
            fixed_params += buffer.numel()
            total_params += buffer.numel()

            # Extract module name
            parts = name.split(".")
            if len(parts) >= 2:
                module_name = ".".join(parts[:-1])
            else:
                module_name = "other"

            if module_name not in param_groups:
                param_groups[module_name] = []
            param_groups[module_name].append((name, buffer, False))

    # Counts
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    # Table
    for module_name, params in sorted(param_groups.items()):
        print(f"\n{tcolors.OKLIGHTBLUE}{'='*125}")
        print(f"Module: {module_name}")
        print(f"{'='*125}{tcolors.ENDC}")
        print(f"{'Parameter Name':<70} {'Shape':<15} {'Trainable':<12} {'Value':<25}")
        print("-" * 125)

        for name, tensor, is_parameter in sorted(params):
            if is_parameter:
                # This is a trainable parameter
                is_trainable = tensor.requires_grad
            else:
                # This is a fixed parameter buffer
                is_trainable = False

            # Format value display
            if tensor.numel() == 1:
                value_str = f"{tensor.data.item():.6f}"
            elif tensor.numel() <= 5:
                # Show full values for small tensors
                value_str = str(tensor.data.detach().cpu().numpy().tolist())
            else:
                # Show summary for large tensors
                value_str = f"min={tensor.data.min().item():.6f}, max={tensor.data.max().item():.6f}"

            short_name = name.split(".")[-1] if "." in name else name
            if short_name.startswith("fixed_param_"):
                try:
                    short_name = f"fixed[{short_name.replace('fixed_param_', '')}]"
                except Exception:
                    pass

            trainable_str = (
                f"{tcolors.GREEN}Yes{tcolors.ENDC}"
                if is_trainable
                else f"{tcolors.WARNING}No (Fixed){tcolors.ENDC}"
            )

            print(
                f"{short_name:<70} {str(list(tensor.shape)):<15} {trainable_str:<25} {value_str[:25]}"
            )

    print(f"\n{tcolors.BOLDWHITE}{'='*125}")
    print(f"Summary:{tcolors.ENDC}")
    print(f"  Total parameters: {total_params:,}")
    print(
        f"  {tcolors.GREEN}Trainable (open to fit): {trainable_params:,}{tcolors.ENDC}"
    )
    print(f"  {tcolors.WARNING}Fixed (not trainable): {fixed_params:,}{tcolors.ENDC}")
    if total_params > 0:
        print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    print(f"{'='*125}\n")

    # Store initial parameters for comparison
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.clone().detach()

    # Build linking info from config (flexible only): reference groups + expression list
    linked_groups = []
    initial_values_by_source = {}
    expression_entries = []
    if hasattr(model, "qcf0") and hasattr(model.qcf0, "fnp_manager"):
        fnp_mgr = model.qcf0.fnp_manager
        reg = getattr(fnp_mgr, "registry", None)
        fnp_config_dict = OmegaConf.load(config_path)
        parser = ParameterLinkParser()
        for param_type in ("pdfs", "ffs"):
            type_config = fnp_config_dict.get(param_type, {})
            for flavor, flavor_cfg in type_config.items():
                if not isinstance(flavor_cfg, dict):
                    continue
                free_mask = flavor_cfg.get("free_mask", [])
                for idx, entry in enumerate(free_mask):
                    parsed = parser.parse_entry(entry, param_type, flavor)
                    if parsed["type"] == "expression":
                        expression_entries.append(
                            (param_type, flavor, idx, parsed["value"])
                        )
        if reg is not None and hasattr(reg, "shared_groups") and reg.shared_groups:
            shared = reg.shared_groups
            source_to_linked = defaultdict(list)
            for key, source in shared.items():
                source_to_linked[tuple(source)].append(tuple(key))
            for source_key, linked_list in source_to_linked.items():
                group_members = [source_key]
                for k in linked_list:
                    if k != source_key:
                        group_members.append(k)
                linked_groups.append((source_key, group_members))
            for source_key, _ in linked_groups:
                param_type, flavor, idx = source_key
                mod_name = "pdf_modules" if param_type == "pdfs" else "ff_modules"
                param_name = f"qcf0.fnp_manager.{mod_name}.{flavor}.free_param_{idx}"
                if param_name in initial_params:
                    initial_values_by_source[source_key] = initial_params[
                        param_name
                    ].clone()

    print(f"{tcolors.GREEN}Stored {len(initial_params)} trainable parameter groups.{tcolors.ENDC}\n")

    # Set up optimizer
    print(f"{tcolors.BOLDWHITE}Setting up optimizer...{tcolors.ENDC}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Optimizer: Adam with learning rate = {args.lr}")

    # Create target (dummy target for demonstration)
    target = initial_output * args.target_scale
    print(f"Target scale factor: {args.target_scale}\n")

    drive_pdfs_u_target = args.drive_pdfs_u_to
    drive_pdfs_u_weight = args.drive_pdfs_u_weight
    pdfs_u_params = []
    if drive_pdfs_u_target is not None:
        if hasattr(model, "qcf0") and hasattr(model.qcf0, "fnp_manager"):
            fnp_mgr = model.qcf0.fnp_manager
            if hasattr(fnp_mgr, "pdf_modules") and "u" in fnp_mgr.pdf_modules:
                pdfs_u_params = list(fnp_mgr.pdf_modules["u"].parameters())
                print(
                    f"{tcolors.GREEN}Driving pdfs.u parameters toward {drive_pdfs_u_target} "
                    f"(penalty weight = {drive_pdfs_u_weight}, {len(pdfs_u_params)} parameter tensors).{tcolors.ENDC}\n"
                )
            else:
                print(
                    f"{tcolors.WARNING}--drive_pdfs_u_to ignored: no pdf_modules.u (not flexible config?).{tcolors.ENDC}\n"
                )
        else:
            print(
                f"{tcolors.WARNING}--drive_pdfs_u_to ignored: model has no qcf0.fnp_manager.pdf_modules.{tcolors.ENDC}\n"
            )

    # Perform minimization
    print(
        f"{tcolors.BOLDWHITE}Performing minimization ({args.epochs} epochs)...{tcolors.ENDC}"
    )
    print("=" * 80)

    losses = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        # Forward pass
        output = model(events_tensor)

        # Compute loss (MSE between output and target)
        loss = torch.nn.functional.mse_loss(output, target)

        # Optional: add penalty to drive pdfs.u parameters toward target value
        if pdfs_u_params:
            penalty = 0.0
            for p in pdfs_u_params:
                if p.requires_grad:
                    penalty = penalty + ((p - drive_pdfs_u_target) ** 2).sum()
            loss = loss + drive_pdfs_u_weight * penalty

        losses.append(loss.item())

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Print progress
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{args.epochs}: Loss = {loss.item():.6e}")

    print("=" * 80)
    print(f"{tcolors.GREEN}Minimization completed!{tcolors.ENDC}")
    print(f"Initial loss: {losses[0]:.6e}")
    print(f"Final loss: {losses[-1]:.6e}")
    if losses[0] > 0:
        reduction = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"Loss reduction: {reduction:.2f}%\n")

    # Run final forward pass
    print(f"{tcolors.BOLDWHITE}Running final forward pass...{tcolors.ENDC}")
    with torch.no_grad():
        final_output = model(events_tensor)
    print(f"Final output shape: {final_output.shape}")
    print(
        f"Final output range: [{final_output.min().item():.6e}, {final_output.max().item():.6e}]"
    )
    print(f"Final output mean: {final_output.mean().item():.6e}\n")

    # Verify parameter changes
    print(f"{tcolors.BOLDWHITE}Verifying parameter changes...{tcolors.ENDC}")
    params_changed = 0
    max_change = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_params:
            initial = initial_params[name]
            current = param.detach()
            if initial.shape == current.shape:
                change = torch.abs(current - initial).max().item()
                if change > 1e-10:
                    params_changed += 1
                    max_change = max(max_change, change)

    print(
        f"  Trainable parameter groups changed: {params_changed}/{len(initial_params)}"
    )
    print(f"  Maximum parameter change: {max_change:.6e}")

    # Compare outputs
    output_diff = torch.abs(final_output - initial_output).mean().item()
    relative_change = output_diff / initial_output.mean().item() * 100
    print(f"\n  Mean absolute output difference: {output_diff:.6e}")
    print(f"  Relative output change: {relative_change:.2f}%")

    # Linking verification (flexible config only): references + expressions
    ref_ok = True
    expr_ok = True
    if hasattr(model, "qcf0") and hasattr(model.qcf0, "fnp_manager"):
        fnp_mgr = model.qcf0.fnp_manager
        reg = getattr(fnp_mgr, "registry", None)
        evaluator = getattr(fnp_mgr, "evaluator", None)
        if linked_groups or expression_entries:
            print(f"\n{tcolors.BOLDWHITE}Linking verification (from config):{tcolors.ENDC}")
            print("=" * 80)
        # Reference links: same tensor, same value
        for source_key, group_members in linked_groups:
            param_type_s, flavor_s, idx_s = source_key
            param = reg.get_parameter(param_type_s, flavor_s, idx_s) if reg else None
            if param is None:
                continue
            current_val = param.data.item()
            label_s = f"{param_type_s}.{flavor_s}[{idx_s}]"
            members_str = [f"{pt}.{fl}[{i}]" for pt, fl, i in group_members]
            initial_val = initial_values_by_source.get(source_key)
            initial_val = initial_val.item() if initial_val is not None else None
            same_param = True
            for pt, fl, idx in group_members:
                p = reg.get_parameter(pt, fl, idx) if reg else None
                if p is not None and (id(p) != id(param) or p.data.item() != current_val):
                    same_param = False
                    ref_ok = False
            status = f"{tcolors.GREEN}OK{tcolors.ENDC}" if same_param else f"{tcolors.FAIL}MISMATCH{tcolors.ENDC}"
            val_str = f"  {initial_val:.6f} -> {current_val:.6f}" if initial_val is not None else f"  {current_val:.6f}"
            print(f"  Ref {label_s}: {', '.join(members_str)}  {val_str}  {status}")
        # Expression links: effective value == evaluator.evaluate(expr)
        for param_type, flavor, idx, expr in expression_entries:
            if evaluator is None:
                expr_ok = False
                print(f"  Expr {param_type}.{flavor}[{idx}] = {expr}: no evaluator")
                continue
            mod = fnp_mgr.pdf_modules[flavor] if param_type == "pdfs" else fnp_mgr.ff_modules[flavor]
            params_t = mod.get_params_tensor()
            effective = params_t[idx].item()
            try:
                expected = evaluator.evaluate(expr, param_type, flavor).item()
                match = abs(effective - expected) < 1e-5
                if not match:
                    expr_ok = False
                status = f"{tcolors.GREEN}OK{tcolors.ENDC}" if match else f"{tcolors.FAIL}MISMATCH{tcolors.ENDC}"
                print(f"  Expr {param_type}.{flavor}[{idx}] = {expr}: value {effective:.6f} (expected {expected:.6f})  {status}")
            except Exception as e:
                expr_ok = False
                print(f"  Expr {param_type}.{flavor}[{idx}] = {expr}: {tcolors.FAIL}error {e}{tcolors.ENDC}")
        if linked_groups or expression_entries:
            if ref_ok and expr_ok:
                print(f"{tcolors.GREEN}  All linkings verified.{tcolors.ENDC}\n")
            else:
                print(f"{tcolors.FAIL}  Some linkings failed.{tcolors.ENDC}\n")

    # Save parameters if requested
    if args.save_params is not None:
        save_path = pathlib.Path(args.save_params)
        if not save_path.is_absolute():
            save_path = script_dir.joinpath(args.save_params)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save only trainable parameters
        trainable_state_dict = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        torch.save(trainable_state_dict, save_path)
        print(
            f"\n{tcolors.GREEN}Saved optimized parameters to: {save_path}{tcolors.ENDC}"
        )

    # Summary
    print(f"\n{tcolors.BOLDWHITE}{'='*80}{tcolors.ENDC}")
    print(f"{tcolors.GREEN}Minimization workflow completed successfully!{tcolors.ENDC}")
    print(f"{tcolors.BOLDWHITE}{'='*80}{tcolors.ENDC}")
