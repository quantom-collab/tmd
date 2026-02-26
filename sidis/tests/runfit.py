"""
runfit.py - Fit fNP simple model parameters to target cross sections.

Loads cross sections from outs/cross_section_output.pt (or .yaml),
loads kinematic events from inputs/mock_events_1000.pt, randomizes fNP
parameters within their bounds, then minimizes a log-space MSE loss using
the PyTorch L-BFGS optimizer to recover the original parameter values.

Loss: mean( (log|pred| - log|target|)^2 )

Usage (from repo root):
    python sidis/tests/runfit.py
    python sidis/tests/runfit.py --seed 99 --epochs 50
    python sidis/tests/runfit.py --cross_section outs/cross_section_output.pt \\
                                 --events inputs/mock_events_1000.pt
    python sidis/tests/runfit.py --use_embedded_events  # use events in .pt file

NOTE: the default cross section output was generated from mock_events_100.dat
(100 points), while the default events file mock_events_1000.pt has 1000 points.
If you see a count-mismatch error, regenerate first:
    python sidis/tests/run_cross_section.py -c fNPconfig_simple.yaml \\
                                            -e inputs/mock_events_1000.pt
"""

import torch
import argparse
import pathlib
import sys

if __name__ == "__main__":

    parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    from omegaconf import OmegaConf
    from sidis.model import TrainableModel
    from sidis.utilities.colors import tcolors

    torch.set_default_dtype(torch.float64)

    script_dir = pathlib.Path(__file__).resolve().parent
    rootdir = script_dir.parent
    cards_dir = rootdir / "cards"

    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Fit fNP simple model parameters to target cross sections (L-BFGS).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python sidis/tests/runfit.py
  python sidis/tests/runfit.py --seed 7 --epochs 100
  python sidis/tests/runfit.py --use_embedded_events
  python sidis/tests/runfit.py --cross_section outs/cross_section_output.pt \\
                               --events inputs/mock_events_1000.pt""",
    )
    parser.add_argument(
        "--cross_section",
        "-cs",
        type=str,
        default=None,
        help="Path to cross section file (.pt or .yaml). Default: outs/cross_section_output.pt",
    )
    parser.add_argument(
        "--events",
        "-e",
        type=str,
        default=None,
        help="Path to events file (.pt). Default: inputs/mock_events_1000.pt",
    )
    parser.add_argument(
        "--use_embedded_events",
        action="store_true",
        default=False,
        help=(
            "Use the events tensor stored inside the .pt cross section file "
            "instead of a separate events file. Guarantees count consistency."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="fNPconfig_simple.yaml",
        help="fNP config file (in sidis/cards/). Default: fNPconfig_simple.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for parameter randomization. Default: 42",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of L-BFGS outer iterations. Default: 50",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        help="L-BFGS learning rate. Default: 1.0",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=20,
        help="L-BFGS max inner iterations per outer step. Default: 20",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Resolve file paths
    # -------------------------------------------------------------------------
    config_path = cards_dir / args.config
    if not config_path.exists():
        print(f"{tcolors.FAIL}Error: config file not found: {config_path}{tcolors.ENDC}")
        exit(1)

    if args.cross_section is None:
        cs_path = script_dir / "outs" / "cross_section_output.pt"
    else:
        cs_path = pathlib.Path(args.cross_section)
        if not cs_path.is_absolute():
            cs_path = script_dir / cs_path

    if not cs_path.exists():
        print(f"{tcolors.FAIL}Error: cross section file not found: {cs_path}{tcolors.ENDC}")
        print(
            f"{tcolors.WARNING}Generate it with:\n"
            f"  python sidis/tests/run_cross_section.py -c {args.config} "
            f"-e inputs/mock_events_1000.pt{tcolors.ENDC}"
        )
        exit(1)

    if args.events is None:
        events_path = script_dir / "inputs" / "mock_events_1000.pt"
    else:
        events_path = pathlib.Path(args.events)
        if not events_path.is_absolute():
            events_path = script_dir / events_path

    # -------------------------------------------------------------------------
    # Load cross sections
    # -------------------------------------------------------------------------
    print(f"\n{tcolors.BOLDWHITE}Loading cross sections from: {cs_path}{tcolors.ENDC}")
    suffix = cs_path.suffix.lower()

    embedded_events = None

    if suffix == ".pt":
        cs_data = torch.load(cs_path)
        if isinstance(cs_data, dict) and "cross_section" in cs_data:
            target = cs_data["cross_section"]
            embedded_events = cs_data.get("events", None)
        else:
            target = cs_data
    elif suffix in (".yaml", ".yml"):
        import yaml

        with open(cs_path) as f:
            cs_dict = yaml.safe_load(f)
        rows = cs_dict["data"]
        target = torch.tensor(
            [row["cross_section"] for row in rows], dtype=torch.float64
        )
        cols = cs_dict.get("kinematic_columns", ["x", "PhT", "Q", "z"])
        embedded_events = torch.tensor(
            [[row[c] for c in cols] for row in rows], dtype=torch.float64
        )
    else:
        print(
            f"{tcolors.FAIL}Error: unsupported file format '{suffix}'. "
            f"Use .pt or .yaml.{tcolors.ENDC}"
        )
        exit(1)

    print(f"  Cross section points: {target.shape[0]}")
    print(f"  Range: [{target.min().item():.4e}, {target.max().item():.4e}]")

    # -------------------------------------------------------------------------
    # Load events
    # -------------------------------------------------------------------------
    if args.use_embedded_events:
        if embedded_events is None:
            print(
                f"{tcolors.FAIL}Error: --use_embedded_events requested but no events "
                f"found in {cs_path}{tcolors.ENDC}"
            )
            exit(1)
        events_tensor = embedded_events
        print(
            f"\n{tcolors.BOLDWHITE}Using events embedded in cross section file "
            f"(shape: {events_tensor.shape}).{tcolors.ENDC}"
        )
    else:
        if not events_path.exists():
            print(
                f"{tcolors.FAIL}Error: events file not found: {events_path}\n"
                f"Use --use_embedded_events to use events stored in the .pt file, "
                f"or provide a matching events file via --events.{tcolors.ENDC}"
            )
            exit(1)
        events_tensor = torch.load(events_path)
        print(
            f"\n{tcolors.BOLDWHITE}Loaded events from: {events_path} "
            f"(shape: {events_tensor.shape}).{tcolors.ENDC}"
        )

    # -------------------------------------------------------------------------
    # Count consistency check
    # -------------------------------------------------------------------------
    n_events = events_tensor.shape[0]
    n_cs = target.shape[0]

    if n_events != n_cs:
        print(
            f"\n{tcolors.FAIL}[ERROR] Count mismatch: events has {n_events} points, "
            f"cross sections has {n_cs} points.{tcolors.ENDC}"
        )
        print(
            f"{tcolors.WARNING}The events file and cross section file must have the "
            f"same number of kinematic points.\n\n"
            f"Options:\n"
            f"  1. Use --use_embedded_events to use the {n_cs}-point events stored in "
            f"the cross section file.\n"
            f"  2. Regenerate cross sections for your events file:\n"
            f"       python sidis/tests/run_cross_section.py -c {args.config} "
            f"-e inputs/mock_events_1000.pt\n"
            f"     then re-run this script.{tcolors.ENDC}"
        )
        exit(1)

    print(
        f"  {tcolors.GREEN}Count check passed: {n_events} events / {n_cs} cross section "
        f"points.{tcolors.ENDC}"
    )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def get_physical_params(mdl) -> dict:
        """
        Return a dict of physical parameter values evaluated by each flavor module.

        Keys: 'pdfs.<flavor>', 'ffs.<flavor>', 'evolution'.
        Expression-linked params (e.g. ffs.u[0] = pdfs.u[0]+pdfs.u[1]) are
        evaluated at their current values, so truth and fit are directly comparable.
        """
        result = {}
        fnp_mgr = mdl.qcf0.fnp_manager
        for flavor, mod in fnp_mgr.pdf_modules.items():
            with torch.no_grad():
                p = mod.get_params_tensor()
            result[f"pdfs.{flavor}"] = [p[i].item() for i in range(p.numel())]
        for flavor, mod in fnp_mgr.ff_modules.items():
            with torch.no_grad():
                p = mod.get_params_tensor()
            result[f"ffs.{flavor}"] = [p[i].item() for i in range(p.numel())]
        evo = fnp_mgr.evolution
        g2_val = evo.g2.item() if hasattr(evo, "g2") else evo.free_g2.item()
        result["evolution"] = [g2_val]
        return result

    def print_param_table(truth: dict, current: dict, title: str) -> None:
        col_w = 80
        print(f"\n{tcolors.BOLDWHITE}{title}{tcolors.ENDC}")
        header = f"  {'Parameter':<32} {'Truth':>12} {'Current':>12} {'Diff':>12} {'Rel.Diff':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key in sorted(truth.keys()):
            t_vals = truth[key]
            c_vals = current.get(key, [None] * len(t_vals))
            for idx, (tv, cv) in enumerate(zip(t_vals, c_vals)):
                pname = f"{key}[{idx}]"
                if tv is not None and cv is not None:
                    diff = cv - tv
                    rel = diff / tv if abs(tv) > 1e-12 else float("nan")
                    print(
                        f"  {pname:<32} {tv:>12.6f} {cv:>12.6f} {diff:>+12.6f} {rel:>+10.4f}"
                    )
                else:
                    print(f"  {pname:<32} {tv!s:>12} {'N/A':>12}")

    # -------------------------------------------------------------------------
    # Initialize model and capture truth parameter values
    # -------------------------------------------------------------------------
    print(f"\n{tcolors.BOLDWHITE}Initializing model ({args.config})...{tcolors.ENDC}")
    model = TrainableModel(fnp_config=args.config)
    print(f"{tcolors.GREEN}Model initialized.{tcolors.ENDC}")

    fnp_config_dict = OmegaConf.load(config_path)
    combo = fnp_config_dict.get("combo", "")
    if combo != "simple":
        print(
            f"{tcolors.FAIL}Error: runfit.py requires combo=simple in the config. "
            f"Got: combo={combo!r}{tcolors.ENDC}"
        )
        exit(1)

    # Truth = physical param values at the config's init_params (before randomization).
    # For expression params (e.g. ffs.u[0] = pdfs.u[0]+pdfs.u[1]) this evaluates the
    # expression, so truth matches what was actually used to generate the cross sections.
    truth_params = get_physical_params(model)
    print(f"\n{tcolors.OKLIGHTBLUE}Truth parameters (config init_params / expressions):{tcolors.ENDC}")
    for key, vals in sorted(truth_params.items()):
        vstr = ", ".join(f"{v:.6f}" for v in vals)
        print(f"  {key:<32} [{vstr}]")

    # -------------------------------------------------------------------------
    # Randomize fNP parameters within bounds
    # -------------------------------------------------------------------------
    fnp_mgr = model.qcf0.fnp_manager
    if hasattr(fnp_mgr, "randomize_params_in_bounds"):
        fnp_mgr.randomize_params_in_bounds(seed=args.seed)
        print(
            f"\n{tcolors.GREEN}Randomized fNP parameters within bounds "
            f"(seed={args.seed}).{tcolors.ENDC}"
        )
    else:
        print(
            f"\n{tcolors.WARNING}fnp_manager has no randomize_params_in_bounds; "
            f"using config init values.{tcolors.ENDC}"
        )

    random_params = get_physical_params(model)
    print(f"\n{tcolors.OKLIGHTBLUE}Initial parameters after randomization:{tcolors.ENDC}")
    for key, vals in sorted(random_params.items()):
        vstr = ", ".join(f"{v:.6f}" for v in vals)
        print(f"  {key:<32} [{vstr}]")

    # -------------------------------------------------------------------------
    # Initial forward pass (informational)
    # -------------------------------------------------------------------------
    print(f"\n{tcolors.BOLDWHITE}Initial forward pass...{tcolors.ENDC}")
    with torch.no_grad():
        initial_pred = model(events_tensor)
    print(
        f"  Prediction range: [{initial_pred.min().item():.4e}, "
        f"{initial_pred.max().item():.4e}]"
    )
    print(
        f"  Target range:     [{target.min().item():.4e}, {target.max().item():.4e}]"
    )

    # -------------------------------------------------------------------------
    # Set up L-BFGS optimizer
    # -------------------------------------------------------------------------
    # Evolution g2 is the only param not handled by sigmoid reparametrization,
    # so it may need clamping to its bounds after each gradient step.
    clamp_after_step = (
        hasattr(fnp_mgr, "get_trainable_bounds") and bool(fnp_mgr.get_trainable_bounds())
    )

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=args.lr,
        max_iter=args.max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    # Log-space MSE: mean( (log|pred| - log|target|)^2 )
    # Floor applied before log to guard against zeros / near-zero values.
    EPS_LOG = 1e-40
    target_log = torch.log(target.abs().clamp(min=EPS_LOG))

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        pred = model(events_tensor)
        pred_log = torch.log(pred.abs().clamp(min=EPS_LOG))
        loss = torch.mean((pred_log - target_log) ** 2)
        loss.backward()
        return loss

    print(
        f"\n{tcolors.BOLDWHITE}Minimizing log-MSE with L-BFGS "
        f"({args.epochs} outer iters × max_iter={args.max_iter})...{tcolors.ENDC}"
    )
    print("=" * 80)

    losses = []
    report_every = max(1, args.epochs // 10)

    for epoch in range(args.epochs):
        loss_val = optimizer.step(closure)

        if clamp_after_step:
            fnp_mgr.clamp_parameters_to_bounds()

        loss_item = loss_val.item() if hasattr(loss_val, "item") else float(loss_val)
        losses.append(loss_item)

        if (epoch + 1) % report_every == 0 or epoch == 0:
            print(f"  Iter {epoch + 1:4d}/{args.epochs}:  log-MSE = {loss_item:.6e}")

    print("=" * 80)
    print(f"{tcolors.GREEN}Minimization complete.{tcolors.ENDC}")
    print(f"  Initial loss : {losses[0]:.6e}")
    print(f"  Final loss   : {losses[-1]:.6e}")
    if losses[0] > 0:
        reduction = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"  Loss reduction: {reduction:.2f}%")

    # -------------------------------------------------------------------------
    # Final prediction
    # -------------------------------------------------------------------------
    with torch.no_grad():
        final_pred = model(events_tensor)
    print(
        f"\n  Final prediction range: [{final_pred.min().item():.4e}, "
        f"{final_pred.max().item():.4e}]"
    )

    # -------------------------------------------------------------------------
    # Parameter comparison tables
    # -------------------------------------------------------------------------
    final_params = get_physical_params(model)

    print_param_table(
        truth_params,
        random_params,
        "Parameter comparison: Truth vs Initial (randomized)",
    )
    print_param_table(
        truth_params,
        final_params,
        "Parameter comparison: Truth vs Fitted",
    )

    # -------------------------------------------------------------------------
    # Recovery score
    # -------------------------------------------------------------------------
    print(f"\n{tcolors.BOLDWHITE}Recovery summary:{tcolors.ENDC}")
    all_rel_errors = []
    for key, t_vals in truth_params.items():
        f_vals = final_params.get(key, [])
        for tv, fv in zip(t_vals, f_vals):
            if abs(tv) > 1e-12:
                all_rel_errors.append(abs(fv - tv) / abs(tv))

    if all_rel_errors:
        mean_rel = sum(all_rel_errors) / len(all_rel_errors)
        max_rel = max(all_rel_errors)
        print(f"  Mean relative error: {mean_rel:.4f}  ({mean_rel * 100:.2f}%)")
        print(f"  Max  relative error: {max_rel:.4f}  ({max_rel * 100:.2f}%)")
        if mean_rel < 0.05:
            print(
                f"{tcolors.GREEN}  Excellent recovery (< 5% mean relative error).{tcolors.ENDC}"
            )
        elif mean_rel < 0.20:
            print(
                f"{tcolors.WARNING}  Partial recovery (5–20% mean relative error). "
                f"Try more epochs or a different seed.{tcolors.ENDC}"
            )
        else:
            print(
                f"{tcolors.FAIL}  Poor recovery (> 20% mean relative error). "
                f"Try more epochs, a different seed, or --use_embedded_events.{tcolors.ENDC}"
            )

    print(f"\n{tcolors.BOLDWHITE}{'=' * 80}{tcolors.ENDC}")
    print(f"{tcolors.GREEN}runfit.py done.{tcolors.ENDC}")
    print(f"{tcolors.BOLDWHITE}{'=' * 80}{tcolors.ENDC}")
