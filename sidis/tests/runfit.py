"""
runfit.py - Fit fNP simple model parameters to target cross sections.

Loads cross sections from files in events/*.pt (or .yaml), randomizes fNP
parameters within their bounds, then minimizes a log-space MSE loss using
Rprop to recover the original parameter values.

Optimizer — Rprop (Resilient Backpropagation):
  Uses only the SIGN of each gradient, completely ignoring its magnitude.
  Each parameter has its own adaptive step size that grows when the gradient
  sign is consistent and shrinks when it flips.  This makes Rprop naturally
  scale-invariant: it performs equally well regardless of whether individual
  cross sections are O(1) or O(1e-15), even after the log transformation.

Loss: mean( (log|pred| - log|target|)^2 )

Usage (from repo root):
    python sidis/tests/runfit.py
    python sidis/tests/runfit.py --seed 99 --epochs 500
    python sidis/tests/runfit.py --cross_section events/cross_section_events_1000.pt

NOTE: Events kinematics are read from the cross section file (.pt or .yaml).
"""

import torch
import argparse
import pathlib
import sys
from typing import Any, Dict, List
from omegaconf import OmegaConf

if __name__ == "__main__":

    # Ensure sidis can be imported from repo root. Assumes the file being
    # run is at sidis/tests/runfit.py (three levels below the repo root).
    _repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

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
        description=(
            "Fit fNP simple model parameters to target cross sections "
            "(Rprop optimizer, log-MSE loss)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  From repo root, the default 
  python sidis/tests/runfit.py
  
  is equivalent to running:
  python sidis/tests/runfit.py --cross_section events/cross_section_events_1000.pt""",
    )
    parser.add_argument(
        "--cross_section",
        "-cs",
        type=str,
        default="events/cross_section_events_1000.pt",
        help="Path to cross section file (.pt or .yaml). Default: events/cross_section_events_1000.pt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for parameter randomization. Default: 7",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of Rprop gradient steps. Default: 1000",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help=(
            "Rprop initial step size (per-parameter, in the internal theta space). "
            "Default: 0.01"
        ),
    )
    parser.add_argument(
        "--fitresults_dir",
        type=str,
        default="fitresults",
        help="Directory where fit output YAML files are saved. Default: fitresults",
    )
    args = parser.parse_args()

    # Print all flags (including defaults) in bold green
    _repo_root = script_dir.parent.parent
    _script_rel = pathlib.Path(__file__).resolve().relative_to(_repo_root)
    _flags = (
        f"--cross_section {args.cross_section} "
        f"--seed {args.seed} --epochs {args.epochs} --lr {args.lr} "
        f"--fitresults_dir {args.fitresults_dir}"
    )
    print(f"{tcolors.BOLDGREEN}running @{_script_rel} {_flags}{tcolors.ENDC}")

    # -------------------------------------------------------------------------
    # Resolve file paths. Relative paths should be resolved relative to the
    # script’s directory, not the current working directory.
    # -------------------------------------------------------------------------
    # Cross section file path. If the path given from command line it's not absolute,
    # make it absolute. A relative path is given from the command line,
    # is assumed to be relative to the script directory (sidis/tests/).
    #  Check if the path exists. If it doesn't, exit with an error.
    cs_path = pathlib.Path(args.cross_section)
    if not cs_path.is_absolute():
        cs_path = script_dir / cs_path.resolve()

    # Fit results directory path. If it's not absolute, make it absolute,
    # with the same criteria as the cross section file path
    # (assumed to be relative to the script directory).
    # Check if it exists. If it doesn't, create it.
    fitresults_dir = pathlib.Path(args.fitresults_dir)
    if not fitresults_dir.is_absolute():
        fitresults_dir = script_dir / fitresults_dir

    # -------------------------------------------------------------------------
    # Load cross sections
    # -------------------------------------------------------------------------
    print(f"\n{tcolors.BOLDWHITE}Loading cross sections from: {cs_path}{tcolors.ENDC}")

    # cs_path.suffix - returns the file extension, including the dot (e.g. .pt, .yaml, .yml).
    # .lower() – converts it to lowercase, making it case-insensitive.
    suffix = cs_path.suffix.lower()

    events_tensor = None
    cs_data = None
    cs_dict = None

    # Suffix is needed to choose which loading function to use
    # (if the torch one or the yaml one).
    if suffix == ".pt":
        cs_data = torch.load(cs_path)
        if isinstance(cs_data, dict) and "cross_section" in cs_data:
            target = cs_data["cross_section"]
            events_tensor = cs_data.get("events", None)
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
        events_tensor = torch.tensor(
            [[row[c] for c in cols] for row in rows], dtype=torch.float64
        )
    else:
        print(
            f"{tcolors.FAIL}Error: unsupported file format '{suffix}'. "
            f"Use .pt or .yaml.{tcolors.ENDC}"
        )
        exit(1)

    # If the events are not found in the cross section file,
    # exit with an error. Cross section files must contain kinematics.
    if events_tensor is None:
        print(
            f"{tcolors.FAIL}Error: no events found in {cs_path}. "
            f"Cross section files must contain kinematics.{tcolors.ENDC}"
        )
        exit(1)

    # Print the events file and shape.
    print(
        f"\n{tcolors.GREEN}Kinematic points read from: {cs_path} "
        f"(shape: {events_tensor.shape}){tcolors.ENDC}"
    )

    # -------------------------------------------------------------------------
    # Load config. It's required that the cross section file contains the
    # config name. runfit.py runs closure test within same model
    # -------------------------------------------------------------------------
    effective_config = None

    # When the cross section file is .pt and was loaded with torch.load() into cs_data
    if cs_data is not None and isinstance(cs_data, dict) and "config" in cs_data:
        effective_config = cs_data["config"]
        print(f"  Config from file: {effective_config}")
    # When the cross section file is .yaml and was loaded with yaml.safe_load() into cs_dict
    elif cs_dict is not None and "config" in cs_dict:
        effective_config = cs_dict["config"]
        print(f"  Config from file: {effective_config}")

    # If the config is not found in the cross section file, exit with an error.
    if effective_config is None:
        print(
            f"{tcolors.FAIL}Error: run generate_mock_events.py first, missing config key in metadata of the cross section event file.{tcolors.ENDC}"
        )
        exit(1)

    # Once established which configuration card to call, resolve its path.
    # It's assumed to be in the cards/ directory.
    # Check if it exists. If it doesn't, exit with an error.
    config_path = cards_dir / effective_config
    if not config_path.exists():
        print(
            f"{tcolors.FAIL}Error: config file not found: {config_path}{tcolors.ENDC}"
        )
        exit(1)

    print(f"  Cross section points: {target.shape[0]}")
    print(f"  Range: [{target.min().item():.4e}, {target.max().item():.4e}]")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def get_physical_params(mdl) -> dict:
        """
        Return a dict of physical parameter values evaluated by each flavor module.

        It always returns the current physical parameters of the model at the time
        it’s called. It does not specifically target "init" or "final" values;
        it just reads the model's state.

        So, in this code:
        - The first call gives the config's init values (before randomization).
        - The second gives the randomized starting values.
        - The third gives the fitted values.
        - Your understanding is correct for the first call: it captures the values the model is initialized with from the config.

        Keys: 'pdfs.<flavor>', 'ffs.<flavor>', 'evolution'.
        Expression-linked params (e.g. ffs.u[0] = pdfs.u[0]+pdfs.u[1]) are
        evaluated at their current values, so truth and fit are directly comparable.
        """
        # Initialize a dictionary to hold the physical parameter values:
        # pdfs.<flavor>, ffs.<flavor>, evolution.
        result = {}

        # Get the fNP manager from the model. It holds PDF, FF, and evolution modules.
        fnp_mgr = mdl.qcf0.fnp_manager

        # -------------------------------------------------------------
        # PDF modules
        # -------------------------------------------------------------
        # Iterate over the PDF modules.
        # flavor: "u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"
        # mod: TMDPDFExponential instance
        for flavor, mod in fnp_mgr.pdf_modules.items():
            # No gradients needed for read-only extraction.
            # Get raw param tensor from this PDF flavor module. get_params_tensor()
            # is a function defined in the TMDPDFExponential class.
            # It returns a 1D tensor of physical parameter values [p0, p1, ...] for this flavor.
            with torch.no_grad():
                p = mod.get_params_tensor()

            # Set map keys for the result dictionary.
            # p[i].item() is the i-th element of the tensor p, converted to a Python float.
            # p.numel() is the number of elements in the tensor p.
            result[f"pdfs.{flavor}"] = [p[i].item() for i in range(p.numel())]

        # -------------------------------------------------------------
        # FF modules
        # -------------------------------------------------------------
        # Iterate over the FF modules.
        # flavor: "u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"
        # mod: TMDPDFExponential instance
        for flavor, mod in fnp_mgr.ff_modules.items():
            # No gradients needed for read-only extraction.
            # Get raw param tensor from this FF flavor module. get_params_tensor()
            # is a function defined in the TMDFFExponential class.
            # It returns a 1D tensor of physical parameter values [p0, p1, ...] for this flavor.
            with torch.no_grad():
                p = mod.get_params_tensor()

            # Set map keys for the result dictionary.
            # p[i].item() is the i-th element of the tensor p, converted to a Python float.
            # p.numel() is the number of elements in the tensor p.
            result[f"ffs.{flavor}"] = [p[i].item() for i in range(p.numel())]

        # -------------------------------------------------------------
        # Evolution
        # -------------------------------------------------------------
        # Get the evolution module from the fNP manager.
        # It holds the g2 parameter.
        evo = fnp_mgr.evolution

        # Get the g2 parameter value.
        # evo.g2.item() is the g2 parameter value, converted to a Python float.
        g2_val = evo.g2.item()

        # Set map keys for the result dictionary, for evolution.
        # Single value: g2
        result["evolution"] = [g2_val]  # Single value: strong coupling g2

        # Return the result dictionary.
        return result

    def build_param_table_rows(truth: dict, current: dict) -> List[Dict[str, Any]]:
        """Build list of parameter rows (truth, current, diff, rel_diff) for YAML output."""
        rows: List[Dict[str, Any]] = []
        for key in sorted(truth.keys()):
            t_vals = truth[key]
            c_vals = current.get(key, [None] * len(t_vals))
            for idx, (tv, cv) in enumerate(zip(t_vals, c_vals)):
                pname = f"{key}[{idx}]"
                row: Dict[str, Any] = {"parameter": pname, "truth": tv}
                if tv is not None and cv is not None:
                    diff = cv - tv
                    rel = diff / tv if abs(tv) > 1e-12 else float("nan")
                    row["current"] = cv
                    row["diff"] = diff
                    row["rel_diff"] = rel
                else:
                    row["current"] = cv
                    row["diff"] = None
                    row["rel_diff"] = None
                rows.append(row)
        return rows

    def print_param_table(truth: dict, current: dict, title: str) -> None:
        print(f"\n{tcolors.BOLDWHITE}{title}{tcolors.ENDC}")
        header = f"  {'Parameter':<32} {'Truth':>12} {'Current':>12} {'Diff':>12} {'Rel.Diff':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row in build_param_table_rows(truth, current):
            pname = row["parameter"]
            tv, cv = row["truth"], row["current"]
            if tv is not None and cv is not None:
                diff, rel = row["diff"], row["rel_diff"]
                print(
                    f"  {pname:<32} {tv:>12.6f} {cv:>12.6f} {diff:>+12.6f} {rel:>+10.4f}"
                )
            else:
                print(f"  {pname:<32} {tv!s:>12} {'N/A':>12}")

    def collect_parameter_records(mdl) -> List[Dict[str, Any]]:
        """
        Collect parameter records with fit-role metadata and physical values.

        fit_role is one of:
          - fixed
          - trainable_direct
          - linked_reference
          - linked_expression
        """
        records: List[Dict[str, Any]] = []
        fnp_mgr_local = mdl.qcf0.fnp_manager

        # Evolution parameter g2
        evo_param = fnp_mgr_local.evolution.free_g2
        evo_is_trainable = bool(evo_param.requires_grad)
        records.append(
            {
                "name": "evolution.g2",
                "fit_role": "trainable_direct" if evo_is_trainable else "fixed",
                "is_fixed": not evo_is_trainable,
                "is_trainable_direct": evo_is_trainable,
                "reference_or_expression": None,
                "value": float(fnp_mgr_local.evolution.g2.item()),
            }
        )

        def _append_module_records(distrib_key: str, module_dict) -> None:
            for flavor, mod in module_dict.items():
                params_tensor = mod.get_params_tensor().detach().cpu()
                for cfg in mod.param_configs:
                    idx = cfg["idx"]
                    parsed = cfg["parsed"]
                    ptype = parsed.get("type")
                    is_fixed = bool(parsed.get("is_fixed", False))
                    expr_or_ref = None

                    if ptype == "reference":
                        ref = parsed.get("value", {})
                        ref_type = ref.get("type") or distrib_key
                        expr_or_ref = (
                            f"{ref_type}.{ref.get('flavor')}[{ref.get('param_idx')}]"
                        )
                    elif ptype == "expression":
                        expr_or_ref = parsed.get("value")

                    if is_fixed:
                        fit_role = "fixed"
                    elif ptype == "boolean":
                        fit_role = "trainable_direct"
                    elif ptype == "reference":
                        fit_role = "linked_reference"
                    elif ptype == "expression":
                        fit_role = "linked_expression"
                    else:
                        fit_role = "fixed"

                    records.append(
                        {
                            "name": f"{distrib_key}.{flavor}[{idx}]",
                            "fit_role": fit_role,
                            "is_fixed": fit_role == "fixed",
                            "is_trainable_direct": fit_role == "trainable_direct",
                            "reference_or_expression": expr_or_ref,
                            "value": float(params_tensor[idx].item()),
                        }
                    )

        _append_module_records("pdfs", fnp_mgr_local.pdf_modules)
        _append_module_records("ffs", fnp_mgr_local.ff_modules)
        return records

    # -------------------------------------------------------------------------
    # Initialize model and capture truth parameter values
    # -------------------------------------------------------------------------
    print(
        f"\n{tcolors.BOLDWHITE}Initializing model ({effective_config})...{tcolors.ENDC}"
    )
    # Initialize the model, which is a TrainableModel instance.
    model = TrainableModel(fnp_config=effective_config)
    print(f"{tcolors.GREEN}Model initialized.{tcolors.ENDC}")

    fnp_mgr = model.qcf0.fnp_manager
    if not hasattr(fnp_mgr, "randomize_params_in_bounds"):
        print(
            f"{tcolors.FAIL}Error: runfit.py requires an fNP combo that implements "
            f"randomize_params_in_bounds. The loaded combo does not provide this method.{tcolors.ENDC}"
        )
        exit(1)

    # Truth = physical param values at the config's init_params (before randomization).
    # For expression params (e.g. ffs.u[0] = pdfs.u[0]+pdfs.u[1]) this evaluates the
    # expression, so truth matches what was actually used to generate the cross sections.
    truth_params = get_physical_params(model)
    print(
        f"\n{tcolors.OKLIGHTBLUE}Truth parameters (config init_params / expressions):{tcolors.ENDC}"
    )
    for key, vals in sorted(truth_params.items()):
        vstr = ", ".join(f"{v:.6f}" for v in vals)
        print(f"  {key:<32} [{vstr}]")

    # -------------------------------------------------------------------------
    # Randomize fNP parameters within bounds
    # -------------------------------------------------------------------------
    fnp_mgr.randomize_params_in_bounds(seed=args.seed)
    print(
        f"\n{tcolors.GREEN}Randomized fNP parameters within bounds "
        f"(seed={args.seed}).{tcolors.ENDC}"
    )

    random_params = get_physical_params(model)
    print(
        f"\n{tcolors.OKLIGHTBLUE}Initial parameters after randomization:{tcolors.ENDC}"
    )
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
    print(f"  Target range:     [{target.min().item():.4e}, {target.max().item():.4e}]")

    # -------------------------------------------------------------------------
    # Optimizer and loss setup
    # -------------------------------------------------------------------------
    # Evolution g2 is the only param not handled by sigmoid reparametrization,
    # so it may need clamping to its bounds after each gradient step.
    clamp_after_step = hasattr(fnp_mgr, "get_trainable_bounds") and bool(
        fnp_mgr.get_trainable_bounds()
    )

    # Log-space MSE: mean( (log|pred| - log|target|)^2 )
    # Floor applied before log to guard against zeros / near-zero values.
    EPS_LOG = 1e-40
    target_log = torch.log(target.abs().clamp(min=EPS_LOG))

    def log_mse_loss() -> torch.Tensor:
        pred = model(events_tensor)
        pred_log = torch.log(pred.abs().clamp(min=EPS_LOG))
        return torch.mean((pred_log - target_log) ** 2)

    # Rprop: sign-based adaptive per-parameter steps.
    # etas=(0.5, 1.2): step shrinks by 50% on sign flip, grows by 20% on consistency.
    # step_sizes=(1e-6, 1.0): cap max step at 1.0 to avoid overshooting in theta space
    #   (sigmoid-reparametrized params live roughly in [-5, 5]).
    optimizer = torch.optim.Rprop(
        model.parameters(),
        lr=args.lr,
        etas=(0.5, 1.2),
        step_sizes=(1e-6, 1.0),
    )

    # -------------------------------------------------------------------------
    # Minimization loop
    # -------------------------------------------------------------------------
    print(
        f"\n{tcolors.BOLDWHITE}Minimizing log-MSE with Rprop "
        f"({args.epochs} epochs, initial step size={args.lr})...{tcolors.ENDC}"
    )
    print("=" * 80)

    losses = []
    report_every = max(1, args.epochs // 10)

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss = log_mse_loss()
        loss.backward()
        optimizer.step()

        if clamp_after_step:
            fnp_mgr.clamp_parameters_to_bounds()

        loss_item = loss.item()
        losses.append(loss_item)

        if (epoch + 1) % report_every == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:5d}/{args.epochs}:  log-MSE = {loss_item:.6e}")

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
                f"Try more epochs (--epochs), a different seed, "
                f"or a different seed.{tcolors.ENDC}"
            )

    # -------------------------------------------------------------------------
    # Save fit outputs
    # -------------------------------------------------------------------------
    fitresults_dir.mkdir(parents=True, exist_ok=True)

    kinematic_cols = ["x", "PhT", "Q", "z"]
    if events_tensor.shape[1] >= 6:
        kinematic_cols.extend(["phih", "phis"])
    else:
        # Generic names if there are additional columns beyond [x, PhT, Q, z]
        for i in range(4, events_tensor.shape[1]):
            kinematic_cols.append(f"col_{i}")

    events_cpu = events_tensor.detach().cpu()
    target_cpu = target.detach().cpu()
    final_pred_cpu = final_pred.detach().cpu()

    fit_rows: List[Dict[str, float]] = []
    for i in range(events_cpu.shape[0]):
        row: Dict[str, float] = {}
        for j, cname in enumerate(kinematic_cols):
            row[cname] = float(events_cpu[i, j].item())
        row["cross_section_original"] = float(target_cpu[i].item())
        row["cross_section_reconstructed"] = float(final_pred_cpu[i].item())
        fit_rows.append(row)

    fitresults_out = {
        "description": "Fit results: kinematics with original and reconstructed cross sections",
        "config": effective_config,
        "cross_section_source": str(cs_path),
        "events_source": str(cs_path),
        "n_points": int(events_cpu.shape[0]),
        "kinematic_columns": kinematic_cols,
        "loss": {
            "type": "log_mse",
            "initial": float(losses[0]),
            "final": float(losses[-1]),
            "reduction_percent": float(
                (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0.0
            ),
        },
        "data": fit_rows,
    }

    # Rebuild with truth and fitted values side-by-side.
    # Truth is taken from earlier captured values in truth_params.
    truth_map: Dict[str, float] = {}
    for key, vals in truth_params.items():
        if key == "evolution":
            truth_map["evolution.g2"] = float(vals[0])
        else:
            for idx, v in enumerate(vals):
                truth_map[f"{key}[{idx}]"] = float(v)

    params_fitted = collect_parameter_records(model)
    params_out_rows: List[Dict[str, Any]] = []
    for rec in params_fitted:
        name = rec["name"]
        params_out_rows.append(
            {
                "parameter": name,
                "fit_role": rec["fit_role"],
                "is_fixed": bool(rec["is_fixed"]),
                "is_trainable_direct": bool(rec["is_trainable_direct"]),
                "reference_or_expression": rec["reference_or_expression"],
                "original_value": truth_map.get(name),
                "fitted_value": float(rec["value"]),
                "difference": (
                    float(rec["value"] - truth_map[name])
                    if name in truth_map and truth_map[name] is not None
                    else None
                ),
            }
        )

    params_out = {
        "description": "Final fitted parameter values with fit role metadata",
        "config": effective_config,
        "notes": {
            "fit_role.fixed": "Parameter fixed by configuration (not fitted).",
            "fit_role.trainable_direct": "Independent parameter optimized directly.",
            "fit_role.linked_reference": "Linked to another parameter by reference.",
            "fit_role.linked_expression": "Computed from an expression of other parameters.",
        },
        "parameters": params_out_rows,
    }

    # Parameter comparison tables (same as printed to terminal)
    truth_vs_initial_rows = build_param_table_rows(truth_params, random_params)
    truth_vs_fitted_rows = build_param_table_rows(truth_params, final_params)

    # Make YAML-serializable (replace nan with None)
    def _sanitize_for_yaml(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            o = dict(r)
            if isinstance(o.get("rel_diff"), float) and (
                o["rel_diff"] != o["rel_diff"]
            ):  # nan
                o["rel_diff"] = None
            out.append(o)
        return out

    parameters_out = {
        "description": "Parameter comparison tables (Truth vs Initial, Truth vs Fitted)",
        "config": effective_config,
        "truth_vs_initial": _sanitize_for_yaml(truth_vs_initial_rows),
        "truth_vs_fitted": _sanitize_for_yaml(truth_vs_fitted_rows),
    }

    fitresults_yaml = fitresults_dir / "fitresults.yaml"
    fitparams_yaml = fitresults_dir / "fit_parameters.yaml"
    parameters_yaml = fitresults_dir / "parameters.yaml"
    with open(fitresults_yaml, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(fitresults_out)))
    with open(fitparams_yaml, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(params_out)))
    with open(parameters_yaml, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(parameters_out)))

    print(f"\n{tcolors.GREEN}Saved fit outputs to:{tcolors.ENDC}")
    print(f"  {fitresults_yaml}")
    print(f"  {fitparams_yaml}")
    print(f"  {parameters_yaml}")

    print(f"\n{tcolors.BOLDWHITE}{'=' * 80}{tcolors.ENDC}")
    print(f"{tcolors.GREEN}runfit.py done.{tcolors.ENDC}")
    print(f"{tcolors.BOLDWHITE}{'=' * 80}{tcolors.ENDC}")
