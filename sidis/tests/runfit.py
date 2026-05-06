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

    from sidis.model import TrainableModel, resolve_card_path
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

    def _parse_bool(s: str) -> bool:
        if s.lower() in ("true", "yes", "1"):
            return True
        if s.lower() in ("false", "no", "0"):
            return False
        raise argparse.ArgumentTypeError(f"Expected true/false, got {s!r}")

    parser.add_argument(
        "--save_loss",
        type=_parse_bool,
        default=True,
        help="Save loss.yaml with loss values every 10 epochs. Use true or false. Default: true",
    )
    parser.add_argument(
        "--allow_negative_cs",
        type=_parse_bool,
        default=False,
        help=(
            "Allow negative target cross sections. If false, runfit exits when negatives "
            "are found. Use true or false. Default: false"
        ),
    )
    args = parser.parse_args()

    def _get_fnp_manager(mdl):
        """
        Compatibility helper:
        - Old layout: model.qcf0.fnp_manager
        - New layout: model.qcf0 is already the manager wrapper
        """
        qcf0 = getattr(mdl, "qcf0", None)
        if qcf0 is None:
            raise AttributeError("Model has no qcf0 attribute")
        return getattr(qcf0, "fnp_manager", qcf0)

    # Print all flags (including defaults) in bold green
    _repo_root = script_dir.parent.parent
    _script_rel = pathlib.Path(__file__).resolve().relative_to(_repo_root)
    _save_loss = f"--save_loss {'true' if args.save_loss else 'false'}"
    _allow_negative_cs = (
        f"--allow_negative_cs {'true' if args.allow_negative_cs else 'false'}"
    )
    _flags = (
        f"--cross_section {args.cross_section} "
        f"--seed {args.seed} --epochs {args.epochs} --lr {args.lr} "
        f"--fitresults_dir {args.fitresults_dir} {_save_loss} {_allow_negative_cs}"
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
        # Join first so paths are relative to sidis/tests/, not the shell cwd.
        cs_path = (script_dir / cs_path).resolve()

    # Fit results directory path. If it's not absolute, make it absolute,
    # with the same criteria as the cross section file path
    # (assumed to be relative to the script directory).
    # Check if it exists. If it doesn't, create it.
    fitresults_dir = pathlib.Path(args.fitresults_dir)
    if not fitresults_dir.is_absolute():
        fitresults_dir = (script_dir / fitresults_dir).resolve()

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
    kinematic_cols = None

    # Suffix is needed to choose which loading function to use
    # (if the torch one or the yaml one).
    if suffix == ".pt":
        cs_data = torch.load(cs_path)
        if isinstance(cs_data, dict) and "cross_section" in cs_data:
            target = cs_data["cross_section"]
            events_tensor = cs_data.get("events", None)
            kinematic_cols = cs_data.get("kinematic_columns")
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
        kinematic_cols = cs_dict.get("kinematic_columns", ["x", "PhT", "Q", "z"])
        events_tensor = torch.tensor(
            [[row[c] for c in kinematic_cols] for row in rows], dtype=torch.float64
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

    # If kinematic_columns were not in the file, infer from events shape.
    if kinematic_cols is None:
        kinematic_cols = ["x", "PhT", "Q", "z"]
        if events_tensor.shape[1] >= 6:
            kinematic_cols.extend(["phih", "phis"])
        else:
            for i in range(4, events_tensor.shape[1]):
                kinematic_cols.append(f"col_{i}")

    # Print the events file and shape.
    print(
        f"{tcolors.WHITE}Kinematic points read from: {cs_path} "
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
        print(f"  {tcolors.GREEN}Config from file: {effective_config}{tcolors.ENDC}")
    # When the cross section file is .yaml and was loaded with yaml.safe_load() into cs_dict
    elif cs_dict is not None and "config" in cs_dict:
        effective_config = cs_dict["config"]
        print(f"  {tcolors.GREEN}Config from file: {effective_config}{tcolors.ENDC}")

    # If the config is not found in the cross section file, exit with an error.
    if effective_config is None:
        print(
            f"{tcolors.FAIL}Error: run generate_mock_events.py first, missing config key in metadata of the cross section event file.{tcolors.ENDC}"
        )
        exit(1)

    # Resolve the unified card the same way as TrainableModel (cards/ or absolute path).
    try:
        config_path = resolve_card_path(effective_config, rootdir)
    except FileNotFoundError:
        print(
            f"{tcolors.FAIL}Error: config card not found for {effective_config!r} "
            f"(expected under {cards_dir} or as an absolute path).{tcolors.ENDC}"
        )
        exit(1)

    print(f"  {tcolors.GREEN}Cross section points: {target.shape[0]}{tcolors.ENDC}")
    print(
        f"  {tcolors.GREEN}Range: [{target.min().item():.4e}, {target.max().item():.4e}]{tcolors.ENDC}"
    )

    # -------------------------------------------------------------------------
    # Target sanity checks
    # -------------------------------------------------------------------------
    # Negative cross sections are unphysical for this closure-test workflow.
    # Fail fast to avoid fitting in log(|sigma|)-space where sign information is lost.
    neg_mask = target < 0
    n_neg = int(neg_mask.sum().item())
    if n_neg > 0:
        min_val = target.min().item()
        if args.allow_negative_cs:
            print(
                f"{tcolors.WARNING}Warning: found {n_neg} negative cross section values "
                f"(minimum={min_val:.6e}) in {cs_path}. "
                "Continuing because --allow_negative_cs true was set."
                f"{tcolors.ENDC}"
            )
        else:
            print(
                f"{tcolors.FAIL}Error: found {n_neg} negative cross section values "
                f"(minimum={min_val:.6e}) in {cs_path}. "
                "Please regenerate or filter events before running runfit.py, "
                "or rerun with --allow_negative_cs true."
                f"{tcolors.ENDC}"
            )
            exit(1)

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
        fnp_mgr = _get_fnp_manager(mdl)

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
        # Sivers modules (if present)
        # -------------------------------------------------------------
        if hasattr(fnp_mgr, "sivers_modules") and fnp_mgr.sivers_modules is not None:
            for flavor, mod in fnp_mgr.sivers_modules.items():
                with torch.no_grad():
                    p = mod.get_params_tensor()
                result[f"sivers.{flavor}"] = [p[i].item() for i in range(p.numel())]

        # -------------------------------------------------------------
        # Qiu-Sterman modules (if present)
        # -------------------------------------------------------------
        if hasattr(fnp_mgr, "qiu_sterman_modules") and fnp_mgr.qiu_sterman_modules is not None:
            for flavor, mod in fnp_mgr.qiu_sterman_modules.items():
                with torch.no_grad():
                    p = mod.get_params_tensor()
                result[f"qiu_sterman.{flavor}"] = [p[i].item() for i in range(p.numel())]

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
        # The dictionary should look something like {"pdfs.u": [p0, p1, p2, ...], "ffs.u": [p0, p1, p2, ...], "evolution": [g2_val], ...}
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
        fnp_mgr_local = _get_fnp_manager(mdl)

        # Evolution parameter g2. When g₂ is fixed, `free_g2` is ``None``.
        evo_param = fnp_mgr_local.evolution.free_g2
        evo_is_trainable = evo_param is not None and bool(evo_param.requires_grad)
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
        if hasattr(fnp_mgr_local, "sivers_modules") and fnp_mgr_local.sivers_modules is not None:
            _append_module_records("sivers", fnp_mgr_local.sivers_modules)
        if hasattr(fnp_mgr_local, "qiu_sterman_modules") and fnp_mgr_local.qiu_sterman_modules is not None:
            _append_module_records("qiu_sterman", fnp_mgr_local.qiu_sterman_modules)
        return records

    # -------------------------------------------------------------------------
    # Initialize model and capture truth parameter values
    # -------------------------------------------------------------------------
    print(
        f"\n{tcolors.BOLDWHITE}Initializing model ({effective_config})...{tcolors.ENDC}"
    )

    # Initialize the model, which is a TrainableModel instance.
    model = TrainableModel(fnp_config=effective_config)
    print(f"{tcolors.BOLDGREEN}[runfit.py] Model initialized.{tcolors.ENDC}")

    # Get the fNP manager from the model.
    fnp_mgr = _get_fnp_manager(model)

    # Some manager layouts expose randomization directly, others do not.
    # If unavailable, we continue from config initialization values.
    if not hasattr(fnp_mgr, "randomize_params_in_bounds"):
        print(
            f"{tcolors.WARNING}[runfit.py] randomize_params_in_bounds is unavailable for this fNP manager layout; proceeding without pre-fit randomization.{tcolors.ENDC}"
        )

    # Truth = physical param values at the config's init_params (before randomization).
    # For expression params (e.g. ffs.u[0] = pdfs.u[0]+pdfs.u[1]) this evaluates the
    # expression, so truth matches what was actually used to generate the cross sections.
    truth_params = get_physical_params(model)
    print(
        f"\n{tcolors.BOLDLIGHTBLUE}Truth parameters \n(from {config_path}){tcolors.ENDC}"
    )

    # Print the truth parameters. Those are the physical parameter values at
    # the card's init_params (before randomization). The closure test assumes the
    # same unified card was used to generate the cross sections and to run the fit.
    for key, vals in sorted(truth_params.items()):
        vstr = ", ".join(f"{v:.6f}" for v in vals)
        print(f"  {key:<32} [{vstr}]")

    # -------------------------------------------------------------------------
    # Randomize fNP parameters within bounds
    # -------------------------------------------------------------------------
    if hasattr(fnp_mgr, "randomize_params_in_bounds"):
        fnp_mgr.randomize_params_in_bounds(seed=args.seed)
    else:
        print(
            f"{tcolors.WARNING}[runfit.py] randomize_params_in_bounds not available for this fNP manager layout; using config initialization as starting point.{tcolors.ENDC}"
        )

    print(
        f"\n{tcolors.BOLDGREEN}[runfit.py] Randomized fNP parameters within bounds "
        f"(seed={args.seed}).{tcolors.ENDC}"
    )

    random_params = get_physical_params(model)

    # Print the randomized parameters.
    print(
        f"\n{tcolors.BOLDLIGHTBLUE}Initial parameters after randomization \n(from {config_path}){tcolors.ENDC}"
    )
    for key, vals in sorted(random_params.items()):
        vstr = ", ".join(f"{v:.6f}" for v in vals)
        print(f"  {key:<32} [{vstr}]")

    # -------------------------------------------------------------------------
    # Initial forward pass (just for informational purposes)
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
    # If the fNP manager has the get_trainable_bounds method and it returns True,
    # then clamp the parameters to their bounds after each gradient step.
    # For fnp_simple, evolution g2 is the only param not handled by sigmoid reparametrization,
    # so it may need clamping to its bounds after each gradient step.
    # Other combos may have more than one param that needs clamping.
    # clamp_after_step is set to either True or False, and used in the minimization loop.
    clamp_after_step = hasattr(fnp_mgr, "get_trainable_bounds") and bool(
        fnp_mgr.get_trainable_bounds()
    )

    # Log-space MSE: mean( (log|pred| - log|target|)^2 )
    # Floor applied before log to guard against zeros / near-zero values.
    # Only absolute values enter the loss function.
    EPS_LOG = 1e-30
    MAX_ABS_FOR_LOG = 1e30
    target_log = torch.log(target.abs().clamp(min=EPS_LOG))

    def log_mse_loss() -> torch.Tensor:
        """
        log_mse_loss() is the objective that gets minimized. For
        that to work, PyTorch needs gradients from loss back to the model parameters.
        That requires:
        - A forward pass that builds the computation graph.
        - A backward pass that computes the gradients.
        """
        # Get predictions from the model. Predictions change with the model parameters.
        # This time we need to compute the gradients.
        if model.qcf0.sivers_flag:
            pred = model.get_FUT_sin_phih_minus_phis(events_tensor)
        else:
            pred = model(events_tensor)
        if not torch.isfinite(pred).all():
            bad = int((~torch.isfinite(pred)).sum().item())
            print(
                f"{tcolors.WARNING}[runfit.py] Warning: prediction contains {bad} non-finite values; "
                "applying nan_to_num and clamping for stable log-loss computation."
                f"{tcolors.ENDC}"
            )
            pred = torch.nan_to_num(
                pred,
                nan=0.0,
                posinf=MAX_ABS_FOR_LOG,
                neginf=-MAX_ABS_FOR_LOG,
            )

        # Compute the log of the absolute values of the predictions.
        pred_log = torch.log(pred.abs().clamp(min=EPS_LOG, max=MAX_ABS_FOR_LOG))

        # Compute the log-space MSE loss.
        return torch.mean((pred_log - target_log) ** 2)

    # Rprop: sign-based adaptive per-parameter steps.
    # - model.parameters(): the parameters to optimize.
    # - lr=args.lr: the initial step size.
    # - etas=(0.5, 1.2): the step shrink and grow factors. They shrink by 50% on sign flip,
    #   grow by 20% on consistent sign.
    # - step_sizes=(1e-6, 1.0): the maximum step size. cap max step at 1.0 to avoid
    #   overshooting in theta space (sigmoid-reparametrized params live roughly in [-5, 5],
    #   so the size cap is chosen because theta is on that scale).
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
        f"\n{tcolors.BOLDGREEN}Minimizing log-MSE with Rprop "
        f"({args.epochs} epochs, initial step size={args.lr})...{tcolors.ENDC}"
    )
    print("=" * 80)

    losses = []
    loss_yaml_entries = []  # (epoch, loss) every 10 epochs for loss.yaml

    # Determine how often to report the loss on terminal.
    if args.epochs > 1000:
        report_every = max(1, args.epochs // 100)
    else:
        report_every = max(1, args.epochs // 10)

    # Run the number of epochs specified by the --epochs flag.
    for epoch in range(args.epochs):

        # Clears the gradient buffers so they can be reused.
        # Put to zero the gradients from previous iteration (PyTorch accumulates by default).
        optimizer.zero_grad()

        # Compute the log-space MSE loss.
        # Forward pass: compute loss and build computational graph.
        loss = log_mse_loss()

        # Compute the gradients of the loss with respect to the model parameters.
        # Backward pass: compute gradients of loss w.r.t. parameters (populates param.grad).
        loss.backward()

        # Guard optimizer against occasional non-finite gradients from unstable points.
        for p in model.parameters():
            if p.grad is None:
                continue
            bad_grad = ~torch.isfinite(p.grad)
            if bad_grad.any():
                p.grad[bad_grad] = 0.0

        # Update the model parameters using the computed gradients.
        optimizer.step()

        # Clamp the physical parameters to their bounds to stay within the
        # allowed intervals after each gradient step.
        if clamp_after_step:
            fnp_mgr.clamp_parameters_to_bounds()

        # Extract scalar for logging. Append the loss to the list of losses.
        loss_item = loss.item()
        losses.append(loss_item)

        # Record (epoch, loss) every 10 epochs for loss.yaml (epoch 1-based).
        if args.save_loss and ((epoch + 1) % 10 == 0 or epoch == 0):
            loss_yaml_entries.append({"epoch": epoch + 1, "loss": float(loss_item)})

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

    # # Uncomment below to have a full table of parameters truth vs final,
    # # including the fixed params, printed out on terminal.
    # print_param_table(
    #     truth_params,
    #     final_params,
    #     "Parameter comparison: Truth vs Fitted",
    # )

    # -------------------------------------------------------------------------
    # Recovery score (trainable parameters only)
    # -------------------------------------------------------------------------
    print(f"\n{tcolors.BOLDWHITE}Recovery per trainable parameter:{tcolors.ENDC}")

    # Collect records and keep only those optimized directly by the fit.
    records = collect_parameter_records(model)
    trainable_records = [r for r in records if r["is_trainable_direct"]]

    def _param_key_and_idx(rec_name: str) -> tuple:
        """Map record name to (key, idx) for truth/random/final_params lookup."""
        if rec_name == "evolution.g2":
            return ("evolution", 0)
        # e.g. "pdfs.u[0]" -> ("pdfs.u", 0)
        bracket = rec_name.rfind("[")
        if bracket >= 0:
            key = rec_name[:bracket]
            idx = int(rec_name[bracket + 1 : -1])
            return (key, idx)
        return (rec_name, 0)

    if trainable_records:
        # Table header: Parameter, Truth, Initial, Final, Abs.Diff, Rel.Diff (%), Recovery.
        header = (
            f"  {'Parameter':<20} {'Truth':>10} {'Initial':>10} {'Final':>10} "
            f"{'Abs.Diff':>10} {'Rel.Diff%':>10}  Recovery"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for rec in trainable_records:
            pname = rec["name"]
            # Map record name to (key, idx) for param dict lookup.
            key, idx = _param_key_and_idx(pname)
            # Extract truth, initial (randomized), and final (fitted) values.
            truth_val = (
                truth_params.get(key, [None])[idx]
                if idx < len(truth_params.get(key, []))
                else None
            )
            init_val = (
                random_params.get(key, [None])[idx]
                if idx < len(random_params.get(key, []))
                else None
            )
            final_val = (
                final_params.get(key, [None])[idx]
                if idx < len(final_params.get(key, []))
                else None
            )
            if truth_val is None or init_val is None or final_val is None:
                print(
                    f"{tcolors.FAIL}No value found for parameter: {pname}{tcolors.ENDC}"
                )
                continue

            # Absolute difference: final - truth.
            abs_diff = final_val - truth_val

            # Relative difference: (final - truth) / |truth|, expressed in %.
            if abs(truth_val) > 1e-12:
                rel_diff = (final_val - truth_val) / abs(truth_val)
            else:
                rel_diff = float("nan")
            abs_rel = abs(rel_diff)

            # Recovery status based on |rel_diff|: <5% Excellent, 5–20% Partial, >20% Poor.
            if abs_rel < 0.05:
                recovery = f"{tcolors.GREEN}Excellent{tcolors.ENDC}"
            elif abs_rel < 0.20:
                recovery = f"{tcolors.WARNING}Partial{tcolors.ENDC}"
            else:
                recovery = f"{tcolors.FAIL}Poor{tcolors.ENDC}"

            # Format rel_diff as percentage; use N/A for NaN.
            rel_str = f"{rel_diff * 100:+.2f}%" if rel_diff == rel_diff else "N/A"
            print(
                f"  {pname:<20} {truth_val:>10.6f} {init_val:>10.6f} {final_val:>10.6f} "
                f"{abs_diff:>+10.6f} {rel_str:>10}  {recovery}"
            )
    else:
        print("  No trainable parameters.")

    # -------------------------------------------------------------------------
    # Save fit outputs
    # -------------------------------------------------------------------------
    # If the output folder doesn't exist, create it.
    fitresults_dir.mkdir(parents=True, exist_ok=True)

    # kinematic_cols was read from the cross section file at load time (or inferred).
    events_cpu = events_tensor.detach().cpu()
    target_cpu = target.detach().cpu()
    final_pred_cpu = final_pred.detach().cpu()

    # Build the fit results rows.
    fit_rows: List[Dict[str, float]] = []

    # Sanity check: kinematic_cols must match the number of event columns.
    # This ensures row[cname] = events_cpu[i, j] maps the correct column to each name.
    assert len(kinematic_cols) == events_tensor.shape[1], (
        f"kinematic_cols length ({len(kinematic_cols)}) must match events columns "
        f"({events_tensor.shape[1]}). Check kinematic_columns in the cross section file."
    )

    # Iterate over the events.
    for i in range(events_cpu.shape[0]):
        # Create a row for the fit results.
        row: Dict[str, float] = {}

        # Fill the row with kinematic values. kinematic_cols was read from the cross
        # section file at load time (or inferred). Column j corresponds to kinematic_cols[j].
        for j, cname in enumerate(kinematic_cols):
            row[cname] = float(events_cpu[i, j].item())

        # Fill the row with the cross section values.
        row["cross_section_original"] = float(target_cpu[i].item())
        row["cross_section_reconstructed"] = float(final_pred_cpu[i].item())

        # Add the row to the fit results.
        fit_rows.append(row)

    fitresults_out = {
        "description": "Fit results: kinematics with original and reconstructed cross sections",
        "config": effective_config,
        "cross_section_source": str(cs_path),
        "events_source": str(cs_path),
        "n_events": int(events_cpu.shape[0]),
        "n_epochs": args.epochs,
        "seed": args.seed,
        "loss": {
            "type": "log_mse",
            "lr": args.lr,
            "initial": float(losses[0]),
            "final": float(losses[-1]),
            "reduction_percent": float(
                (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0.0
            ),
        },
        "data": fit_rows,
    }

    # Build unified parameters output: one entry per parameter with truth, initial, final, diff, rel_diff.
    params_fitted = collect_parameter_records(model)
    parameters_out_rows: List[Dict[str, Any]] = []
    for rec in params_fitted:
        name = rec["name"]
        key, idx = _param_key_and_idx(name)
        truth_val = (
            truth_params.get(key, [None])[idx]
            if idx < len(truth_params.get(key, []))
            else None
        )
        init_val = (
            random_params.get(key, [None])[idx]
            if idx < len(random_params.get(key, []))
            else None
        )
        final_val = float(rec["value"])
        if truth_val is not None:
            diff = final_val - truth_val
            rel_diff = diff / abs(truth_val) if abs(truth_val) > 1e-12 else None
            if rel_diff is not None and rel_diff != rel_diff:
                rel_diff = None
        else:
            diff = None
            rel_diff = None
        parameters_out_rows.append(
            {
                "parameter": name,
                "fit_role": rec["fit_role"],
                "is_fixed": bool(rec["is_fixed"]),
                "is_trainable_direct": bool(rec["is_trainable_direct"]),
                "reference_or_expression": rec["reference_or_expression"],
                "truth": float(truth_val) if truth_val is not None else None,
                "initial": float(init_val) if init_val is not None else None,
                "final": final_val,
                "diff": float(diff) if diff is not None else None,
                "rel_diff": float(rel_diff) if rel_diff is not None else None,
            }
        )

    parameters_out = {
        "description": "Parameter information.",
        "config": effective_config,
        "notes": {
            "fit_role.fixed": "Parameter fixed by configuration (not fitted).",
            "fit_role.trainable_direct": "Independent parameter optimized directly.",
            "fit_role.linked_reference": "Linked to another parameter by reference.",
            "fit_role.linked_expression": "Computed from an expression of other parameters.",
        },
        "parameters": parameters_out_rows,
    }

    fitresults_yaml = fitresults_dir / "fitresults.yaml"
    parameters_yaml = fitresults_dir / "parameters.yaml"
    with open(fitresults_yaml, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(fitresults_out)))
    with open(parameters_yaml, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(parameters_out)))

    # Save loss at every 10 epochs for plotting (if enabled).
    loss_yaml = fitresults_dir / "loss.yaml"
    if args.save_loss:
        loss_out = {
            "description": "Loss values every 10 epochs for plotting.",
            "config": effective_config,
            "trainable_parameters": len(trainable_records),
            "n_events": int(events_cpu.shape[0]),
            "n_epochs": args.epochs,
            "seed": args.seed,
            "loss_info": {
                "type": "log_mse",
                "lr": args.lr,
                "initial": float(losses[0]),
                "final": float(losses[-1]),
                "reduction_percent": float(
                    (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0.0
                ),
            },
            "loss": loss_yaml_entries,
        }
        with open(loss_yaml, "w") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.create(loss_out)))

    print(f"\n{tcolors.GREEN}Saved fit outputs to:{tcolors.ENDC}")
    print(f"  {fitresults_yaml}")
    print(f"  {parameters_yaml}")
    if args.save_loss:
        print(f"  {loss_yaml}")

    print(f"\n{tcolors.BOLDWHITE}{'=' * 80}{tcolors.ENDC}")
    print(f"{tcolors.GREEN}runfit.py done.{tcolors.ENDC}")
    print(f"{tcolors.BOLDWHITE}{'=' * 80}{tcolors.ENDC}")
