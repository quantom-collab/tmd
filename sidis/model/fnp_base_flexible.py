"""
Flexible fNP Implementation with Parameter Linking

This module provides a flexible fNP implementation that supports parameter linking
and constraints between flavors. Parameters can be:
- Independent (boolean true/false)
- Linked to another parameter (e.g., u[0] means link to parameter 0 of flavor u)
- Complex expressions (e.g., 2*u[1] + 0.1 means parameter = 2 * u[1] + 0.1, evaluated dynamically)

Contents of this file:
- Parameter Link Parser: Parses free_mask entries to identify references and expressions
- Parameter Registry: Tracks which parameters are linked and manages shared Parameter objects
- Dependency Resolver: Resolves circular dependencies and builds parameter dependency graph
- Expression Evaluator: Safely evaluates mathematical expressions during forward pass
- Flexible PDF/FF Base Classes: Modified PDF/FF classes that support parameter linking
- Manager class (fNPManager): Orchestrates the flexible combo implementation

Author: Chiara Bissolotti (cbissolotti@anl.gov)
Based on: fnp_base_flavor_dep.py
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Union, Tuple
import re
import math

# Try to import simpleeval for safe expression evaluation
try:
    from simpleeval import SimpleEval

    HAS_SIMPLEEVAL = True
except ImportError:
    HAS_SIMPLEEVAL = False
    print(
        "Warning: simpleeval not available. Complex expressions will use basic evaluation."
    )

# Import tcolors - handle both relative and absolute imports
try:
    from ..utilities.colors import tcolors
except ImportError:
    try:
        from utilities.colors import tcolors
    except ImportError:
        from sidis.utilities.colors import tcolors

# Import evolution and base classes from flavor_dep
from .fnp_base_flavor_dep import (
    fNP_evolution,
    MAP22_DEFAULT_EVOLUTION,
    MAP22_DEFAULT_PDF_PARAMS,
    MAP22_DEFAULT_FF_PARAMS,
)


###############################################################################
# 1. Parameter Link Parser
###############################################################################
class ParameterLinkParser:
    """
    Parser for free_mask entries that can contain:
    - Boolean values (true/false)
    - Simple references (u[0], d[1], pdfs.u[2], ffs.d[1], etc.)
    - Complex expressions (2*u[1] + 0.1, u[0] - d[0], etc.)
    """

    # Pattern to match parameter references: flavor[index] or type.flavor[index]
    PARAM_REF_PATTERN = re.compile(r"(\w+)\.(\w+)\[(\d+)\]|(\w+)\[(\d+)\]")

    @staticmethod
    def parse_entry(
        entry: Any, current_type: str, current_flavor: str
    ) -> Dict[str, Any]:
        """
        Parse a single free_mask entry.

        Args:
            entry: The entry to parse (bool, str, int, etc.)
            current_type: Current parameter type ('pdfs' or 'ffs')
            current_flavor: Current flavor name (e.g., 'u', 'd')

        Returns:
            Dict with keys:
                - 'type': 'boolean', 'reference', or 'expression'
                - 'value': boolean value, or reference info, or expression string
                - 'is_fixed': True if parameter is fixed (False)
        """
        # Handle boolean values
        if isinstance(entry, bool):
            return {"type": "boolean", "value": entry, "is_fixed": not entry}

        # Handle string/int representations of booleans
        if isinstance(entry, (str, int)):
            entry_str = str(entry).strip().lower()
            if entry_str in ("true", "1", "yes"):
                return {"type": "boolean", "value": True, "is_fixed": False}
            elif entry_str in ("false", "0", "no"):
                return {"type": "boolean", "value": False, "is_fixed": True}

        # Handle string expressions
        if isinstance(entry, str):
            entry_str = entry.strip()

            # Check if it's a simple reference (e.g., "u[0]" or "pdfs.u[0]")
            match = ParameterLinkParser.PARAM_REF_PATTERN.match(entry_str)
            if match and match.group(0) == entry_str:
                # Simple reference
                if match.group(1):  # type.flavor[index] format
                    param_type = match.group(1)
                    flavor = match.group(2)
                    param_idx = int(match.group(3))
                else:  # flavor[index] format (same type)
                    param_type = current_type
                    flavor = match.group(4)
                    param_idx = int(match.group(5))

                return {
                    "type": "reference",
                    "value": {
                        "type": param_type,
                        "flavor": flavor,
                        "param_idx": param_idx,
                    },
                    "is_fixed": False,
                }
            else:
                # Complex expression
                return {"type": "expression", "value": entry_str, "is_fixed": False}

        # Default: treat as fixed
        return {"type": "boolean", "value": False, "is_fixed": True}

    @staticmethod
    def extract_references(expression: str) -> List[Dict[str, Any]]:
        """
        Extract all parameter references from an expression.

        Args:
            expression: Expression string

        Returns:
            List of reference dicts with keys: 'type', 'flavor', 'param_idx', 'full_match'
        """
        references = []
        for match in ParameterLinkParser.PARAM_REF_PATTERN.finditer(expression):
            if match.group(1):  # type.flavor[index]
                references.append(
                    {
                        "type": match.group(1),
                        "flavor": match.group(2),
                        "param_idx": int(match.group(3)),
                        "full_match": match.group(0),
                    }
                )
            else:  # flavor[index]
                references.append(
                    {
                        "type": None,  # Will be resolved later
                        "flavor": match.group(4),
                        "param_idx": int(match.group(5)),
                        "full_match": match.group(0),
                    }
                )
        return references


###############################################################################
# 2. Parameter Registry
###############################################################################
class ParameterRegistry:
    """
    Registry to track and manage parameter objects across all flavors.
    Maps (type, flavor, param_idx) -> Parameter object.
    """

    def __init__(self):
        self.registry: Dict[Tuple[str, str, int], nn.Parameter] = {}
        self.shared_groups: Dict[Tuple[str, str, int], Tuple[str, str, int]] = {}

    def register_parameter(
        self,
        param_type: str,
        flavor: str,
        param_idx: int,
        param: nn.Parameter,
        source: Optional[Tuple[str, str, int]] = None,
    ):
        """
        Register a parameter in the registry.

        Args:
            param_type: 'pdfs' or 'ffs'
            flavor: Flavor name
            param_idx: Parameter index
            param: The Parameter object
            source: If this parameter is linked, the source (type, flavor, idx)
        """
        key = (param_type, flavor, param_idx)
        self.registry[key] = param

        if source:
            self.shared_groups[key] = source

    def get_parameter(
        self, param_type: str, flavor: str, param_idx: int
    ) -> Optional[nn.Parameter]:
        """
        Get a parameter from the registry.

        Args:
            param_type: 'pdfs' or 'ffs'
            flavor: Flavor name
            param_idx: Parameter index

        Returns:
            Parameter object or None if not found
        """
        key = (param_type, flavor, param_idx)

        # If this parameter is linked, return the source parameter
        if key in self.shared_groups:
            source_key = self.shared_groups[key]
            return self.registry.get(source_key)

        return self.registry.get(key)

    def create_shared_parameter(
        self, source_type: str, source_flavor: str, source_idx: int, init_value: float
    ) -> nn.Parameter:
        """
        Create or get a shared parameter.

        Args:
            source_type: Source parameter type
            source_flavor: Source flavor
            source_idx: Source parameter index
            init_value: Initial value

        Returns:
            Shared Parameter object
        """
        source_key = (source_type, source_flavor, source_idx)

        if source_key not in self.registry:
            # Create new shared parameter
            param = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))
            self.registry[source_key] = param
            return param
        else:
            # Return existing shared parameter
            return self.registry[source_key]


###############################################################################
# 3. Dependency Resolver
###############################################################################
class DependencyResolver:
    """
    Resolves parameter dependencies and handles circular dependencies.
    """

    @staticmethod
    def build_dependency_graph(
        config: Dict[str, Any], param_type: str, flavor_keys: List[str]
    ) -> Dict[Tuple[str, str, int], List[Tuple[str, str, int]]]:
        """
        Build dependency graph from configuration.

        Args:
            config: Configuration dictionary
            param_type: 'pdfs' or 'ffs'
            flavor_keys: List of flavor keys

        Returns:
            Dict mapping (type, flavor, idx) -> list of dependencies
        """
        graph = {}
        parser = ParameterLinkParser()

        type_config = config.get(param_type, {})

        for flavor in flavor_keys:
            flavor_cfg = type_config.get(flavor, {})
            free_mask = flavor_cfg.get("free_mask", [])

            for param_idx, entry in enumerate(free_mask):
                parsed = parser.parse_entry(entry, param_type, flavor)
                key = (param_type, flavor, param_idx)

                dependencies = []
                if parsed["type"] == "reference":
                    dep = parsed["value"]
                    # Resolve type if not specified
                    dep_type = dep["type"] if dep["type"] else param_type
                    dep_key = (dep_type, dep["flavor"], dep["param_idx"])
                    dependencies.append(dep_key)
                elif parsed["type"] == "expression":
                    # Extract all references from expression
                    refs = parser.extract_references(parsed["value"])
                    for ref in refs:
                        # Resolve type if not specified
                        ref_type = ref["type"] if ref["type"] else param_type
                        dep_key = (ref_type, ref["flavor"], ref["param_idx"])
                        dependencies.append(dep_key)

                graph[key] = dependencies

        return graph

    @staticmethod
    def resolve_circular_dependencies(
        graph: Dict[Tuple[str, str, int], List[Tuple[str, str, int]]],
    ) -> Dict[Tuple[str, str, int], Tuple[str, str, int]]:
        """
        Resolve circular dependencies by using the first definition encountered.

        Args:
            graph: Dependency graph

        Returns:
            Dict mapping dependent -> source (for circular dependencies)
        """
        resolved = {}
        visited = set()
        visiting = set()

        def dfs(node):
            if node in visiting:
                # Circular dependency detected - use first definition
                return node
            if node in visited:
                return None

            visiting.add(node)
            for dep in graph.get(node, []):
                source = dfs(dep)
                if source and source != node:
                    resolved[node] = source
            visiting.remove(node)
            visited.add(node)
            return None

        for node in graph:
            if node not in visited:
                dfs(node)

        return resolved


###############################################################################
# 4. Expression Evaluator
###############################################################################
class ExpressionEvaluator:
    """
    Safely evaluates mathematical expressions with parameter references.
    """

    def __init__(self, registry: ParameterRegistry):
        self.registry = registry
        if HAS_SIMPLEEVAL:
            self.evaluator = SimpleEval()
            # Register functions
            self.evaluator.functions.update(
                {
                    "exp": math.exp,
                    "log": math.log,
                    "sqrt": math.sqrt,
                    "sin": math.sin,
                    "cos": math.cos,
                    "tan": math.tan,
                }
            )
        else:
            self.evaluator = None

    def evaluate(
        self, expression: str, current_type: str, current_flavor: str
    ) -> torch.Tensor:
        """
        Evaluate an expression with current parameter values.

        Args:
            expression: Expression string
            current_type: Current parameter type
            current_flavor: Current flavor name

        Returns:
            Evaluated value as torch.Tensor
        """
        # Replace parameter references with actual values
        parser = ParameterLinkParser()
        refs = parser.extract_references(expression)

        # Build replacement dict
        replacements = {}
        for ref in refs:
            ref_type = ref["type"] if ref["type"] else current_type
            param = self.registry.get_parameter(
                ref_type, ref["flavor"], ref["param_idx"]
            )
            if param is not None:
                # Get current value - handle both scalar and tensor parameters
                if param.numel() == 1:
                    value = param.item()
                else:
                    value = param[0].item() if len(param.shape) > 0 else param.item()
                # Use parentheses to ensure proper evaluation order
                replacements[ref["full_match"]] = f"({value})"
            else:
                raise ValueError(
                    f"Parameter reference '{ref['full_match']}' not found in registry. "
                    f"Type: {ref_type}, Flavor: {ref['flavor']}, Index: {ref['param_idx']}"
                )

        # Replace in expression (replace longer matches first to avoid partial replacements)
        eval_expr = expression
        for old, new in sorted(replacements.items(), key=lambda x: -len(x[0])):
            eval_expr = eval_expr.replace(old, new)

        # Evaluate
        if HAS_SIMPLEEVAL:
            try:
                # Create a new evaluator instance for this expression
                evaluator = SimpleEval(
                    functions={
                        "exp": math.exp,
                        "log": math.log,
                        "sqrt": math.sqrt,
                        "sin": math.sin,
                        "cos": math.cos,
                        "tan": math.tan,
                    }
                )
                result = evaluator.eval(eval_expr)
            except Exception as e:
                raise ValueError(
                    f"Error evaluating expression '{expression}' (evaluated as '{eval_expr}'): {e}"
                )
        else:
            # Basic evaluation (less safe, but works for simple expressions)
            try:
                result = eval(
                    eval_expr,
                    {"__builtins__": {}},
                    {
                        "exp": math.exp,
                        "log": math.log,
                        "sqrt": math.sqrt,
                        "sin": math.sin,
                        "cos": math.cos,
                        "tan": math.tan,
                        "math": math,
                    },
                )
            except Exception as e:
                raise ValueError(
                    f"Error evaluating expression '{expression}' (evaluated as '{eval_expr}'): {e}"
                )

        return torch.tensor([float(result)], dtype=torch.float32)


###############################################################################
# 5. Flexible PDF Base Class
###############################################################################
class TMDPDFFlexible(nn.Module):
    """
    Flexible TMD PDF class with parameter linking support.
    """

    def __init__(
        self,
        flavor: str,
        init_params: List[float],
        free_mask: List[Any],
        registry: ParameterRegistry,
        evaluator: ExpressionEvaluator,
        param_type: str = "pdfs",
    ):
        super().__init__()

        if len(init_params) != 11:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_base_flexible.py] MAP22 TMD PDF requires 11 parameters, got {len(init_params)}{tcolors.ENDC}"
            )
        if len(free_mask) != len(init_params):
            raise ValueError(
                f"{tcolors.FAIL}[fnp_base_flexible.py] free_mask length ({len(free_mask)}) must match init_params length ({len(init_params)}){tcolors.ENDC}"
            )

        self.flavor = flavor
        self.param_type = param_type
        self.n_params = len(init_params)
        self.registry = registry
        self.evaluator = evaluator
        self.parser = ParameterLinkParser()

        # Reference point x_hat = 0.1 (MAP22 standard)
        self.register_buffer("x_hat", torch.tensor(0.1, dtype=torch.float32))

        # Parse free_mask entries
        self.param_configs = []
        self.fixed_params = []
        self.free_params_list = []

        for param_idx, (init_val, entry) in enumerate(zip(init_params, free_mask)):
            parsed = self.parser.parse_entry(entry, param_type, flavor)
            self.param_configs.append(
                {"idx": param_idx, "init_val": init_val, "parsed": parsed}
            )

            if parsed["is_fixed"]:
                # Fixed parameter
                self.fixed_params.append((param_idx, init_val))
            elif parsed["type"] == "boolean" and parsed["value"]:
                # Independent free parameter
                param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
                self.free_params_list.append((param_idx, param))
                registry.register_parameter(param_type, flavor, param_idx, param)
            elif parsed["type"] == "reference":
                # Linked parameter - use shared parameter
                ref = parsed["value"]
                ref_type = ref["type"] if ref["type"] else param_type
                shared_param = registry.create_shared_parameter(
                    ref_type, ref["flavor"], ref["param_idx"], init_val
                )
                self.free_params_list.append((param_idx, shared_param))
                registry.register_parameter(
                    param_type,
                    flavor,
                    param_idx,
                    shared_param,
                    source=(ref_type, ref["flavor"], ref["param_idx"]),
                )
            elif parsed["type"] == "expression":
                # Expression-based parameter - will be evaluated dynamically
                # Store expression and create a placeholder parameter for gradient flow
                param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
                self.free_params_list.append((param_idx, param))
                registry.register_parameter(param_type, flavor, param_idx, param)
                # Store expression for dynamic evaluation
                parsed["expression"] = parsed["value"]

        # Register fixed parameters as buffers
        for param_idx, val in self.fixed_params:
            self.register_buffer(
                f"fixed_param_{param_idx}", torch.tensor([val], dtype=torch.float32)
            )

        # Register free parameters
        for param_idx, param in self.free_params_list:
            self.register_parameter(f"free_param_{param_idx}", param)

    def get_params_tensor(self) -> torch.Tensor:
        """Return the full parameter tensor, evaluating expressions dynamically."""
        params = [0.0] * self.n_params

        # Set fixed parameters
        for param_idx, val in self.fixed_params:
            params[param_idx] = val

        # Set free parameters (including linked and expression-based)
        for param_idx, param in self.free_params_list:
            config = self.param_configs[param_idx]
            parsed = config["parsed"]

            if parsed["type"] == "boolean" or parsed["type"] == "reference":
                # Direct or linked parameter
                if param.numel() == 1:
                    params[param_idx] = param.item()
                else:
                    params[param_idx] = (
                        param[0].item() if len(param.shape) > 0 else param.item()
                    )
            elif parsed["type"] == "expression":
                # Evaluate expression dynamically
                expr_value = self.evaluator.evaluate(
                    parsed["expression"], self.param_type, self.flavor
                )
                params[param_idx] = expr_value.item()
                # Update the parameter for gradient tracking
                param.data = expr_value

        return torch.tensor(params, dtype=torch.float32)

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        """Compute TMD PDF using MAP22 parameterization."""
        # Ensure x can broadcast with b (x: [n_events], b: [n_events, n_b])
        if b.dim() > x.dim():
            x = x.unsqueeze(-1)

        # Handle x >= 1 case (return zero)
        if torch.any(x >= 1):
            mask_val = (x < 1).type_as(b)
        else:
            mask_val = torch.ones_like(x)

        # Get parameters (evaluates expressions dynamically)
        p = self.get_params_tensor()

        # Extract parameters (MAP22 order)
        N1 = p[0]
        alpha1 = p[1]
        sigma1 = p[2]
        lam = p[3]
        N1B = p[4]
        N1C = p[5]
        lam2 = p[6]
        alpha2 = p[7]
        alpha3 = p[8]
        sigma2 = p[9]
        sigma3 = p[10]

        # Compute intermediate g-functions (MAP22 exact implementation)
        g1 = (
            N1
            * torch.pow(x / self.x_hat, sigma1)
            * torch.pow((1 - x) / (1 - self.x_hat), alpha1**2)
        )
        g1B = (
            N1B
            * torch.pow(x / self.x_hat, sigma2)
            * torch.pow((1 - x) / (1 - self.x_hat), alpha2**2)
        )
        g1C = (
            N1C
            * torch.pow(x / self.x_hat, sigma3)
            * torch.pow((1 - x) / (1 - self.x_hat), alpha3**2)
        )

        # Compute (b/2)² term
        b_half_sq = (b / 2.0) ** 2

        # Numerator (exact MAP22 formula)
        numerator = (
            g1 * torch.exp(-g1 * b_half_sq)
            + (lam**2) * (g1B**2) * (1 - g1B * b_half_sq) * torch.exp(-g1B * b_half_sq)
            + g1C * (lam2**2) * torch.exp(-g1C * b_half_sq)
        )

        # Denominator (exact MAP22 formula)
        denominator = g1 + (lam**2) * (g1B**2) + g1C * (lam2**2)

        # Complete TMD PDF (evolution factor applied in manager)
        result = numerator / denominator

        return result * mask_val


###############################################################################
# 6. Flexible FF Base Class
###############################################################################
class TMDFFFlexible(nn.Module):
    """
    Flexible TMD FF class with parameter linking support.
    """

    def __init__(
        self,
        flavor: str,
        init_params: List[float],
        free_mask: List[Any],
        registry: ParameterRegistry,
        evaluator: ExpressionEvaluator,
        param_type: str = "ffs",
    ):
        super().__init__()

        if len(init_params) != 9:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_base_flexible.py] MAP22 TMD FF requires 9 parameters, got {len(init_params)}{tcolors.ENDC}"
            )
        if len(free_mask) != len(init_params):
            raise ValueError(
                f"{tcolors.FAIL}[fnp_base_flexible.py] free_mask length ({len(free_mask)}) must match init_params length ({len(init_params)}){tcolors.ENDC}"
            )

        self.flavor = flavor
        self.param_type = param_type
        self.n_params = len(init_params)
        self.registry = registry
        self.evaluator = evaluator
        self.parser = ParameterLinkParser()

        # Reference point z_hat = 0.5 (MAP22 standard)
        self.register_buffer("z_hat", torch.tensor(0.5, dtype=torch.float32))

        # Parse free_mask entries (same logic as PDF)
        self.param_configs = []
        self.fixed_params = []
        self.free_params_list = []

        for param_idx, (init_val, entry) in enumerate(zip(init_params, free_mask)):
            parsed = self.parser.parse_entry(entry, param_type, flavor)
            self.param_configs.append(
                {"idx": param_idx, "init_val": init_val, "parsed": parsed}
            )

            if parsed["is_fixed"]:
                self.fixed_params.append((param_idx, init_val))
            elif parsed["type"] == "boolean" and parsed["value"]:
                param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
                self.free_params_list.append((param_idx, param))
                registry.register_parameter(param_type, flavor, param_idx, param)
            elif parsed["type"] == "reference":
                ref = parsed["value"]
                ref_type = ref["type"] if ref["type"] else param_type
                shared_param = registry.create_shared_parameter(
                    ref_type, ref["flavor"], ref["param_idx"], init_val
                )
                self.free_params_list.append((param_idx, shared_param))
                registry.register_parameter(
                    param_type,
                    flavor,
                    param_idx,
                    shared_param,
                    source=(ref_type, ref["flavor"], ref["param_idx"]),
                )
            elif parsed["type"] == "expression":
                param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
                self.free_params_list.append((param_idx, param))
                registry.register_parameter(param_type, flavor, param_idx, param)
                parsed["expression"] = parsed["value"]

        # Register fixed parameters as buffers
        for param_idx, val in self.fixed_params:
            self.register_buffer(
                f"fixed_param_{param_idx}", torch.tensor([val], dtype=torch.float32)
            )

        # Register free parameters
        for param_idx, param in self.free_params_list:
            self.register_parameter(f"free_param_{param_idx}", param)

    def get_params_tensor(self) -> torch.Tensor:
        """Return the full parameter tensor, evaluating expressions dynamically."""
        params = [0.0] * self.n_params

        for param_idx, val in self.fixed_params:
            params[param_idx] = val

        for param_idx, param in self.free_params_list:
            config = self.param_configs[param_idx]
            parsed = config["parsed"]

            if parsed["type"] == "boolean" or parsed["type"] == "reference":
                if param.numel() == 1:
                    params[param_idx] = param.item()
                else:
                    params[param_idx] = (
                        param[0].item() if len(param.shape) > 0 else param.item()
                    )
            elif parsed["type"] == "expression":
                expr_value = self.evaluator.evaluate(
                    parsed["expression"], self.param_type, self.flavor
                )
                params[param_idx] = expr_value.item()
                param.data = expr_value

        return torch.tensor(params, dtype=torch.float32)

    def forward(
        self,
        z: torch.Tensor,
        b: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        """Compute TMD FF using MAP22 parameterization."""
        # Ensure z can broadcast with b (z: [n_events], b: [n_events, n_b])
        if b.dim() > z.dim():
            z = z.unsqueeze(-1)

        # Handle z >= 1 case (return zero)
        if torch.any(z >= 1):
            mask_val = (z < 1).type_as(b)
        else:
            mask_val = torch.ones_like(z)

        # Get parameters (evaluates expressions dynamically)
        p = self.get_params_tensor()

        # Extract parameters (MAP22 order)
        N3 = p[0]
        beta1 = p[1]
        delta1 = p[2]
        gamma1 = p[3]
        lambdaF = p[4]
        N3B = p[5]
        beta2 = p[6]
        delta2 = p[7]
        gamma2 = p[8]

        # Compute intermediate functions (MAP22 exact implementation)
        cmn1 = (
            (z**beta1 + delta1**2) / (self.z_hat**beta1 + delta1**2)
        ) * torch.pow((1 - z) / (1 - self.z_hat), gamma1**2)
        cmn2 = (
            (z**beta2 + delta2**2) / (self.z_hat**beta2 + delta2**2)
        ) * torch.pow((1 - z) / (1 - self.z_hat), gamma2**2)

        g3 = N3 * cmn1
        g3B = N3B * cmn2

        # z² factor
        z2 = z * z

        # Compute (b/2)² term
        b_half_sq = (b / 2.0) ** 2

        # Numerator (exact MAP22 formula)
        numerator = g3 * torch.exp(-g3 * b_half_sq / z2) + (lambdaF / z2) * (
            g3B**2
        ) * (1 - g3B * b_half_sq / z2) * torch.exp(-g3B * b_half_sq / z2)

        # Denominator (exact MAP22 formula)
        denominator = g3 + (lambdaF / z2) * (g3B**2)

        # Complete TMD FF (evolution factor applied in manager)
        result = numerator / denominator

        return result * mask_val


###############################################################################
# 7. Flexible Manager Class
###############################################################################
class fNPManager(nn.Module):
    """
    Manager for flexible fNP system with parameter linking.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.hadron = config.get("hadron", "proton")
        self.pdf_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]
        self.ff_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]

        print(
            f"{tcolors.BLUE}\n[fNPManager] Initializing flexible fNP manager with parameter linking"
        )
        print(f"  Hadron: {self.hadron}")
        print(f"  Total number of flavors: {len(self.pdf_flavor_keys)}\n{tcolors.ENDC}")

        # Setup evolution (independent, no linking)
        evolution_config = config.get("evolution", {})
        init_g2 = evolution_config.get("init_g2", 0.12840)
        free_mask = evolution_config.get("free_mask", [True])
        self.evolution = fNP_evolution(init_g2=init_g2, free_mask=free_mask)

        # Initialize parameter registry and evaluator
        self.registry = ParameterRegistry()
        self.evaluator = ExpressionEvaluator(self.registry)

        # Build dependency graphs
        pdf_graph = DependencyResolver.build_dependency_graph(
            config, "pdfs", self.pdf_flavor_keys
        )
        ff_graph = DependencyResolver.build_dependency_graph(
            config, "ffs", self.ff_flavor_keys
        )

        # Resolve circular dependencies
        pdf_resolved = DependencyResolver.resolve_circular_dependencies(pdf_graph)
        ff_resolved = DependencyResolver.resolve_circular_dependencies(ff_graph)

        # Setup PDF modules
        pdf_config = config.get("pdfs", {})
        pdf_modules = {}
        for flavor in self.pdf_flavor_keys:
            flavor_cfg = pdf_config.get(flavor, None)
            if flavor_cfg is None:
                print(
                    f"{tcolors.WARNING}[fNPManager] Warning: Using MAP22 defaults for PDF flavor '{flavor}'{tcolors.ENDC}"
                )
                flavor_cfg = MAP22_DEFAULT_PDF_PARAMS.copy()
            else:
                print(
                    f"{tcolors.OKLIGHTBLUE}[fNPManager] Using user-defined PDF flavor '{flavor}'{tcolors.ENDC}"
                )

            pdf_modules[flavor] = TMDPDFFlexible(
                flavor=flavor,
                init_params=flavor_cfg["init_params"],
                free_mask=flavor_cfg["free_mask"],
                registry=self.registry,
                evaluator=self.evaluator,
                param_type="pdfs",
            )

        self.pdf_modules = nn.ModuleDict(pdf_modules)
        print(
            f"{tcolors.GREEN}[fNPManager] Initialized {len(self.pdf_modules)} PDF flavor modules\n{tcolors.ENDC}"
        )

        # Setup FF modules
        ff_config = config.get("ffs", {})
        ff_modules = {}
        for flavor in self.ff_flavor_keys:
            flavor_cfg = ff_config.get(flavor, None)
            if flavor_cfg is None:
                print(
                    f"{tcolors.WARNING}[fNPManager] Warning: Using MAP22 defaults for FF flavor '{flavor}'{tcolors.ENDC}"
                )
                flavor_cfg = MAP22_DEFAULT_FF_PARAMS.copy()
            else:
                print(
                    f"{tcolors.OKLIGHTBLUE}[fNPManager] Using user-defined FF flavor '{flavor}'{tcolors.ENDC}"
                )

            ff_modules[flavor] = TMDFFFlexible(
                flavor=flavor,
                init_params=flavor_cfg.get(
                    "init_params", MAP22_DEFAULT_FF_PARAMS["init_params"]
                ),
                free_mask=flavor_cfg.get(
                    "free_mask", MAP22_DEFAULT_FF_PARAMS["free_mask"]
                ),
                registry=self.registry,
                evaluator=self.evaluator,
                param_type="ffs",
            )

        self.ff_modules = nn.ModuleDict(ff_modules)
        print(
            f"{tcolors.GREEN}[fNPManager] Initialized {len(self.ff_modules)} FF flavor modules\n{tcolors.ENDC}"
        )

    def _compute_zeta(self, Q: torch.Tensor) -> torch.Tensor:
        """Compute rapidity scale zeta from hard scale Q."""
        return Q**2

    def forward_pdf(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        Q: torch.Tensor,
        flavors: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate TMD PDFs for specified flavors."""
        if flavors is None:
            flavors = self.pdf_flavor_keys

        zeta = self._compute_zeta(Q)
        shared_evol = self.evolution(b, zeta)

        outputs = {}
        for flavor in flavors:
            if flavor in self.pdf_modules:
                base_result = self.pdf_modules[flavor](x, b, 0)
                outputs[flavor] = base_result * shared_evol
            else:
                raise ValueError(f"Unknown PDF flavor: {flavor}")

        return outputs

    def forward_ff(
        self,
        z: torch.Tensor,
        b: torch.Tensor,
        Q: torch.Tensor,
        flavors: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate TMD FFs for specified flavors."""
        if flavors is None:
            flavors = self.ff_flavor_keys

        zeta = self._compute_zeta(Q)
        shared_evol = self.evolution(b, zeta)

        outputs = {}
        for flavor in flavors:
            if flavor in self.ff_modules:
                base_result = self.ff_modules[flavor](z, b, 0)
                outputs[flavor] = base_result * shared_evol
            else:
                raise ValueError(f"Unknown FF flavor: {flavor}")

        return outputs

    def forward_sivers(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate Sivers function.

        Note: Sivers function support is not yet implemented in the flexible model.
        This method returns zeros with the correct shape as a placeholder.

        Args:
            x (torch.Tensor): Bjorken x values (1D: [n_events])
            b (torch.Tensor): Impact parameter values (2D: [n_events, n_b])
            Q (torch.Tensor): Hard scale Q in GeV (1D: [n_events], used to compute zeta = Q²)

        Returns:
            torch.Tensor: Sivers function values (currently zeros as placeholder).
                         Shape: [n_events, n_b] (same as b)
        """
        # Ensure x can broadcast with b
        if b.dim() > x.dim():
            x = x.unsqueeze(-1)

        # Compute zeta from Q (zeta = Q² for standard SIDIS)
        zeta = self._compute_zeta(Q)

        # Compute shared evolution factor
        shared_evol = self.evolution(b, zeta)

        # TODO: Implement Sivers function with parameter linking support
        # For now, return zeros with correct shape
        return torch.zeros_like(b) * shared_evol

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        Q: torch.Tensor,
        pdf_flavors: Optional[List[str]] = None,
        ff_flavors: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Evaluate both TMD PDFs and FFs simultaneously."""
        x = x.unsqueeze(-1)
        z = z.unsqueeze(-1)
        return {
            "pdfs": self.forward_pdf(x, b, Q, pdf_flavors),
            "ffs": self.forward_ff(z, b, Q, ff_flavors),
        }
