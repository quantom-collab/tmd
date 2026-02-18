"""
Config parsing utilities for fNP models.

This module provides classes and functions for reading and parsing fNP configuration
cards, independent of any specific parametrization. All models (simple, flexible, etc.)
can import these utilities.

Contents:
- parse_bound: Parse [lo, hi] interval from config (handles OmegaConf ListConfig)
- ParameterLinkParser: Parses free_mask entries (boolean, reference, expression)
- ParameterRegistry: Tracks linked parameters and manages shared Parameter objects
- DependencyResolver: Builds dependency graph and resolves circular dependencies
- ExpressionEvaluator: Safely evaluates mathematical expressions with parameter refs

Author: Chiara Bissolotti (cbissolotti@anl.gov)
"""

import re
import math
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

# Try to import simpleeval for safe expression evaluation
try:
    from simpleeval import SimpleEval

    HAS_SIMPLEEVAL = True
except Exception as err:
    HAS_SIMPLEEVAL = False
    print(
        "Warning: simpleeval not available. Complex expressions will use basic evaluation.",
        f"(Import error: {err})",
    )

try:
    from ..utilities.colors import tcolors
except ImportError:
    try:
        from utilities.colors import tcolors
    except ImportError:
        from sidis.utilities.colors import tcolors


###############################################################################
# 1. Bound parsing
###############################################################################
def parse_bound(b: Any) -> Optional[Tuple[float, float]]:
    """
    Parse a bound entry [lo, hi] from config (handles OmegaConf ListConfig).

    Args:
        b: Bound entry (list, tuple, or OmegaConf ListConfig)

    Returns:
        (lo, hi) tuple or None if invalid
    """
    # If the bound is missing, return None immediately. This indicates
    # that the parameter is not bounded.
    if b is None:
        return None
    try:
        # Check whether b has and __iter__ method, i.e. whether is
        # iterable, and whether it is not a string. If so, convert it to a list.
        # Exclude strings. Strings are iterable (you can do list("ab") → ['a','b']),
        # but we don’t want "0.1" to become ['0','.','1'].
        seq = list(b) if hasattr(b, "__iter__") and not isinstance(b, str) else b
        # Check whether seq has a length and at least two elements.
        # We need [lo, hi], so fewer than two elements is invalid and we return None.
        if not hasattr(seq, "__len__") or len(seq) < 2:
            return None
        # Convert the first and second elements to floats and return them as a tuple.
        return (float(seq[0]), float(seq[1]))
    # If the conversion to floats fails, e.g. there are non-numeric characters,
    # print an error message and return None.
    except (TypeError, ValueError):
        print(f"{tcolors.FAIL}[fnp_config.py] Error parsing bound: {b}{tcolors.ENDC}")
        return None


###############################################################################
# 2. Parameter Link Parser
###############################################################################
class ParameterLinkParser:
    """
    Parser for free_mask entries that can contain:
    - Boolean values (true/false)
    - Simple references (u[0], d[1], pdfs.u[2], ffs.d[1], etc.)
    - Complex expressions (2*u[1] + 0.1, u[0] - d[0], etc.)
    """

    # Class attribute: compiled regex (regular expression) pattern, shared by all instances.
    # re.compile() pre-compiles the pattern for efficient repeated matching.
    # Pattern has two alternatives (|):
    #   - (\w+)\.(\w+)\[(\d+)\]  matches "type.flavor[index]" (e.g. pdfs.u[0])
    #     \w+ = word chars (type), \. = literal dot, \w+ = flavor, \[\d+\] = [digits]
    #   - (\w+)\[(\d+)\]        matches "flavor[index]" (e.g. u[0]) when in same type
    # Groups 1,2,3 for first form; groups 4,5 for second
    PARAM_REF_PATTERN = re.compile(r"(\w+)\.(\w+)\[(\d+)\]|(\w+)\[(\d+)\]")

    # @staticmethod: method belongs to the class, not to instances.
    # No 'self' argument; can be called as ParameterLinkParser.parse_entry(...)
    # NOTE: current_flavor is passed into parse_entry but never used in the function body.
    # TODO: Could be used later for validation (e.g. checking that a reference is valid
    # TODO: for the current flavor) or clearer error messages.
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
        # isinstance(entry, bool): check exact type bool (True/False from YAML).
        # Return dict with type, value, and is_fixed (fixed = not free = value is False).
        if isinstance(entry, bool):
            return {"type": "boolean", "value": entry, "is_fixed": not entry}

        # Handle YAML/string forms of booleans (e.g. "true", 1, "yes"). Either a string or an integer.
        # str(entry).strip().lower(): normalize to lowercase string for comparison.
        # Tuple ("true", "1", "yes"): 'in' checks membership in this tuple.
        if isinstance(entry, (str, int)):
            entry_str = str(entry).strip().lower()
            if entry_str in ("true", "1", "yes"):
                return {"type": "boolean", "value": True, "is_fixed": False}
            elif entry_str in ("false", "0", "no"):
                return {"type": "boolean", "value": False, "is_fixed": True}

        # Handle string expressions: references or formulas.
        if isinstance(entry, str):
            entry_str = entry.strip()

            # .match() tries to match the pattern only at the beginning of the
            # string (position 0). It does not look for the pattern anywhere else.
            # match.group(0) is the full matched substring.
            # match.group(0) == entry_str: entire string must match (no extra chars).
            match = ParameterLinkParser.PARAM_REF_PATTERN.match(entry_str)
            if match and match.group(0) == entry_str:
                # Simple reference: whole string is "flavor[idx]" or "type.flavor[idx]".
                # match.group(1) is truthy only for "type.flavor[index]" form.
                if match.group(1):  # type.flavor[index] format
                    param_type = match.group(1)  # e.g. "pdfs"
                    flavor = match.group(2)  # e.g. "u"
                    param_idx = int(match.group(3))  # e.g. 0
                else:  # flavor[index] format: use current_type as type
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
                # Complex expression: contains operators, multiple refs, etc.
                return {"type": "expression", "value": entry_str, "is_fixed": False}

        # Fallback: unknown type (e.g. None, float) -> treat as fixed parameter.
        print(f"{tcolors.FAIL}[fnp_config.py] Unknown type: {entry}{tcolors.ENDC}")
        return {"type": "boolean", "value": False, "is_fixed": True}

    @staticmethod
    def extract_references(expression: str) -> List[Dict[str, Any]]:
        """
        Extract all parameter references from an expression. It is used
        after an entry is known to be an expression. It is used to :
         - build the dependency graph
         - evaluate the expression

        Args:
            expression: Expression string

        Returns:
            List of reference dicts with keys: 'type', 'flavor', 'param_idx', 'full_match'
        """
        # List to collect all reference dicts found in the expression.
        references = []

        # .finditer() returns an iterator of match objects for all non-overlapping
        # occurrences of the pattern in the string (unlike .match() which only checks start).
        for match in ParameterLinkParser.PARAM_REF_PATTERN.finditer(expression):
            # Same group logic as parse_entry: group 1 set => type.flavor[idx] form.
            if match.group(1):  # type.flavor[index]
                references.append(
                    {
                        "type": match.group(1),
                        "flavor": match.group(2),
                        "param_idx": int(match.group(3)),
                        "full_match": match.group(
                            0
                        ),  # original substring, e.g. "pdfs.u[0]"
                    }
                )
            else:  # flavor[index] form; type resolved later using context
                references.append(
                    {
                        "type": None,  # Caller uses current_type when evaluating
                        "flavor": match.group(4),
                        "param_idx": int(match.group(5)),
                        "full_match": match.group(0),
                    }
                )
        return references


###############################################################################
# 3. Parameter Registry
###############################################################################
class ParameterRegistry:
    """
    Registry to track and manage parameter objects across all flavors.
    Maps (type, flavor, param_idx) -> Parameter object.
    """

    def __init__(self):
        # registry: main storage for all parameters
        # Maps (param_type, flavor, param_idx) → nn.Parameter
        # Every parameter (independent or linked) is stored here.
        # Example: ("pdfs", "u", 0) → the nn.Parameter for the u-quark PDF parameter 0.
        self.registry: Dict[Tuple[str, str, int], nn.Parameter] = {}
        # shared_groups: redirect map for linked parameters
        # Maps (param_type, flavor, param_idx) → (param_type, flavor, param_idx)
        # Every shared parameter is stored here. Used only for parameters that are
        # linked to another (e.g. d[0] = "0.5*u[0]").
        # Example: ("pdfs", "u", 0) → ("pdfs", "u", 0) means “d’s param 0 uses u’s param 0”.
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
        When you call e.g. get_parameter("pdfs", "d", 0):
            - If ("pdfs", "d", 0) is in shared_groups, you look up the source
              key and return that parameter from registry. This is the linked parameter.
            - Otherwise you return the parameter for that key directly from registry.

        Args:
            param_type: 'pdfs' or 'ffs'
            flavor: Flavor name
            param_idx: Parameter index

        Returns:
            Parameter object or None if not found (this is why nn.Parameter is Optional
            in the return type).
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
        Create the source parameter if it does not exist, or return
        the existing one if it does. Used when setting up linked parameters
        (e.g. d[0] = "0.5*u[0]").

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
# 4. Dependency Resolver
###############################################################################
class DependencyResolver:
    """
    Builds and analyzes the dependency graph of fNP parameters.

    When parameters are linked (e.g. d[0] = "0.5*u[0]") or use expressions
    (e.g. d[1] = "u[0] + u[1]"), they depend on other parameters. This class
    builds that graph and resolves circular dependencies so that evaluation
    order is well-defined.
    """

    # @staticmethod: method belongs to the class, not to instances.
    # No 'self' argument is automatically passed; can be called as DependencyResolver.build_dependency_graph(...)
    @staticmethod
    def build_dependency_graph(
        config: Dict[str, Any], param_type: str, flavor_keys: List[str]
    ) -> Dict[Tuple[str, str, int], List[Tuple[str, str, int]]]:
        """
        Build dependency graph from configuration.

        For each parameter (type, flavor, param_idx), determines which other
        parameters it depends on (references or expressions that mention them).

        Args:
            config: Full fNP config dict (OmegaConf or plain dict).
            param_type: 'pdfs' or 'ffs' - which block to process.
            flavor_keys: List of flavor names to iterate (e.g. ['u','d','s',...]).

        Returns:
            Dict mapping (type, flavor, idx) -> list of (type, flavor, idx)
            dependencies. Empty list means no dependencies (independent param).
        """
        graph = {}
        parser = ParameterLinkParser()

        # Get the pdfs or ffs section from config.
        type_config = config.get(param_type, {})

        for flavor in flavor_keys:
            flavor_cfg = type_config.get(flavor, {})
            free_mask = flavor_cfg.get("free_mask", [])

            # Iterate over the free_mask entries, as they are
            # the ones that define the dependencies between parameters.
            for param_idx, entry in enumerate(free_mask):
                # Parse the single entry to get the type, value,
                # and is_fixed.
                parsed = parser.parse_entry(entry, param_type, flavor)
                # Create the key for the parameter.
                key = (param_type, flavor, param_idx)

                # Initialize the list of dependencies.
                dependencies = []
                if parsed["type"] == "reference":
                    # Simple ref: "u[0]" or "pdfs.u[0]" -> single dependency.
                    # Extract the dependency information from the parsed entry.
                    dep = parsed["value"]
                    dep_type = dep["type"] if dep["type"] else param_type
                    dep_key = (dep_type, dep["flavor"], dep["param_idx"])
                    dependencies.append(dep_key)
                elif parsed["type"] == "expression":
                    # Expression: "0.5*u[0] + d[1]" -> multiple dependencies.
                    # Extract the dependencies from the expression.
                    refs = parser.extract_references(parsed["value"])
                    for ref in refs:
                        ref_type = ref["type"] if ref["type"] else param_type
                        dep_key = (ref_type, ref["flavor"], ref["param_idx"])
                        dependencies.append(dep_key)

                # Add the dependencies to the graph.
                # The key is the parameter we are currently processing,
                # and the value is a list of parameters it depends on.
                graph[key] = dependencies

        return graph

    @staticmethod
    def resolve_circular_dependencies(
        graph: Dict[Tuple[str, str, int], List[Tuple[str, str, int]]],
    ) -> Dict[Tuple[str, str, int], Tuple[str, str, int]]:
        """
        Detect circular dependencies and choose a canonical source per cycle.

        If A depends on B and B depends on A, we pick one as the "source" and
        treat the other as linked to it. Uses Depth-First Search (DFS): when we re-enter a node
        that is still in the current path (visiting), we have a cycle.

        Args:
            graph: Output of build_dependency_graph.

        Returns:
            Dict mapping (dependent_key) -> (source_key) for params involved
            in circular dependencies. Use this to redirect dependent -> source.
        """
        resolved = {}
        visited = set()  # Nodes we have finished processing.
        visiting = set()  # Nodes in the current DFS path (not yet finished).

        # Define the Depth-First Search function.
        def dfs(node):
            """
            Depth-First Search function.
            The dependency graph has nodes (parameters) and edges (A depends on B).
            DFS is used to walk from each parameter to the ones it depends on:
            - Start at a node (e.g. parameter d[0]).
            - Follow edges to its dependencies (e.g. u[0]).
            - Recursively visit each dependency before returning.
            - Track which nodes are on the current path (visiting) vs already finished (visited).
            Why it matters for cycles:
                - If you reach a node that is already in visiting, you’ve found a back edge:
                  the current path forms a cycle (e.g. A → B → C → A).
                  That’s how the code detects circular dependencies.

            Args:
                node: The current node to process.
            Returns:
                The source node for the cycle, or None if there is no cycle.
            """

            if node in visiting:
                # Back edge: we've returned to a node on the current path.
                # That node is the start of a cycle; use it as the source.
                return node
            if node in visited:
                # Already processed in a previous DFS; nothing to do.
                return None

            visiting.add(node)
            for dep in graph.get(node, []):
                source = dfs(dep)
                # If dep is in a cycle, source is the cycle's root.
                # Record that node depends on source (for cycle resolution).
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
# 5. Expression Evaluator
###############################################################################
class ExpressionEvaluator:
    """
    Safely evaluates mathematical expressions with parameter references.
    """

    def __init__(self, registry: ParameterRegistry):
        # ParameterRegistry: needed to resolve refs like "u[0]" or "pdfs.u[0]" to their
        # current values. evaluate() looks up each ref in the registry and substitutes
        # the numeric value before evaluating the expression. The registry holds the
        # nn.Parameter objects that are updated during training.
        self.registry = registry

        # If simpleeval is available, use it to evaluate the expressions.
        if HAS_SIMPLEEVAL:
            self.evaluator = SimpleEval()
            # SimpleEval restricts allowed functions for security (avoids eval of arbitrary code).
            # By default it does not expose math functions. We register exp, log, sqrt, etc.
            # so that expressions like "0.5*exp(u[0])" or "sqrt(u[0]+1)" work in the config.
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
            # If simpleeval is not available, use basic evaluation. The fallback is Python's
            # eval() function. It is less safe, but works for simple expressions.
            # self.evaluator = None is effectively unused; the behavior is driven by HAS_SIMPLEEVAL
            # (see function below)
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
