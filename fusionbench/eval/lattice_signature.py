"""Truth-table signatures for scalar, identity-postprocessed OKAPI MIN/MAX trees."""

from collections.abc import Hashable, Iterable, Mapping


def provider_signatures(names: Iterable[Hashable]) -> dict[Hashable, int]:
    """Encode K providers on all corners, with provider i in bit i of the corner index.

    Bit s of a returned integer is the provider's binary value at corner s.
    Up to 12 providers use at most 4096 bits per integer.
    """
    names = tuple(names)
    if not 1 <= len(names) <= 12:
        raise ValueError("Expected 1 through 12 provider names")
    if len(set(names)) != len(names):
        raise ValueError("Provider names must be unique")

    corners = 1 << len(names)
    signatures = {}
    for index, name in enumerate(names):
        width = 1 << index
        ones = (1 << width) - 1
        signatures[name] = sum(
            ones << offset for offset in range(width, corners, 2 * width)
        )
    return signatures


def lattice_signature(tree, signatures: Mapping[Hashable, int]) -> int:
    """Evaluate every binary corner of a native MIN/MAX tree with bitwise operations.

    Pass the same provider_signatures mapping to compare trees. For finite real
    provider values, thresholding commutes with MIN and MAX. Thus equality of
    all corner bits proves equality on every input under identity postprocessing.
    """
    from okapi.node import MaxNode, MinNode, ValueNode

    def evaluate(node):
        if type(node) is not ValueNode:
            raise TypeError("Expected a built-in OKAPI ValueNode")
        result = signatures[node.id]
        # Each operator consumes the previous result of its parent value node.
        for op in node.children:
            if type(op) is MinNode:
                for child in op.children:
                    result &= evaluate(child)
            elif type(op) is MaxNode:
                for child in op.children:
                    result |= evaluate(child)
            else:
                raise TypeError(f"Unsupported lattice operator {type(op).__name__}")
        return result

    return evaluate(tree.root)
