from .bonsai_node import BonsaiTernaryNode

NODE_CLASS_MAPPINGS = {
    "BonsaiTernaryNode": BonsaiTernaryNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BonsaiTernaryNode": "🌳 Bonsai 4B (Gemlite 2-bit)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']