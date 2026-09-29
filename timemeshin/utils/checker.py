class MalformedArchitectureError(Exception):
    """Raised when the layer sequence breaks gradient or information flow laws."""
    pass

def verify_timemeshin_layout(architecture_class):
    """
    Decorator that verifies strict structural layout alignment.
    Ensures that discrete boundaries do not break gradient or state continuity.
    """
    original_init = architecture_class.__init__

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "layer_execution_sequence"):
            return

        sequence = [layer.__class__.__name__ for layer in self.layer_execution_sequence]

        for idx, layer_name in enumerate(sequence):
            if layer_name == "QRCodeLayer":
                if idx > 0 and "OuijaBoardLayer" in sequence[:idx]:
                    raise MalformedArchitectureError(
                        f"CRITICAL COMPILATION FAULT: Poorly placed QR block at index {idx} "
                        f"sitting after an Ouija state vector pointer. This will cause catastrophic history loss."
                    )
                if idx == len(sequence) - 1:
                    raise MalformedArchitectureError(
                        "COMPILATION BLOCKED: A discrete boundary layer cannot sit at the absolute terminal "
                        "point without a downstream smooth analytical reconstruction head."
                    )

    architecture_class.__init__ = wrapped_init
    return architecture_class
