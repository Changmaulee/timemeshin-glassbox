import torch
import torch.nn as nn

class TimeMeshinTimelineEngine(nn.Module):
    """
    Multi-Scale State Space Timeline Engine implementing Timescale Hierarchy.
    - I-Frames: Slow clock (tau_macro), immutable anchor point saved to Flight Recorder.
    - B-Frames: Fast clock (tau_delta), high-sensitivity local delta tracking.
    """
    def __init__(self, dim: int = 256, tau_macro: float = 0.95, tau_delta: float = 0.40):
        super().__init__()
        self.dim = dim
        self.tau_macro = tau_macro
        self.tau_delta = tau_delta
        # Flight Recorder Registry for deterministic point-in-time state scrubbing
        self.flight_recorder = {}

    def process_step(self, frame_metadata: dict, embedding_vector: torch.Tensor, current_timestep: int, running_state: torch.Tensor) -> torch.Tensor:
        """
        Executes point-in-time clock updates across timescale tiers.
        """
        is_iframe = (frame_metadata.get("type") == "I-FRAME")

        if is_iframe:
            # Macro Update Pathway: Structural keyframe anchor
            new_state = (self.tau_macro * running_state) + ((1.0 - self.tau_macro) * embedding_vector)
            # Register immutable snapshot
            self.flight_recorder[current_timestep] = new_state.clone().detach()
        else:
            # Local Delta Pathway: Fast transient tracking
            new_state = (self.tau_delta * running_state) + ((1.0 - self.tau_delta) * embedding_vector)

        return new_state

    def scrub_to_timestamp(self, target_timestep: int):
        """
        Deterministic point-in-time recovery command.
        Rolls back the neural state to the exact keyframe checkpoint.
        """
        if target_timestep in self.flight_recorder:
            return self.flight_recorder[target_timestep].clone(), f"Successfully scrubbed state back to step {target_timestep}."
        
        available_keys = sorted([k for k in self.flight_recorder.keys() if k < target_timestep])
        if available_keys:
            nearest = available_keys[-1]
            return self.flight_recorder[nearest].clone(), f"Rolled back to closest keyframe anchor at step {nearest}."
        
        return torch.zeros(1, self.dim), "Registry blank. Cleared engine state to zero."
