import os

def print_glassbox_ui_dashboard(step_idx: int, token: str, frame_type: str, grand_idx: int, med_idx: int, tiny_idx: int, confidence: float = 0.95):
    """
    Renders an on-the-fly terminal dashboard visualization of the Glassbox state trajectory.
    """
    print("==========================================================================")
    print("  📡 TIMEMESHIN-GLASSBOX ENGINE: LIVE INFERENCE STATE TRACE INSPECTOR     ")
    print("==========================================================================")
    print(f" TIMESTEP CURRENT: [{step_idx}] | ACTIVE SYMBOL: '{token}'")
    print(f" TIMELINE TIMESCALE MATRIX TAG: [{frame_type}]")
    print("--------------------------------------------------------------------------")
    print(" 🧮 LAYERS 1 & 2: DISCRETE ARCHITECTURAL COORDINATES")
    grand_bar = "■" * (grand_idx + 1) + "□" * max(0, 7 - grand_idx)
    med_bar = "■" * (med_idx + 1) + "□" * max(0, 7 - med_idx)
    tiny_bar = "■" * (tiny_idx + 1) + "□" * max(0, 7 - tiny_idx)
    print(f" ├─ Grand Bead Category Index [{grand_idx}]: [{grand_bar}]")
    print(f" ├─ Medium Bead Context Index [{med_idx}]: [{med_bar}]")
    print(f" └─ Tiny Bead Grammatical Index [{tiny_idx}]: [{tiny_bar}]")
    print("--------------------------------------------------------------------------")
    print(" 🔮 LAYER 4: STATE SPACE OUIJA PLANCHETTE LOCATION TRACE")
    pointer_map = ["."] * 15
    active_pointer_slot = (grand_idx + med_idx + tiny_idx) % 15
    pointer_map[active_pointer_slot] = "🎯"
    print(f" └─ Global Spatial Coordinate Track: [{''.join(pointer_map)}]")
    print("--------------------------------------------------------------------------")
    print(" 📝 LAYER 6: DETERMINISTIC PROBABILITY DISTRIBUTION TARGET WINNER")
    print(f" └─ Top Candidate Allocation Match: Confidence Scale -> {confidence * 100:.2f}%")
    print("==========================================================================")
