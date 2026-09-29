import unittest
import torch
from timemeshin.tokenizers.code_tmot import CodeTMOTokenizer
from timemeshin.layers.timeline import TimeMeshinTimelineEngine

class TestASTContextIsolation(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CodeTMOTokenizer()
        self.engine = TimeMeshinTimelineEngine(dim=4)
        self.sample_code_repo = """
def calculate_prime_numbers():
    secret_key_variable = 999
    return True

def render_user_profile():
    display_name = "Chandra"
    return False
"""

    def test_variable_context_isolation(self):
        timeline_tokens = self.tokenizer.tokenize_source_file(self.sample_code_repo)
        running_state = torch.zeros(1, 4)
        stored_timeline_states = {}

        for idx, token_meta in enumerate(timeline_tokens):
            mock_emb = torch.randn(1, 4)
            running_state = self.engine.process_step(token_meta, mock_emb, idx, running_state)
            stored_timeline_states[token_meta["identifier"]] = running_state.clone()

        secret_var_state = stored_timeline_states.get("Assign", None)
        second_func_state = stored_timeline_states.get("DEF_render_user_profile", None)

        self.assertIsNotNone(secret_var_state, "AST processing failed to extract inner assignment blocks.")
        self.assertIsNotNone(second_func_state, "AST processing missed functional boundary shifts.")

        state_difference_norm = torch.norm(second_func_state - secret_var_state).item()
        self.assertGreater(state_difference_norm, 0.001, "Context leak caught! B-Frame parameters bled into the separate I-Frame.")

if __name__ == "__main__":
    unittest.main()
