import ast

class CodeTMOTokenizer:
    """
    Code TMOT Tokenizer: Parses source code AST into structural timeline units.
    - Major definitions (classes, functions, global assignments) -> I-Frames (Macro Keyframes)
    - Functional bodies and internal statements -> B-Frames (Local Deltas)
    """
    def __init__(self):
        self.special_tokens = {"[I_FRAME]": 0, "[B_FRAME]": 1, "[PAD]": 2}
        self.vocab = {**self.special_tokens}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize_source_file(self, source_code_string: str) -> list:
        try:
            parsed_ast = ast.parse(source_code_string)
        except SyntaxError:
            raise ValueError("Invalid Python syntax provided to Code TMOT Tokenizer.")

        structured_tokens = []
        for node in parsed_ast.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                structured_tokens.append({
                    "type": "I-FRAME",
                    "identifier": f"DEF_{node.name}",
                    "content": ast.unparse(node).split('\n')[0]
                })
                for sub_node in node.body:
                    structured_tokens.append({
                        "type": "B-FRAME",
                        "identifier": sub_node.__class__.__name__,
                        "content": ast.unparse(sub_node)
                    })
            elif isinstance(node, (ast.Assign, ast.Import, ast.ImportFrom)):
                structured_tokens.append({
                    "type": "I-FRAME",
                    "identifier": f"GLOBAL_{node.__class__.__name__}",
                    "content": ast.unparse(node)
                })
            else:
                structured_tokens.append({
                    "type": "B-FRAME",
                    "identifier": "EXPR_BLOCK",
                    "content": ast.unparse(node)
                })
        return structured_tokens
