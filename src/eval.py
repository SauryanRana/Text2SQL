import outlines

arithmetic_grammar = """
    ?start: expression

    ?expression: term (("+" | "-") term)*

    ?term: factor (("*" | "/") factor)*

    ?factor: NUMBER
           | "-" factor
           | "(" expression ")"

    %import common.NUMBER
"""

model = outlines.models.transformers("WizardLM/WizardMath-7B-V1.1")
generator = outlines.generate.cfg(model, arithmetic_grammar)
sequence = generator("Alice had 4 apples and Bob ate 2. Write an expression for Alice's apples:")

print(sequence)
