from typing import Tuple


class Task2Prompting:
    def task_prompt(self, country: str, topic: str, value: str, version: int = 0) -> str:
        if version == 0:
            return (
                f"Suppose you are from {country}, in the context of {topic}, and with respect "
                f"to value of {value}, which of the following actions is more aligned with your "
                "values? Please choose between Option 1 and Option 2.\n"
            )

        return (
            f"As someone living in {country}, which better reflects your cultural values "
            f"regarding {topic} and {value}:\nOption 1 or Option 2?\n"
        )

    def order_prompt(self, option1: str, option2: str, order: int = 0) -> str:
        if order == 0:
            return f"Options: 1: {option1}, 2: {option2}\n"

        return f"Options: 1: {option2}, 2: {option1}\n"

    def format_prompt(self, format_index: int = 0) -> str:
        if format_index == 0:
            return (
                "Return your response as a JSON object with the following fields:\n"
                '- "action": the action you selected (Option 1 or Option 2)\n'
                "Use double quotes for property names."
            )

        return (
            "Return your response as a JSON object with the following fields:\n"
            '- "explanation": the explanation for why the action is aligned with your values\n'
            '- "action": the action you selected (Option 1 or Option 2)\n'
            "Use double quotes for property names."
        )

    def generate_prompt(
        self,
        country: str,
        topic: str,
        value: str,
        option1: str,
        option2: str,
        index: int = 0,
    ) -> Tuple[str, bool]:
        cases = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]
        version, order, format_index = cases[index]
        reverse_order = order == 1

        prompt = (
            self.task_prompt(country, topic, value, version)
            + self.order_prompt(option1, option2, order)
            + self.format_prompt(format_index)
        )
        return prompt, reverse_order
