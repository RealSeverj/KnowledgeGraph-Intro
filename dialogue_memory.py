class DialogueMemory:
    def __init__(self, max_length):
        self.history = []
        self.max_length = max_length

    def add_entry(self, user_input, ai_response):
        self.history.append({"user": user_input, "ai": ai_response})
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def get_formatted_history(self):
        return "\n".join(
            [f"User: {entry['user']}\nAI: {entry['ai']}"
             for entry in self.history]
        )
