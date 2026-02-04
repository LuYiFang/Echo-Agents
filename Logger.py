class Logger:
    def __init__(self):
        self.lines = []

    def log(self, msg):
        print(msg)
        self.lines.append(msg)

    def save(self, filename="game_log.txt"):
        with open(filename, "w", encoding="utf-8") as f:
            for line in self.lines:
                f.write(line + "\n")
