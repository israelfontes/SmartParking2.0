import json

class Conf:
    def __init__(self, file):
        with open(file, 'r') as json_file:
            data = json.load(json_file)
        self.data = data
    
    def load(self):
        return self.data