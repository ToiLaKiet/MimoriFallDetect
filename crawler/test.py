import json
from pathlib import Path

j = Path(__file__).with_name("har_up_links.json").read_text()
data = json.loads(j)
print(len(data))
