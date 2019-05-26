#Using colors to understand word vectors
import json
color_data = json.loads(open("xkcd.json").read())

#Convert hex code to tuple of integers
def hex_to_int(s):
	s = s.lstrip("#")
	return int(s[:2],16), int(s[2:4],16), int(s[4:6],16)

#Creating a mapping from RGB vectors to the names
colors = dict()
for item in color_data['colors']:
	colors[item["color"]]=hex_to_int(item["hex"])

print(colors['olive'])