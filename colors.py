import json

def hex_to_rgb(hex_value):
    # Remove the '#' if it's present in the hex string
    hex_value = hex_value.lstrip('#')
    
    # Convert the hex value to an RGB tuple
    r, g, b = tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))
    
    return r, g, b

def load_pantone_colors(json_file):
    pantone_colors = {}
    
    # Open and load the JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Loop through the names and corresponding hex values
    for name, hex_value in zip(data['names'], data['values']):
        r, g, b = hex_to_rgb(hex_value)
        pantone_colors[name] = (r, g, b)
    
    return pantone_colors

# Load the Pantone colors from the JSON file
pantone_colors = load_pantone_colors('pantone-colors.json')
