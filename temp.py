import json

# Read the data from the file
with open('temp.json', 'r') as file:
    input_data = json.load(file)

# Create a set to store unique JSON strings
unique_json_strings = set()

# Remove duplicates and store unique JSON strings in a new list
output_data = []
for item in input_data:
    # Convert the dictionary to a JSON string
    json_string = json.dumps(item, sort_keys=True)

    # Check if the JSON string is already in the set
    if json_string not in unique_json_strings:
        unique_json_strings.add(json_string)
        output_data.append(item)

# Write the updated data back to the file
with open('temp.json', 'w') as file:
    json.dump(output_data, file, indent=2)

print("Duplicates removed and file updated successfully.")
