# Original list
data_list = [
    "1 W D5621 BOT 7 750.00 AFIYAH-2 5250.00 2 W D5621 DUP 7 750.00 AFIYAH-2 5250.00",
    "3 W D5621 TOP 7 750.00 AFIYAH-2 5250.00 4 W D5622 BOT 7 750.00 AFIYAH-2 5250.00",
    # Other entries...
]

# Design numbers to split the list on
design_numbers = ["D5621", "D5443"]

# Dictionary to store split lists
split_lists = {design_number: [] for design_number in design_numbers}
split_lists["other"] = []  # For entries without the specified design numbers

# Split the list based on design numbers
for entry in data_list:
    found_design_number = False
    for design_number in design_numbers:
        if design_number in entry:
            split_lists[design_number].append(entry)
            found_design_number = True
            break
    if not found_design_number:
        split_lists["other"].append(entry)

# Print the split lists
for key, value in split_lists.items():
    print(f"Entries with design number '{key}':")
    for entry in value:
        print(entry)
    print()
