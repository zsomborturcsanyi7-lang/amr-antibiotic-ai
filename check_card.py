import json

with open('data/raw/card.json') as f:
    card = json.load(f)

# Check for Drug Class categories
drug_class_entries = 0
for key, entry in list(card.items())[:20]:
    cats = entry.get('ARO_category', {})
    for cat_id, cat_info in cats.items():
        if isinstance(cat_info, dict):
            class_name = cat_info.get('category_aro_class_name', '')
            if 'Drug Class' in class_name or 'drug' in class_name.lower():
                print(f'{entry.get("ARO_name", "?")}: {cat_info.get("category_aro_name", "?")} [{class_name}]')
                drug_class_entries += 1

print(f'\nTotal drug class entries in first 20: {drug_class_entries}')

# Also check aro_categories files
print('\n--- aro_categories.tsv sample ---')
with open('data/raw/aro_categories.tsv') as f:
    lines = f.readlines()
    print(f'Total lines: {len(lines)}')
    for line in lines[:5]:
        print(line.strip())
