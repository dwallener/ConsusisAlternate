import xml.etree.ElementTree as ET

def print_xml_tree(element, indent=0):
    """ Recursively prints the XML tree structure with indentation. """
    prefix = " " * (indent * 4)  # Indentation for readability
    tag_name = element.tag.split("}")[-1]  # Remove namespace if present
    attributes = element.attrib  # Get attributes
    text = element.text.strip() if element.text and element.text.strip() else None

    # Print the tag, attributes, and text
    print(f"{prefix}<{tag_name}>", end="")
    if attributes:
        print(f"  [Attributes: {attributes}]", end="")
    if text:
        print(f"  [Text: {text}]", end="")
    print()

    # Recursively process child elements
    for child in element:
        print_xml_tree(child, indent + 1)

def display_rml_structure(rml_file):
    """ Parses an .rml file and prints its full XML structure. """
    tree = ET.parse(rml_file)
    root = tree.getroot()
    
    print(f"\nRoot Element: <{root.tag.split('}')[-1]}>\n")
    print_xml_tree(root)

# Example usage
rml_file = "S001R01.rml"  # Replace with your actual .rml filename
display_rml_structure(rml_file)

