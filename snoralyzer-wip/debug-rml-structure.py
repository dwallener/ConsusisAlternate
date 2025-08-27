import xml.etree.ElementTree as ET

def debug_parse(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    for elem in root.iter():
        print("Tag:", elem.tag, "Attributes:", elem.attrib)

# Replace "annotations.rml" with your actual file path.
debug_parse("annotations.rml")

