import xml.etree.cElementTree as ET
tree = ET.parse('Y:\project\xml_doc\0.xml')
root = tree.getroot()
tag = root.tag
attrib = root.attrib
for child in root:
    print(child.tag, child.attrib)