import xml.etree.cElementTree as ET
import os

box_name = []


def xmlxixni(xmlpath):
    global box_name
    xmllist = os.listdir(xmlpath)
    for i, xml in enumerate(xmllist):
        i += 1
        xml_path = xmlpath + xml
        tree = ET.parse(xml_path)
        for obj in tree.findall('object'):
            name = obj.find('name').text
            box_name.extend([name])
    print(box_name)
    print(len(box_name))
    box_name = list(set(box_name))
    print(box_name)
    print(len(box_name))


xmlpath3 = "D:/python_program/data/"
xmlxixni(xmlpath3)
