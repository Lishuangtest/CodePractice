# import xml.dom.minidom
# domtree=xml.dom.minidom.parse("0.xml")
# collection=domtree.documentElement
# objects=collection.getElementByTagName("object")
# for i,object in enumerate(objects):
#     if object.hasAttrbute("name"):
#         print("number:%d,Name:%s"%(i,object.getAttribute("name")))
import xml.etree.cElementTree as ET
import os
name=[]
def xmlxml(xmlpath):
    global name
    xmlist=os.listdir(xmlpath)
    for i,xml in enumerate(xmlist):
        xml_path=xmlpath+xml
        tree=ET.parse(xml_path)
        for obj in tree:
            if obj.findall('object'):
                name1=obj.find(name).text
                name.extend([name1])
    print(name)
    print(len(name))
xmlpath="D:\pycharm\CodePractice1\second\xml"
xmlxml(xmlpath)

