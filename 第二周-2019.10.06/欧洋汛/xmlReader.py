#%%
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
tree = ET.parse("E:/CodePractice/第二周-2019.10.06/xml文件/xml/0.xml")
root = tree.getroot()

#遍历根的子节点
for child in root:
    print(child.tag,child.attrib)
#%%
#使用iter迭代器遍历所有的元素
for elem in tree.iter():
    print(elem.tag,elem.attrib)


#%%
#使用iter迭代器寻找tag是name的元素
for elem in tree.iter(tag='name'):
    print(elem.tag,elem.attrib)
    print(elem.text)#打印close tag中的文字


#%%
#使用iter迭代器寻找tag是name的元素
for elem in tree.iter(tag='name'):
    print(elem.tag,elem.attrib)