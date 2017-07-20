import sys
from xml.dom import minidom

xmldoc = minidom.parse(sys.argv[1])
itemlist = xmldoc.getElementsByTagName('term')
for s in itemlist:
    sys.stdout.write(s.attributes['lemma'].value)
    sys.stdout.write(" ")
