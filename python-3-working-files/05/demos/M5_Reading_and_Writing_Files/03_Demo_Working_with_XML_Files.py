import xml.etree.ElementTree as ET
import xml.dom.minidom
import os


def parse_xml_et(fn):
    if not os.path.exists(fn):
        print(f"[SKIP] {fn} (not found)")
        return

    tree = ET.parse(fn)
    root = tree.getroot()
    print('Domains for: ' + root.attrib['name'])
    for child in root:
        print('\t' + child.attrib['name'], child.tag)


def pretty_print_and_write(tree, filename):
    rough_string = ET.tostring(tree.getroot(), 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # Remove extra blank lines (filter empty or whitespace-only lines)
    cleaned_xml = "\n".join([line for line in pretty_xml.split('\n') if line.strip()])

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(cleaned_xml)

    print(f"[SUCCESS] XML written to {filename}")


def add_xml_element_et(fn, el, attr, val):
    if not os.path.exists(fn):
        print(f"[SKIP] {fn} (not found)")
        return

    tree = ET.parse(fn)
    root = tree.getroot()

    # Check if element with the same attribute already exists
    exists = root.find(f"./{el}[@{attr}='{val}']")
    if exists is not None:
        print(f"[SKIP] <{el} {attr}='{val}'> already exists in {fn}")
        return

    # Add new element
    child = ET.Element(el)
    child.attrib[attr] = val
    root.append(child)

    pretty_print_and_write(tree, fn)
    print(f"[SUCCESS] Added <{el} {attr}='{val}'> to {fn}")


def change_xml_element_et(fn, el, attr, oldval, newval):
    if not os.path.exists(fn):
        print(f"[SKIP] {fn} (not found)")
        return

    tree = ET.parse(fn)
    root = tree.getroot()
    child = root.find(f"./{el}[@{attr}='{oldval}']")
    if child is not None:
        child.attrib[attr] = newval

    pretty_print_and_write(tree, fn)


def change_xml_element_et(fn, el, attr, oldval, newval):
    if not os.path.exists(fn):
        print(f"[SKIP] {fn} (not found)")
        return

    tree = ET.parse(fn)
    root = tree.getroot()
    child = root.find("./" + el + "[@" + attr + "='" + oldval + "']")
    child.attrib[attr] = newval
    tree.write(fn)
    print(f"[SUCCESS] Replaced <{el} {attr}='{oldval}'> with <{el} {attr}='{newval}'> in {fn}")


# parse_xml_et('../files_to_read/ef_author.xml')
# add_xml_element_et('../files_to_read/ef_author.xml', 'domain', 'name', 'Java')
change_xml_element_et('../files_to_read/ef_author.xml', 'domain', 'name', 'Java', 'TypeScript')