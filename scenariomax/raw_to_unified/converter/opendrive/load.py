import os
import glob

from lxml import etree
from scenariomax.raw_to_unified.converter.opendrive.opendriveparser import parse_opendrive
from scenariomax.raw_to_unified.converter.opendrive.opendriveparser.elements.openDrive import OpenDrive, Header

def load_xodr_and_parse(file) -> OpenDrive:
    assert os.path.exists(file), FileNotFoundError(file)
    _, ne = os.path.split(file)
    name, _ = os.path.splitext(ne)
    with open(file, 'r') as fh:
        parser = etree.XMLParser()
        root_node = etree.parse(fh, parser).getroot()
        road_network: OpenDrive = parse_opendrive(root_node)
    # We add the name into the header manually
    named_header = Header()
    named_header._name = name
    road_network._header = named_header
    return road_network


def get_opendrive_maps(data_root, map_names=None) -> list[OpenDrive]:
    xodr_files = [f for f in glob.glob(os.path.join(data_root, "*.xodr"))]
    if map_names != None:
        xodr_files = [file for file in xodr_files if file in map_names]
    return [load_xodr_and_parse(file=xodr_file) for xodr_file in xodr_files]