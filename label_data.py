import os
from pathlib import Path
import logging
from lxml import etree
import cv2
import pandas as pd
import numpy as np

from grapher import ObjectTree, Graph

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def get_xy_min_max(coords):
    xs = []
    ys = []
    for xy in coords.split():
        x, y = xy.split(',')
        xs.append(float(x))
        ys.append(float(y))
    return min(xs), min(ys), max(xs), max(ys)


def compute_intersection(p1, p2):
    x1min, y1min, x1max, y1max = p1
    x2min, y2min, x2max, y2max = p2

    xminmax = max(x1min, x2min)
    xmaxmin = min(x1max, x2max)

    yminmax = max(y1min, y2min)
    ymaxmin = min(y1max, y2max)

    return max(0, xmaxmin - xminmax) * max(0, ymaxmin - yminmax)


def compute_area(xmin, ymin, xmax, ymax):
    return max(0, xmax-xmin) *max(0, ymax-ymin)


def load_entities(gt_file):
    namespaces = {None: "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
                  "ns2": "http://searchink.com/xml/frankstahlapi/1.0"}
    entities = []
    root = etree.parse(open(gt_file))
    for e in root.findall("//TextRegion",
                          namespaces=namespaces):
        property = e.find("Property", namespaces=namespaces)
        if property is not None and property.get("key") == "entity":
            label = property.get("value")
            coords = e.find("Coords", namespaces=namespaces).get("points")
            box = get_xy_min_max(coords)
            entities.append([box, label])
    return entities


def load_ocr_file(ocr_file):
    data = []
    namespaces = {None: "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    root = etree.parse(open(ocr_file))
    for e in root.findall("//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextRegion"):
        coords = e.find("{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Coords").get("points")
        words = [w.strip() for w in e.xpath(".//text()")]
        if "".join(words).strip():
            xmin, ymin, xmax, ymax = get_xy_min_max(coords)
            data.append([xmin, ymin, xmax, ymax, " ".join(words).strip()])
    return data


def label_ocr(gt_file, ocr_file, threshold_area=0.5, default_label='other'):
    entities = load_entities(gt_file)
    ocr_data = load_ocr_file(ocr_file)
    result = []
    for node in ocr_data:
        xmin, ymin, xmax, ymax, text = node
        box = xmin, ymin, xmax, ymax
        box_area = compute_area(*box)
        node_entity = default_label
        for entity in entities:
            entity_box, entity_label = entity
            intersection_area = compute_intersection(box, entity_box)
            if intersection_area/box_area > threshold_area:
                node_entity = entity_label
                break
        result.append([xmin, ymin, xmax, ymax, text, node_entity])
    return result


LABEL_ID = {'other':0,
            'invoice_info':1,
            'positions':2,
            'receiver':3,
            'supplier':4,
            'total':5}


def compute(file_id, dataset_directory, numpy_output_dir="./grapher_outputs/numpy"):
    gt_file = os.path.join(dataset_directory, file_id+"_gt.xml")
    ocr_file = os.path.join(dataset_directory, file_id+"_ocr.xml")
    img_path = os.path.join(dataset_directory, file_id+".tif")

    df = pd.DataFrame(label_ocr(gt_file, ocr_file),
                      columns=['xmin', 'ymin', 'xmax', 'ymax', 'Object', "label"])
    c = ObjectTree(label_column='label', count=file_id)

    img = cv2.imread(img_path, 0)

    c.read(df, img)

    graph_dict, text_list = c.connect(plot=True, export_df=True)

    graph = Graph()
    A, X = graph.make_graph_data(graph_dict, text_list)
    if not os.path.exists(numpy_output_dir):
        os.makedirs(numpy_output_dir)
    np.save(Path(numpy_output_dir) / f"{file_id}_A.npy", A)

    node_mat = np.zeros((50, 7))
    box_array = df[["xmin", "ymin", "xmax", "ymax"]].values[:50]
    node_mat[:box_array.shape[0], :4] = box_array
    node_mat[:X.shape[0], 4:] = X
    np.save(Path(numpy_output_dir) / f"{file_id}_X.npy", node_mat)

    Y = np.zeros(50)
    labels = df.label.apply(lambda l: LABEL_ID[l]).values[:50]
    Y[:labels.shape[0]] = labels
    np.save(Path(numpy_output_dir) / f"{file_id}_Y.npy", Y)
    #
    # print("A")
    # print(A)
    # print(100 * "-")
    # print("X")
    # print(X)
    # print(100 * "-")


def label_dataset(dataset_directory):
    logging.info(f"Compute labels for dataset in {dataset_directory}")
    for f in Path(dataset_directory).glob('*.tif'):
        file_identifier = f.stem
        logging.info(f"computing element {file_identifier}")
        try:
            compute(file_identifier, dataset_directory)
        except KeyError:
            logging.warning(f"KeyError for file {file_identifier}")
        except TypeError:
            logging.warning(f"TypeError for file {file_identifier}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    label_dataset(args.dataset)


if __name__ == "__main__":
    print("a")
    main()