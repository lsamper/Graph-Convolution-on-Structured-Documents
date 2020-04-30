# Layout Analysis Groundtruth for the RVL-CDIP Dataset
[Lutz Goldmann](lutz@omnius.com) ([omni:us](http://www.omnius.com/))

## Purpose

This purpose of this dataset is to develop and evaluate layout analysis techniques. More specifically it is focused on the classification of indiviual words and the detection of semantic regions described by boxes.

## Contents

- readme.md: This file providing a short description of the dataset in markdown format.
- dataset.zip: An archive which contains the complete dataset
    - *.tif: Document images corresponding to the ones in the original RVL-CDIP dataset.
    - *_ocr.xml: PageXML file containing the text extracted from the images using OCR.
    - *_gt.xml: PageXML file containing the groundtruth for layout analysis.

## Use

Both extracted text and layout groundtruth are stored in [PageXML](https://www.primaresearch.org/tools/PAGELibraries) format which is a common format for annotating documents. The xml files can be viewed and edited using the [nw-page-editor](https://github.com/mauvilsa/nw-page-editor). For python the [pagexml](https://github.com/omni-us/pagexml) library can be used.

## Citation

If you are using this dataset for experiments within your publications please cite the related paper:

Pau Riba, Anjan Dutta, Lutz Goldmann, Alicia Fornes, Oriol Ramos, Josep LLados: "Table Detection in Invoice Documents by Graph Neural Networks", ICDAR, 2019.