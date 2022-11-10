"""
a python script to execute the cord 19 normalizer for a document collection
"""

import argparse
import csv
from multiprocessing import Queue, Process

from code.preprocessing import cord19_normalizer
from code.utils import io_util


def worker(proc_num, input_files_queue, input_dir, output_dir, paper_id__cord_uid__map):
    while True:
        if input_files_queue.empty():
            break
        input_file = input_files_queue.get()
        print(proc_num, "Input file.. ", input_file)
        output_file = str(input_file[:-5]) + '_normalized.json'
        if io_util.path_exits(io_util.join(output_dir, output_file)):
            continue
        cord19_normalizer_inst = cord19_normalizer.Cord19Normalizer()
        cord19_normalizer_inst.normalize_input_doc(io_util.join(input_dir, input_file),
                                                   io_util.join(output_dir, output_file),
                                                   paper_id__cord_uid__map)


def construct_paper_id__cord_uid__map(metadata_file):
    paper_id__cord_uid__map = {}
    with open(metadata_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            paper_id__cord_uid__map[row['sha']] = row['cord_uid']
    return paper_id__cord_uid__map


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    metadata_path = args.metadata_path

    input_pdf_papers_queue = get_input_pdf_papers_queue(input_dir)
    paper_id__cord_uid__map = construct_paper_id__cord_uid__map(metadata_path)

    if not io_util.path_exits(output_dir):
        io_util.mkdir(output_dir)

    procs = [Process(target=worker, args=[i, input_pdf_papers_queue, input_dir, output_dir, paper_id__cord_uid__map])
             for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


def get_input_pdf_papers_queue(input_dir):
    pdf_files = [file for file in io_util.list_files_in_dir(input_dir) if file.endswith('.json')]
    input_pdf_papers_queue = Queue()
    for input_file in pdf_files:
        if input_file.startswith('.'):
            continue
        input_pdf_papers_queue.put(input_file)
    return input_pdf_papers_queue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given a directory of json cord19 articles, normalize of the articles'
    )

    parser.add_argument('input_dir', help='Input directory of json source from cord19 ')
    parser.add_argument('output_dir', help='The output directory of the normalized data')
    parser.add_argument('metadata_path', help='The metadata csv file of the cord 19 collection')

    args = parser.parse_args()
    main(args)
