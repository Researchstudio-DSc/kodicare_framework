"""
a python script to execute the cord 19 normalizer for a document collection
"""

import argparse

from multiprocessing import Queue, Process
from code.preprocessing import cord19_normalizer
from code.utils import io_util


def worker(proc_num, input_files_queue, input_dir, output_dir):
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
                                                   io_util.join(output_dir, output_file))


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    input_pdf_papers_queue = get_input_pdf_papers_queue(input_dir)

    if not io_util.path_exits(output_dir):
        io_util.mkdir(output_dir)

    procs = [Process(target=worker, args=[i, input_pdf_papers_queue, input_dir, output_dir])
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

    args = parser.parse_args()
    main(args)
