import os
import sys
import tensorflow as tf
from glob import glob


def aggregate_event_files(input_dir, output_dir='merged_logs'):
    # Create a new summary writer
    sw = tf.summary.create_file_writer(input_dir)

    # Find all event files in input_dir and its subdirectories
    event_files = glob(os.path.join(
        input_dir, '**/events.out.tfevents*',), recursive=True)

    # sort the event files by name
    event_files.sort()

    unwanted_tags = ['Image', 'Predicted_label', 'Groundtruth_label']

    for ef in event_files:
        print('Loading event file:', ef)
        for e in tf.compat.v1.train.summary_iterator(ef):
            for v in e.summary.value:
                if v.tag not in unwanted_tags:
                    # print('Writing summary:', v, e.step)
                    with sw.as_default():
                        tf.summary.scalar(v.tag, v.simple_value, step=e.step)
    sw.close()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python aggregate_event_files.py <input_dir>')
        exit(1)

    input_dir = sys.argv[1]
    aggregate_event_files(input_dir)
