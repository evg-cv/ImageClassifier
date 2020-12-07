import os
import tarfile
import sys

from six.moves import urllib


def download_and_extract_model(data_url, save_dir):
    # Download and extract imgnet tar file.

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = data_url.split('/')[-1]
    filepath = os.path.join(save_dir, filename)

    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):

            sys.stdout.write("\r>> Downloading %s %.1f%%" % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        stat_info = os.stat(filepath)
        sys.stdout.write("\nSuccessfully downloaded {} {} bytes.\n".format(filename, stat_info.st_size))

    tarfile.open(filepath, 'r:gz').extractall(save_dir)
