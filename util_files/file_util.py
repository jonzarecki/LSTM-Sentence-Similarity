from distutils.dir_util import copy_tree
import os
import random
import tempfile
import shutil
import errno


def shuffle_rows(in_filename, out_filename):
    """
    shuffles the rows of $in_filename and puts the output in $out_filename
    :param in_filename: file name of the input file
    :type in_filename: str
    :param out_filename: file name of the output file
    :type out_filename: str
    :return: None
    """
    with open(in_filename, 'r') as source:
        data = [(random.random(), line) for line in source]
    data.sort()
    with open(out_filename, 'w') as target:
        for _, line in data:
            target.write(line)


def merge_similar_rows(in_filename, out_filename):
    """
    merges exact rows in $in_filename and puts the output file in $out_filename
    :param in_filename: file name of the input file
    :type in_filename: str
    :param out_filename: file name of the output file
    :type out_filename: str
    :return: None
    """
    lines_seen = set()  # holds lines already seen

    with open(in_filename, 'r') as in_file, open(out_filename, 'w') as out_file:
        for line in in_file:
            if line not in lines_seen:  # not a duplicate
                out_file.write(line)
                lines_seen.add(line)


def create_temp_folder(prefix=None):
    """
    creates a new temporary directory and returns it's path
    :param prefix: the prefix for the temp folder 
    :return: full path of the new directory
    """
    if prefix is not None:
        return tempfile.mkdtemp(prefix=prefix)
    else:
        return tempfile.mkdtemp()


def copy_folder_contents(src_dir, dst_dir):
    # type: (str, str) -> None
    """
    copies all files from one directory to another
    :param src_dir: path to src directory
    :param dst_dir: path to dst directory
    :return: None
    """
    assert src_dir != dst_dir, "src and dst directories shouldn't be the same, check code"
    copy_tree(src_dir, dst_dir)
    # shutil.copytree(src_dir, dst_dir)  # /copies all files :D


def delete_folder_with_content(folder_path):
    # type: (str) -> None
    """
    Deletes a folder recursively with all it's contents (no warnings)
    DANGEROUS USE WITH CARE
    :param folder_path: The absolute path to folder
    :return: None
    """
    shutil.rmtree(folder_path)


def makedirs(folder_path, exists_ok=True):
    # type: (str, bool) -> None
    """
    Create all folders in the path, doesn't fail of exists_ok is True
    :param folder_path: the absolute path to the folder
    :param exists_ok: states if we should fail when the folder already exists
    :return: $folder_path
    """
    if exists_ok:
        try:
            os.makedirs(folder_path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
    else:
        os.makedirs(folder_path)
