import datetime
import os.path
import time

from .tee import Tee
from . import file_util
import Constants as cn

expr_log_dir_abspath = os.path.join(os.path.expanduser('~'), "Dropbox/SiameseLSTM/experiments_files/")
tmp_expr_folder_prefix = "/tmp/SiameseLSTM/"


def experiment_on_data_and_save_results(experiment_fun, run_num, expr_id=None):
    """
    runs the given experiment function with loaded configurations in Constants,
        saves the results as .csv and figure .png
    :param experiment_fun: the wanted experiment function
    :param expr_id: desired experiment id (if needed)
    :return: None
    """
    start_time = time.time()
    if expr_id is None:
        expr_id = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')  # used in all files to identify this experiment

    file_util.makedirs(tmp_expr_folder_prefix, exists_ok=True)
    cn.tmp_expr_foldpath = file_util.create_temp_folder(prefix=tmp_expr_folder_prefix)
    tmp_expr_foldpath = cn.tmp_expr_foldpath

    with Tee(os.path.join(tmp_expr_foldpath, 'output_log' + expr_id + '.txt')):
        print "run number: " + str(run_num)
        experiment_name = experiment_fun()
        print "total run time: " + str(time.time() - start_time)

    # experiment finished !
    experiment_dir_abspath = os.path.join(expr_log_dir_abspath, experiment_name, expr_id)
    print experiment_dir_abspath

    os.makedirs(experiment_dir_abspath)  # copy temp folder to the destination folder
    file_util.copy_folder_contents(src_dir=tmp_expr_foldpath, dst_dir=experiment_dir_abspath)
    file_util.delete_folder_with_content(tmp_expr_foldpath)
