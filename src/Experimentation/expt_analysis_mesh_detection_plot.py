import numpy as np
import pandas as pd
import commons.my_file_path_manager as myfs

from typing import Tuple
def get_cgs_camera_positions()-> Tuple(np.ndarray, np.array):
    """一周する"""
    t = np.sin(np.radians(np.arange(10, 90, 10)))
    phi = np.radians(np.arange(0, 360, 10))
    return phi, t

def plot_mesh(mesh, key, plot_title, color, *args, **kwargs):
    import matplotlib as mtpl
    mtpl.use('pdf')
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pylab as plt
    plt.switch_backend('pdf')
    import seaborn as sns
    import os

    print('plotting {0} - {1}'.format(key, plot_title))
    path = myfs.setup_output_directory(kwargs.get('expt_dir', ''), kwargs.get('dir_path_full', ''), *args)
    plt.figure(figsize=(16,12))
    sns.set("paper", "whitegrid", "dark", font_scale=1.5)
    fmt = 'f' if mesh.dtype == np.float64 else 'd'
    h = sns.heatmap(mesh, annot=True, fmt=fmt, cmap=color)
    h.set(xlabel="Mesh map X")
    h.set(ylabel="Mesh map Y")
    h.set(title=plot_title + " heat map - " + key)
    h_fig = h.get_figure()
    pp = PdfPages(os.path.join(path, plot_title + key + kwargs.get('name', '') + '.pdf'))
    h_fig.savefig(pp, format='pdf')
    plt.close()
    pp.close()

def test_data():
    #expt_dir = "/home/taopipi_g/prjects/data/outputs/expt_mesh_detection_performance_spltASIFT"
    expt_dir = "/home/taopipi_g/projects/data/outputs/expt_mesh_detection_performance_spltASIFT"
    fn = "pl_qrmarker.npz"
    return expt_dir, fn

def analysis_each_mesh_keypoints(template_npz):
    #[phi, t]のリスト
    params = [[]]
    for i, key in enumerate(template_npz.keys()):
        a = template_npz[key]
        #a (8 * 8 * 3)の各メッシュ内の特徴点数の行列
        #a[:, :, 0] =>メッシュ内のHomographyで求めた誤差範囲内のキーポイント数
        #a[:, :, 1] =>メッシュ内のマッチングポイント数全て
        #メッシュ毎のリファイン率
        flag = a[:, :, 1]==0
        s = a[:, :, 1]
        s[flag] = 1
        ratio = a[:, :, 0] / s
        #メッシュ毎の平均特徴点数
        means = list(np.mean(s) for s in [a[:, :, 0], a[:, :, 1], ratio])
        #マッチングできた数
        sums = list(np.sum(s) for s in [a[:, :, 0], a[:, :, 1]])
        #メッシュの数
        mesh_num = np.count_nonzero(a[:, :, 1])

        #params= [phi, t, sum_refine, sum_match, mean_refine, mean_match, mean_ratio, mesh_num]
        params[i].extend(sums)
        params[i].extend(means)
        params[i].append(mesh_num)
    return np.array(params)

def plot_each_cam_position(expt_dir, fn):
    import os
    name, ext = os.path.splitext(fn)
    template_npz = np.load(os.path.join(expt_dir, fn))

    for i, key in enumerate(template_npz.keys()):
        a = template_npz[key]
        #a (8 * 8 * 3)の各メッシュ内の特徴点数の行列
        #a[:, :, 0] =>メッシュ内のHomographyで求めた誤差範囲内のキーポイント数
        #a[:, :, 1] =>メッシュ内のマッチングポイント数全て
        plot_mesh(a[:, :, 0], key, "Refine points", 'Blues', name, expt_dir=expt_dir, name=name)

        plot_mesh(a[:, :, 1], key, "Matched points", 'Reds', name, expt_dir=expt_dir, name=name)
        flag = a[:, :, 1]==0
        s = a[:, :, 1]
        s[flag] = 1
        ratio = a[:, :, 0] / s
        plot_mesh(ratio, key, "Ratio of refine and matched points \n", 'Purples', name, expt_dir=expt_dir, name=name)

def plot_passage(df, plot_title, path, key, *kwargs):
    import os
    import matplotlib as mtpl
    mtpl.use('pdf')
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pylab as plt
    plt.switch_backend('pdf')
    import seaborn as sns
    print('plotting {}'.format(plot_title))
    phi, t = get_cgs_camera_positions()
    x, y = np.meshgrid(phi, t)
    columns = ["longitude", "latitude", "sum_refine", "sum_match", "mean_refine", "mean_match", "mean_ratio", "mesh_num"]
    def plot_data(title, data):
        plt.figure(figsize=(16,12))
        sns.set("paper", "whitegrid", "dark", font_scale=1.5)
        pp = PdfPages(os.path.join(path, plot_title + key + kwargs.get('name', '') + '.pdf'))


def plot_detection_passage(expt_dir, fn):
    import os
    name, ext = os.path.splitext(fn)

    phi_ticklabels = [r"${}^\circle$".format(phi) for phi in np.arange(0, 360, 10)]
    t_rgrids = [r"${}^\circle$".format(t) for t in np.arange(90, 10, 10)]
    columns = ["longitude", "latitude", "sum_refine", "sum_match", "mean_refine", "mean_match", "mean_ratio", "mesh_num"]

    template_npz = np.load(os.path.join(expt_dir, fn))
    params = analysis_each_mesh_keypoints(template_npz)
    df = pd.DataFrame({'longitude': params[:, 0],
                       'latitude': params[:, 1],
                       'sum_refine': params[:, 2],
                       'sum_match': params[:, 3],
                       'mean_refine': params[:, 4],
                       'mean_match': params[:, 5],
                       'mean_ratio': params[:, 6],
                       'mesh_num': params[:, 7]}, columns=columns)
    df.to_csv(os.path.join(expt_dir, 'detection_analysis_via_cam_position' + name + '.csv'))


if __name__ == '__main__':
    expt_dir, fn = test_data()
    plot_detection_passage(expt_dir, fn)
