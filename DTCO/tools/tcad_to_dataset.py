import os
import csv


def Read_plt_IV(filename):

    name_tmp = []
    data_tmp = []
    variable_name = []
    data_result = []

    with open(filename, "r") as f:

        line = f.readline()
        while line:
            if line.split():

                if line.strip().split()[0] == "datasets":
                    while line:
                        line = f.readline()
                        for value in line.split():
                            if value.strip('"') != "]":
                                name_tmp.append(value.strip('"'))
                        if line.strip().split()[-1] == "]":
                            break

                if line.strip().split()[0] == "Data":
                    while line:
                        line = f.readline()
                        for value in line.split():
                            data_tmp.append(value.strip('"'))
                        if line.strip().split()[-1] == "}":
                            break
            line = f.readline()

    for i in range(len(name_tmp)):
        if i == 0:
            variable_name.append(name_tmp[i])
        if (i != 0) & (i % 2 == 0) :
            variable_name.append(name_tmp[i-1] + "_" + name_tmp[i])

    time_tmp = []
    for i in range(len(data_tmp[:-1])):
        time_tmp.append(float(data_tmp[i]))
        if len(time_tmp) == len(variable_name):
            data_result.append(time_tmp)
            time_tmp = []

    data_result.insert(0, variable_name)

    # ===== 在这里把 2D list 转换成 dict =====
    header = data_result[0]
    rows = data_result[1:]
    data_dict = {col: [row[i] for row in rows] for i, col in enumerate(header)}

    return data_dict

def Read_plt_CV(filename):

    name_tmp = []
    data_tmp = []
    variable_name = []
    data_result = []

    with open(filename, "r") as f:

        line = f.readline()
        while line:
            if line.split():

                if line.strip().split()[0] == "datasets":
                    while line:
                        line = f.readline()
                        for value in line.split():
                            if value.strip('"') != "]":
                                name_tmp.append(value.strip('"'))
                        if line.strip().split()[-1] == "]":
                            break

                if line.strip().split()[0] == "Data":
                    while line:
                        line = f.readline()
                        for value in line.split():
                            data_tmp.append(value.strip('"'))
                        if line.strip().split()[-1] == "}":
                            break
            line = f.readline()

    for i in range(len(name_tmp)):
        # CV的PLT格式可以直接提取所有变量名，不需要用下划线组合
        variable_name.append(name_tmp[i])
        # if i == 0:
        #     variable_name.append(name_tmp[i])
        # if (i != 0) & (i % 2 == 0) :
        #     variable_name.append(name_tmp[i-1] + "_" + name_tmp[i])

    time_tmp = []
    for i in range(len(data_tmp[:-1])):
        time_tmp.append(float(data_tmp[i]))
        if len(time_tmp) == len(variable_name):
            data_result.append(time_tmp)
            time_tmp = []

    data_result.insert(0, variable_name)

    # ===== 在这里把 2D list 转换成 dict =====
    header = data_result[0]
    rows = data_result[1:]
    data_dict = {col: [row[i] for row in rows] for i, col in enumerate(header)}

    return data_dict


def _build_iv_rows(data_dict, Lg, nfin, ToxHK):
    """根据 IV 字典数据构建 CSV 行"""
    try:
        vg_list = data_dict["gate_OuterVoltage"]
        vd_list = data_dict["drain_OuterVoltage"]
        vs_list = data_dict["source_OuterVoltage"]
        vb_list = data_dict["substrate_OuterVoltage"]
        id_list = data_dict["drain_TotalCurrent"]
    except KeyError as e:
        raise KeyError(f"IV 数据中缺少必要字段: {e}. 可用字段: {list(data_dict.keys())}")

    n = len(vg_list)
    rows = []
    for i in range(n):
        rows.append(
            [
                Lg,
                nfin,
                ToxHK,  # 作为 eot 输出
                vg_list[i],
                vd_list[i],
                vs_list[i],
                vb_list[i],
                id_list[i],
            ]
        )
    return rows


def _build_cv_rows(data_dict, Lg, nfin, ToxHK):
    """根据 CV 字典数据构建 CSV 行"""
    try:
        vg_list = data_dict["v(g)"]
        vd_list = data_dict["v(d)"]
        vs_list = data_dict["v(s)"]
        vb_list = data_dict["v(b)"]
        cgg_list = data_dict["c(g,g)"]
        cgs_list = data_dict["c(g,s)"]
        cgd_list = data_dict["c(g,d)"]
        cbg_list = data_dict["c(b,g)"]
    except KeyError as e:
        raise KeyError(f"CV 数据中缺少必要字段: {e}. 可用字段: {list(data_dict.keys())}")

    n = len(vg_list)
    rows = []
    for i in range(n):
        rows.append(
            [
                Lg,
                nfin,
                ToxHK,  # 作为 eot 输出
                vg_list[i],
                vd_list[i],
                vs_list[i],
                vb_list[i],
                cgg_list[i],
                cgs_list[i],
                cgd_list[i],
                cbg_list[i],
            ]
        )
    return rows


def _write_csv(path, header, rows):
    """写 CSV 文件"""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def build_nn_dataset(
    plt_n_iv_list,
    plt_p_iv_list,
    plt_n_cv_list,
    plt_p_cv_list,
    filelist=None,
    Lg=0.018,
    nfin=2,
    ToxHK=1.13e-9,
):
    """
    从 n / p 器件的 I-V、C-V 仿真结果 plt 文件列表中生成四个 CSV 文件:
    - n_iv.csv, p_iv.csv, n_cv.csv, p_cv.csv (文件名可通过 filelist 覆盖)

    输入
    ----
    plt_n_iv_list : list[str] 或 str
        n 型器件的 I-V 结果 plt 文件列表 (Id-Vg / Id-Vd 等)。
        若传入单个 str，将自动视作长度为 1 的列表。
    plt_p_iv_list : list[str] 或 str
        p 型器件的 I-V 结果 plt 文件列表。
    plt_n_cv_list : list[str] 或 str
        n 型器件的 C-V 结果 plt 文件列表。
    plt_p_cv_list : list[str] 或 str
        p 型器件的 C-V 结果 plt 文件列表。
    filelist : list[str], optional
        长度为 4 的列表，依次为 [n_iv, p_iv, n_cv, p_cv] 的 CSV 文件名。
        若为 None，则默认为 ["n_iv.csv", "p_iv.csv", "n_cv.csv", "p_cv.csv"]。
    Lg, nfin, ToxHK :
        器件几何/工艺参数，其中 ToxHK 在 CSV 中作为 eot 输出。

    返回
    ----
    n_iv_path, p_iv_path, n_cv_path, p_cv_path : str
        四个 CSV 文件的绝对路径。
    """
    # 统一将单个字符串转换为列表，方便后续处理多个 plt
    if isinstance(plt_n_iv_list, str):
        plt_n_iv_list = [plt_n_iv_list]
    if isinstance(plt_p_iv_list, str):
        plt_p_iv_list = [plt_p_iv_list]
    if isinstance(plt_n_cv_list, str):
        plt_n_cv_list = [plt_n_cv_list]
    if isinstance(plt_p_cv_list, str):
        plt_p_cv_list = [plt_p_cv_list]

    if filelist is None:
        filelist = ["n_iv.csv", "p_iv.csv", "n_cv.csv", "p_cv.csv"]
    if len(filelist) != 4:
        raise ValueError("filelist 必须是长度为 4 的列表: [n_iv, p_iv, n_cv, p_cv]")

    n_iv_path, p_iv_path, n_cv_path, p_cv_path = [
        os.path.abspath(fname) for fname in filelist
    ]

    # ===== 构建 IV / CV 数据行 =====
    iv_header = ["l", "nfin", "eot", "vg", "vd", "vs", "vb", "id"]
    cv_header = ["l", "nfin", "eot", "vg", "vd", "vs", "vb", "cgg", "cgs", "cgd", "cbg"]

    # 对每一类器件 / 测试，吃 plt 的 file list，累加所有点到同一个 CSV 中
    n_iv_rows = []
    for plt_path in plt_n_iv_list:
        dict_n_iv = Read_plt_IV(plt_path)
        n_iv_rows.extend(_build_iv_rows(dict_n_iv, Lg, nfin, ToxHK))

    p_iv_rows = []
    for plt_path in plt_p_iv_list:
        dict_p_iv = Read_plt_IV(plt_path)
        p_iv_rows.extend(_build_iv_rows(dict_p_iv, Lg, nfin, ToxHK))

    n_cv_rows = []
    for plt_path in plt_n_cv_list:
        dict_n_cv = Read_plt_CV(plt_path)
        n_cv_rows.extend(_build_cv_rows(dict_n_cv, Lg, nfin, ToxHK))

    p_cv_rows = []
    for plt_path in plt_p_cv_list:
        dict_p_cv = Read_plt_CV(plt_path)
        p_cv_rows.extend(_build_cv_rows(dict_p_cv, Lg, nfin, ToxHK))

    # ===== 写出 CSV =====
    _write_csv(n_iv_path, iv_header, n_iv_rows)
    _write_csv(p_iv_path, iv_header, p_iv_rows)
    _write_csv(n_cv_path, cv_header, n_cv_rows)
    _write_csv(p_cv_path, cv_header, p_cv_rows)

    return n_iv_path, p_iv_path, n_cv_path, p_cv_path