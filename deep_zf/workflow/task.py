from typing import Callable, List, Dict, Union

import os
from pathlib import Path
import traceback


def dict_contradict(d1: dict, d2: dict):
    for k in d1:
        if k in d2 and d2[k] != d1[k]:
            return True
    return False


class ParamScheme:
    def __init__(self, param_type_default_dict: dict):
        self.param_type_default_dict = param_type_default_dict

    def fill_param(self, params: dict):
        param_dict = dict()
        for name in self.param_type_default_dict:
            param_dict[name] = self.param_type_default_dict[name][1]
        param_dict.update(params)
        return param_dict

    def get_param_string(self, params: dict, partial: bool = True):
        """
        :param params: a dict containing param name and value
        :param partial: whether the param string contain all possible parameters (which are defined in the default dict)
        :return:
        """
        for p in params:
            if p not in self.param_type_default_dict:
                raise AssertionError(f"Invalid argument {p}")

        s = ""
        for name in self.param_type_default_dict:
            if s != "":
                s += "_"

            name_added = False
            default_val = self.param_type_default_dict[name][1]
            val = params.get(name) or default_val
            if (not partial) or (name in params):
                s += f"{name}-{val}"
                name_added = True

            if not name_added:
                s = s[:-1]

        return s

    def get_param_dict(self, param_string: str, partial: bool = True):
        param_dict = dict()
        if not partial:
            for name in self.param_type_default_dict:
                param_dict[name] = self.param_type_default_dict[name][1]

        name_val_pairs = param_string.split("_")
        for pair in name_val_pairs:
            name, val = pair.split("-")
            if val == 'None':
                val = None
            else:
                val = self.param_type_default_dict[name][0](val)
            param_dict[name] = val

        return param_dict


class TaskManager:
    def __init__(self, func: Callable, param_scheme: Union[Dict, ParamScheme], folder_params: List[List[str]] = None,
                 ignored_params_for_path: List[str] = None):
        """

        :param func: take parameter dict as kwargs, with `file_name` parameter
        :param param_scheme:
        :param folder_params:
        """
        self.func = func
        if not isinstance(param_scheme, ParamScheme):
            param_scheme = ParamScheme(param_scheme)
        self.param_scheme = param_scheme
        self.folder_params = folder_params or []
        self.ignored_params_for_path = ignored_params_for_path or []

        if len(set(sum(self.folder_params, start=[])) & set(self.ignored_params_for_path)) != 0:
            raise ValueError("folder_params shall not contain ignored_prams_for_path")

    def run_task(self, params):
        print("Start task: ")
        for k in params:
            print(f"{k:40} - {params[k]}")

        complete_params = self.param_scheme.fill_param(params)

        original_dir = os.getcwd()
        for folder_param in self.folder_params:
            partial_params = {name: complete_params[name] for name in folder_param}
            folder_name = self.param_scheme.get_param_string(partial_params)
            Path(folder_name).mkdir(exist_ok=True)
            os.chdir(folder_name)

        path_params = params.copy()
        for p in self.ignored_params_for_path:
            del path_params[p]

        file_name = self.param_scheme.get_param_string(path_params)
        if Path(file_name + "--metric.csv").is_file():
            print(f"Task already exists: {file_name}, exit...")
            os.chdir(original_dir)
            return
        else:
            f = open(file_name + ".lock", "w")
            f.close()

        try:
            self.func(**complete_params, file_name=file_name)
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            print("==================Task failed. exit... ========================")
            try:
                os.remove(file_name + "--metric.csv")
            except:
                pass

        os.remove(file_name + ".lock")
        os.chdir(original_dir)
