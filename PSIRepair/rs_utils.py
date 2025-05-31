import os
import io
import sys
import json
import enum
import math
import time
import importlib
import functools
import tempfile
import subprocess

try:
    import tqdm
except:
    print("[W] tqdm not found, install it with `pip install tqdm`")


# NOTE: Don't delete these flags
_ENABLE_INFO_LOG = True
_ENABLE_WARNING_LOG = True
_ENABLE_ERROR_LOG = True
_ENABLE_FATAL_LOG = True


def _init_logging_config():
    global _ENABLE_INFO_LOG, _ENABLE_WARNING_LOG, _ENABLE_ERROR_LOG, _ENABLE_FATAL_LOG
    if "RSU_LOG_LEVEL" in os.environ:
        level = os.environ["RSU_LOG_LEVEL"]
        if level == "INFO":
            enable_flags = [True, True, True, True]
        elif level == "WARNING":
            enable_flags = [False, True, True, True]
        elif level == "ERROR":
            enable_flags = [False, False, True, True]
        elif level == "FATAL":
            enable_flags = [False, False, False, True]
        else:
            raise ValueError(f"Unknown log level: {level}")
        _ENABLE_INFO_LOG = enable_flags[0]
        _ENABLE_WARNING_LOG = enable_flags[1]
        _ENABLE_ERROR_LOG = enable_flags[2]
        _ENABLE_FATAL_LOG = enable_flags[3]

    if "RSU_ENABLE_INFO_LOG" in os.environ:
        _ENABLE_INFO_LOG = bool(os.environ["RSU_ENABLE_INFO_LOG"])
    if "RSU_ENABLE_WARNING_LOG" in os.environ:
        _ENABLE_WARNING_LOG = bool(os.environ["RSU_ENABLE_WARNING_LOG"])
    if "RSU_ENABLE_ERROR_LOG" in os.environ:
        _ENABLE_ERROR_LOG = bool(os.environ["RSU_ENABLE_ERROR_LOG"])
    if "RSU_ENABLE_FATAL_LOG" in os.environ:
        _ENABLE_FATAL_LOG = bool(os.environ["RSU_ENABLE_FATAL_LOG"])


_init_logging_config()


class _LogLevel(enum.Enum):
    INFO = 0  # Green
    WARNING = 1  # Yellow
    ERROR = 2  # Red
    FATAL = 3  # Red

    def short_name(self):
        return self.name[0]

    def color_code(self):
        return ["32", "33", "31", "31"][self.value]


def _log_impl(level: _LogLevel, msg: str, fback: int, **kwargs):
    back_frame = sys._getframe()
    for _ in range(fback):
        back_frame = back_frame.f_back
    back_filename = os.path.basename(back_frame.f_code.co_filename)
    back_funcname = back_frame.f_code.co_name
    back_lineno = back_frame.f_lineno
    pos = msg.find(": ")
    pos = pos + 2 if pos != -1 else 0
    prefix, suffix = msg[:pos], msg[pos:]
    if eval(f"_ENABLE_{level.name}_LOG"):
        print(
            f"\033[0;{level.color_code()}m[{level.short_name()}]({back_filename}:{back_lineno}@{back_funcname}, timestamp={time.time()}) $$ {prefix}\033[0m {suffix}",
            **kwargs,
        )


def _plog(head, *obj, fback: int = 1, log=None, **kwargs):
    from pprint import pprint
    from io import StringIO

    log = log or _ilog
    msg_b = io.StringIO()
    for e in obj:
        pprint(e, stream=msg_b)
    buffer = io.StringIO()
    lines = map(lambda x: "  " + x, msg_b.getvalue().splitlines())
    log(f"{head}: ", fback=fback + 1, file=buffer)
    for line in lines:
        log(line, fback=fback + 1, file=buffer)
    print(buffer.getvalue(), end="", **kwargs)


def _ilog(*msg: str, fback: int = 1, **kwargs):
    msg = " ".join(map(lambda x: str(x), msg))
    _log_impl(_LogLevel.INFO, msg, fback + 1, **kwargs)


def _wlog(*msg: str, fback: int = 1, **kwargs):
    msg = " ".join(map(lambda x: str(x), msg))
    _log_impl(_LogLevel.WARNING, msg, fback + 1, **kwargs)


def _elog(*msg: str, fback: int = 1, **kwargs):
    msg = " ".join(map(lambda x: str(x), msg))
    _log_impl(_LogLevel.ERROR, msg, fback + 1, **kwargs)


def _flog(*msg: str, fback: int = 1, exp=None, exit_code=-1, **kwargs):
    msg = " ".join(map(lambda x: str(x), msg))
    _log_impl(_LogLevel.FATAL, msg, fback + 1, flush=True, **kwargs)
    if exp:
        raise exp
    raise RuntimeError("Fatal Error, exit with code {}".format(exit_code))


def _system(cmd: str) -> int:
    _ilog(f"system: {cmd}")
    code = os.system(cmd)
    _ilog(f"exit {cmd}: with {code}")
    return code


def _sp_system(
    cmds, logging=True, redirect_stdout=True, redirect_stderr=True, workdir=None
) -> int:
    import subprocess as sp

    assert isinstance(cmds, (str, tuple, list, set))
    if isinstance(cmds, str):
        cmd = cmds
    else:
        cmds = list(cmds)
        cmd = " ".join(cmds)
    if logging:
        _ilog("system: {}".format(cmd))
    try:
        ret = -1
        ps = sp.Popen(
            cmds,
            stdin=sys.stdin,
            stdout=sys.stdout if redirect_stdout else sp.DEVNULL,
            stderr=sys.stderr if redirect_stderr else sp.DEVNULL,
            shell=True,
            cwd=workdir,
        )
        ret = ps.wait()
    except KeyboardInterrupt:
        if "y" == input("exit? (y/n)").lower():
            exit(0)
    if logging:
        _ilog("exit {}: with {}".format(cmd, ret))
    return ret


def _sp_run(command, redirect_stderr_to_stdout=False, **kwargs):
    import subprocess as sp

    try:
        result = sp.run(
            command,
            shell=True,
            check=True,
            stdout=sp.PIPE,
            stderr=sp.PIPE if not redirect_stderr_to_stdout else sp.STDOUT,
            **kwargs,
        )

        return_code = result.returncode
        stdout = result.stdout.decode("utf-8")
        if not redirect_stderr_to_stdout:
            stderr = result.stderr.decode("utf-8")
        else:
            stderr = None

        if return_code != 0:
            _wlog(f"Run '{command}' failed: out='{stdout}', err='{stderr}'")

        return return_code, stdout, stderr
    except KeyboardInterrupt:
        if "y" == input("exit? (y/n)").lower():
            exit(0)
        return -1, "", ""
    except sp.CalledProcessError as e:
        stdout = e.output.decode("utf-8")
        if not redirect_stderr_to_stdout:
            stderr = e.stderr.decode("utf-8")
        else:
            stderr = None
        return e.returncode, stdout, stderr


def _split_list(origin_list: list, n: int):
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1

    for i in range(0, n):
        yield origin_list[i * cnt : (i + 1) * cnt]


def _split_list_as_chunks(origin_list: list, chunk_size: int):
    for i in range(0, len(origin_list), chunk_size):
        yield origin_list[i : i + chunk_size]


def _get_urec_key(user, repo, entry_file_relpath, commit):
    efp = entry_file_relpath.replace("\\", "/").replace("/", ".")
    return f"u{len(user)}-{user}-r{len(repo)}-{repo}-e{len(efp)}-{efp}-c{commit}"


def mark_processed(ckp_dir, user: str, repo: str, efp: str, commit: str):
    ckp_fn = _get_urec_key(user, repo, efp, commit)
    with open(f"{ckp_dir}/{ckp_fn}", "wb") as fp:
        pass


def clean_mark(ckp_dir, user, repo, efp, commit):
    ckp_fn = _get_urec_key(user, repo, efp, commit)
    _system("rm {ckp_dir}/{ckp_fn}")


def is_processed(ckp_dir, user: str, repo: str, efp: str, commit: str):
    ckp_fn = _get_urec_key(user, repo, efp, commit)
    return os.path.exists(f"{ckp_dir}/{ckp_fn}")


def flat_path(path: str) -> str:
    assert ":" not in path
    return path.replace("\\", ".").replace("/", ".").replace(":", ".")


def _iter_repos(repos_path: str):
    for user in os.listdir(repos_path):
        for repo in os.listdir(f"{repos_path}/{user}"):
            yield user, repo


def _load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def _save_as_json(obj, filename, encoding=None, auto_mkdir=False):
    if auto_mkdir:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding=encoding or "UTF-8") as fp:
        json.dump(obj, fp)


def _save_txt(text: str, filename, encoding=None, auto_mkdir=False):
    if auto_mkdir:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding=encoding or "UTF-8") as fp:
        fp.write(text)


_log_fn_call_call_level = 0


def _log_fn_call(fn=None, *, default=True, enable=None, args=True, ret=True):
    enable_args = args
    enable_ret = ret

    def make_obj_str(arg):
        arg_str = [*str(arg).splitlines(), ""]
        return (
            (arg_str[0][:40] + "..." + arg_str[0][-40:])
            if len(arg_str[0]) > 80
            else arg_str[0]
        )

    class PrintCallInfo:
        def __init__(self, fn, args, kwargs, enable, enable_args, enable_ret) -> None:
            self.fn = fn
            self.args = args
            self.kwargs = kwargs
            self.ret = (None,)
            self.log = _ilog if enable else lambda *args, **kwargs: ...
            self.enable_args = enable_args
            self.enable_ret = enable_ret

        def save_ret(self, ret):
            self.ret = ret
            if not isinstance(self.ret, (tuple, list)):
                self.ret = (self.ret,)
            return ret

        def __enter__(self):
            fn_name = self.fn.__name__
            global _log_fn_call_call_level
            _log_fn_call_call_level += 1
            self.log(
                f'[Call] {"="*_log_fn_call_call_level}>: {self.fn.__name__}', fback=3
            )
            if self.enable_args:
                for i, arg in enumerate(self.args):
                    self.log(
                        f'[Call:{fn_name}] {"="*_log_fn_call_call_level}>: args[{i}] = {make_obj_str(arg)}',
                        fback=3,
                    )
                for name, arg in self.kwargs.items():
                    self.log(
                        f'[Call:{fn_name}] {"="*_log_fn_call_call_level}>: kwargs[{name}] = {make_obj_str(arg)}',
                        fback=3,
                    )
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            fn_name = self.fn.__name__
            global _log_fn_call_call_level
            _log_fn_call_call_level -= 1
            if self.enable_ret:
                for i, r in enumerate(self.ret):
                    self.log(
                        f'[Call:{fn_name}] {"="*_log_fn_call_call_level}>: ret[{i}] = {make_obj_str(r)}',
                        fback=3,
                    )
            else:
                self.log(
                    f'[Call:{fn_name}] {"="*_log_fn_call_call_level}>: ret = ...',
                    fback=3,
                )

    def try_eval(exp, glbs, default):
        try:
            return eval(exp, glbs)
        except:
            return default

    def inner(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            configs = {
                "enable": bool(try_eval(enable, kwargs.copy(), default)),
                "enable_args": bool(enable_args),
                "enable_ret": bool(enable_ret),
            }
            with PrintCallInfo(fn, args, kwargs, **configs) as p:
                return p.save_ret(fn(*args, **kwargs))

        return wrapped

    return inner(fn) if fn else inner


class IteratorWithLen:
    def __init__(self, length, iter, **kwargs):
        self.__len = length
        self.__iter = iter

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        return self.__len()

    def __iter__(self):
        return self.__iter()


def _parse_cmd_as_call_like(argv, onerror=None):  # -> func, args, kwargs
    def raise_ex(ex):
        raise ex

    def parse_arg_as_name(arg: str):
        if arg.startswith("--"):
            return arg[len("--") :].replace("-", "_")
        elif arg.startswith("-"):
            return arg[len("-") :].replace("-", "_")
        else:
            return arg.replace("-", "_")

    assert len(argv) >= 1
    onerror = onerror or raise_ex
    func = argv[0]
    args, kwargs = [], {}
    rem_args, pos = argv[1:], 0
    rem_args_count = len(rem_args)
    try:
        while pos < rem_args_count:
            arg: str = rem_args[pos]
            if arg.startswith("--") or arg.startswith("-"):
                name = parse_arg_as_name(arg)
                if pos + 1 >= rem_args_count:
                    raise ValueError(f"Expected a value specified for `{arg}`")
                val = rem_args[pos + 1]
                kwargs[name] = val
                pos += 2
            else:
                args.append(arg)
                pos += 1
    except Exception as ex:
        onerror(ex)

    return func, args, kwargs


class RedirectStdOutErrToFileUseSysIO:
    def __init__(self, *, file_path=None, open_mode=None, filep=None):
        # _wlog('RedirectStdOutErrToFileUseSysIO is not recommended')
        assert file_path or filep
        self.file_path = file_path
        self.open_mode = open_mode or "w"
        self.filep = filep
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def __enter__(self):
        if not self.filep:
            self.filep = open(self.file_path, self.open_mode, buffering=1)
        sys.stdout = self.filep
        sys.stderr = self.filep
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        if self.file_path:
            self.filep.close()


class RedirectStdOutErrToFileUseDup:
    def __init__(self, *, file_path=None, filep=None):
        _flog("RedirectStdOutErrToFileUseDup has bug now")
        assert file_path or filep
        self.file_path = file_path
        self.filep = filep
        self.ori_stdout_fd = None
        self.ori_stderr_fd = None

    def __enter__(self):
        if not self.filep:
            self.filep = open(self.file_path, "w", buffering=1)
        self.ori_stdout_fd = os.dup(sys.stdout.fileno())
        self.ori_stderr_fd = os.dup(sys.stderr.fileno())
        os.dup2(self.filep.fileno(), sys.stdout.fileno())
        os.dup2(self.filep.fileno(), sys.stderr.fileno())
        # self.stdout = os.fdopen(self.ori_stdout_fd, 'w', buffering=1)
        # self.stderr = os.fdopen(self.ori_stderr_fd, 'w', buffering=1)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self.ori_stdout_fd, sys.stdout.fileno())
        os.dup2(self.ori_stderr_fd, sys.stderr.fileno())
        os.close(self.ori_stdout_fd)
        os.close(self.ori_stderr_fd)
        if self.file_path:
            self.filep.close()


if sys.platform == "win32":
    RedirectStdOutErrToFile = RedirectStdOutErrToFileUseSysIO
else:
    RedirectStdOutErrToFile = RedirectStdOutErrToFileUseSysIO


def _tqdm(iter, *, title=None, log2file=True, file=-1, len=None, **kwargs):
    file = sys.stdout if file == -1 else file
    if file:
        if not log2file:
            return tqdm.tqdm(iter, file=file, desc=title, total=len, **kwargs)
        title = f"[{title or '_tqdm'}] "
        len = len or (getattr(iter, "__len__", None) or (lambda: None))()
        print_1_per = math.ceil(len / 80) if len else -1
        _ilog(f"{title}: ", end="", flush=True, fback=2, file=file)
        if print_1_per < 0:
            print("doing...", end="", flush=True, file=file)
        for i, e in enumerate(iter):
            yield e
            if print_1_per > 0 and i % print_1_per == 0:
                print("#", end="", flush=True, file=file)
        print(" [DONE]", flush=True, file=file)
    else:
        yield from iter


def _load_config_from_python(filename, name=None):
    name = name or "config"
    spec = importlib.util.spec_from_file_location("custom_module", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_variable = getattr(module, name)
    return config_variable


# NOTE: No error proecessing
def _simple_clike_preprocess(source: str, macros) -> str:
    m_define = "#define"
    m_undef = "#undef"
    m_if = "#if"
    m_elif = "#elif"
    m_else = "#else"
    m_endif = "#endif"
    m_line = "#line"
    all_m = [m_define, m_undef, m_if, m_elif, m_else, m_endif, m_line]
    if_stack = [True]

    class ArgParser:
        def __init__(self, string: str):
            self.string = string
            self.pos = 0
            self.eat_all_spaces()

        def eat_all_spaces(self):
            while self.pos < len(self.string) and self.string[self.pos].isspace():
                self.pos += 1

        def next_arg(self) -> str:
            last_pos = self.pos
            while self.pos < len(self.string) and not self.string[self.pos].isspace():
                self.pos += 1
            result = self.string[last_pos : self.pos]
            self.eat_all_spaces()
            return result

        def rem_arg(self) -> str:
            result = self.string[self.pos :]
            self.pos = len(self.string)
            return result

    def try_parse(line: str):
        line = line.strip()
        if not any(line.startswith(m) for m in all_m):
            return None
        arg_parser = ArgParser(line)
        return arg_parser.next_arg(), arg_parser

    def process_m(m: str, arg_parser: ArgParser) -> str:
        enable = all(if_stack)
        if m == m_define:
            if enable:
                constant = arg_parser.next_arg()
                macros[constant] = arg_parser.rem_arg()
        elif m == m_undef:
            if enable:
                macros.pop(args[0], None)
        elif m == m_if:
            if_stack.append(enable and eval(arg_parser.rem_arg(), macros))
        elif m == m_elif:
            if_stack[-1] = (
                all(if_stack[:-1])
                and not if_stack[-1]
                and eval(arg_parser.rem_arg(), macros)
            )
        elif m == m_else:
            if_stack[-1] = all(if_stack[:-1]) and not if_stack[-1]
        elif m == m_endif:
            if_stack.pop(-1)
        elif m == m_line:
            if enable:
                return str(eval(arg_parser.rem_arg(), macros))

    lines = source.splitlines()
    out_lines = []
    for line in lines:
        m = try_parse(line)
        if m:
            eline = process_m(*m)
            if eline:
                out_lines.append(eline)
        elif if_stack[-1]:
            out_lines.append(line)
    return "\n".join(out_lines)


def _remove_common_prefix_suffix(str1, str2):
    common_prefix = os.path.commonprefix([str1, str2])
    common_suffix = os.path.commonprefix([str1[::-1], str2[::-1]])[::-1]

    if common_prefix:
        str1 = str1[len(common_prefix) :]
        str2 = str2[len(common_prefix) :]

    if common_suffix:
        str1 = str1[: -len(common_suffix)]
        str2 = str2[: -len(common_suffix)]

    return str1, str2


@_log_fn_call
def _CMD_clike_preprocess(f, o, **macros):
    with open(f, "r", encoding="UTF-8") as fp:
        text = fp.read()
    with open(o, "w", encoding="UTF-8") as fp:
        fp.write(_simple_clike_preprocess(text, macros))
    macros.pop("__builtins__", None)
    return macros


def _try_import(module_name, base_dir=None):
    module_name = os.path.abspath(module_name)
    if os.path.exists(module_name):
        base_dir = os.path.dirname(module_name)
        module_name = os.path.basename(module_name).replace(".py", "")
    try:
        import sys

        sys.path.append(base_dir)
        return __import__(module_name)
    except Exception as ex:
        _wlog(f"Import {module_name} failed: {ex}")
        return None
    finally:
        sys.path.remove(base_dir)


def _available_port():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    _, port = s.getsockname()
    s.close()
    return port


def _make_diff(src, trg):
    import difflib

    diff = difflib.ndiff(src.splitlines(), trg.splitlines())
    return "\n".join(diff)


def _make_diff_html(src, trg, src_title="bug", trg_title="fix"):
    import difflib

    return difflib.HtmlDiff().make_file(
        src.splitlines(), trg.splitlines(), src_title, trg_title
    )


class TimeCounter:
    __result_stack = [
        {},
    ]  # [tag => duration(nano), ...]

    @classmethod
    def __add_result(cls, tag, duration):
        if tag not in cls.__result_stack[-1]:
            cls.__result_stack[-1][tag] = 0
        cls.__result_stack[-1][tag] += duration

    @classmethod
    def push_result(cls, result=None):
        result = result or {}
        cls.__result_stack.append(result)

    @classmethod
    def pop_result(cls):
        result = cls.__result_stack.pop()
        if not cls.__result_stack:
            cls.push_result()
        return result

    @classmethod
    def peek_result(cls):
        return cls.__result_stack[-1]

    @classmethod
    def clear_result(cls):
        cls.__result_stack = [
            {},
        ]

    def __init__(self, *args, sep=":"):
        self.__tag = sep.join(args)

    def __enter__(self):
        self.__start = time.time_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__class__.__add_result(self.__tag, time.time_ns() - self.__start)


def _tcfor(iter, tag_maker):
    for e in iter:
        with TimeCounter(*tag_maker(e)):
            yield e


class FileLock:
    def __init__(self, lockfile, retry=-2, interval=0.1):
        self.__lockfile = lockfile
        self.__retry = retry
        self.__interval = interval
        self.__fd = None

    @property
    def locked(self):
        return self.__fd is not None

    def try_lock(self):
        self.__enter__()
        return self.locked

    def unlock(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        _ilog(f"Try to lock file: {self.__lockfile}", fback=2)
        try_ = self.__retry + 1
        while try_:
            try:
                self.__fd = os.open(self.__lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                _ilog(f"Locked file: {self.__lockfile}", fback=2)
                return self
            except FileExistsError:
                if try_ >= 0:
                    _wlog(f"File locked: try {try_} times left", fback=2)
                try_ -= 1
                time.sleep(self.__interval)
        assert self.__fd is None
        _wlog(f"Lock file failed: {self.__lockfile}", fback=2)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.__fd is not None:
            os.close(self.__fd)
            os.remove(self.__lockfile)
            _ilog(f"Unlocked file: {self.__lockfile}", fback=2)


def _append_to_file(filename, content, *, suffix=".lock", encoding="UTF-8"):
    filename = os.path.abspath(filename)
    with FileLock(filename + suffix):
        with open(filename, "a", encoding=encoding) as fp:
            fp.write(content)


def _touch_gpu():
    if os.getenv("PGRSU_DISABLE_TOUCH_GPU", "0") == "1":
        _ilog("Touch GPU is disabled")
        return

    import torch

    n_gpus = torch.cuda.device_count()

    if n_gpus == 0:
        _wlog("No GPU found")
        return
    _ilog(f"Touching {n_gpus} GPUs...")

    devices = [torch.device(f"cuda:{i}") for i in range(n_gpus)]

    x_list = []
    for d in devices:
        try:
            x = torch.zeros((1024**3), device=d)
            x_list.append(x)
        except Exception as ex:
            _wlog(f"Allocating memory on {d} failed: {ex}")

    for x in x_list:
        for i in range(0, len(x), 1024 * 1024):  # touch every 1MB
            x[i] = i % 1000

    for x in x_list:
        del x


def _random_touch_gpu():
    import random

    return _touch_gpu() if random.random() <= 0.5 else None


_last_touch_gpu_time = 0


def _schedule_touch_gpu(interval=10 * 60):  # default 10 minutes
    global _last_touch_gpu_time
    if time.time() - _last_touch_gpu_time > interval:
        _touch_gpu()
        _last_touch_gpu_time = time.time()


def _get_env_desc():
    import subprocess
    import re
    import os

    def get_os_version():
        try:
            result = subprocess.run(["lsb_release", "-d"], stdout=subprocess.PIPE)
            return result.stdout.decode("utf-8").split(":")[1].strip()
        except Exception as e:
            return "Unknown OS"

    def get_cpu_info():
        try:
            result = subprocess.run(["lscpu"], stdout=subprocess.PIPE)
            output = result.stdout.decode("utf-8")
            cores = re.search(r"CPU\(s\):\s+(\d+)", output).group(1)
            speed = re.search(r"MHz:\s+(\d+.\d+)", output).group(1)
            return cores, float(speed) / 1000
        except Exception as e:
            return "Unknown", -1.0

    def get_ram_info():
        try:
            result = subprocess.run(["free", "--mega"], stdout=subprocess.PIPE)
            output = result.stdout.decode("utf-8")
            mem = re.search(r"Mem:\s+(\d+)", output).group(1)
            return int(mem) / 1024  # Convert MB to GB
        except Exception as e:
            return "Unknown"

    def get_gpu_info():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=gpu_name,memory.total",
                    "--format=csv,noheader",
                ],
                stdout=subprocess.PIPE,
            )
            output = result.stdout.decode("utf-8").strip().split("\n")[0].split(",")
            gpu_name = output[0].strip()
            memory = (
                output[1].strip().split(" ")[0]
            )  # Assuming the memory is listed in MiB
            return gpu_name, float(memory) / 1024  # Convert MiB to GiB
        except Exception as e:
            return "Unknown", -1.0

    # Gather information
    os_version = get_os_version()
    cpu_cores, cpu_speed = get_cpu_info()
    ram_gb = get_ram_info()
    gpu_name, gpu_memory_gb = get_gpu_info()

    # Format the template
    template = f"""All the experiments are conducted on {os_version} server \
with {cpu_cores} cores of {cpu_speed:.1f}GHz CPU, {ram_gb:.0f}GB RAM and NVIDIA \
{gpu_name} with {gpu_memory_gb}GB memory."""

    return template


def _CMD_print_env_desc():
    print(_get_env_desc())


def _abstractmethod(fn):
    from abc import abstractmethod

    fn = abstractmethod(fn)

    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        raise NotImplementedError(f"{fn.__name__} is not implemented")

    return wrapped


class FnCallRecorder:
    def __init__(self, file=None):
        self.__calls = []
        self.__file = file

        if isinstance(file, str) and file.endswith(".jsonl"):
            self.__file = open(file, "w")
        elif callable(getattr(file, "write", None)):
            self.__file = file
        else:
            raise ValueError("file must be a filename(.jsonl) or a file-like object")

    @classmethod
    def load(cls, file: str):
        assert file.endswith(".jsonl")
        self = cls()
        with open(file, "r") as f:
            for line in f:
                self.__calls.append(json.loads(line))
        return self

    def save(self, file: str):
        assert file.endswith(".jsonl")
        with open(file, "w") as f:
            for call in self.__calls:
                f.write(json.dumps(call) + "\n")

    def record(self, fn_name, args, kwargs, ret):
        self.__calls.append(
            {
                "fn": fn_name,
                "args": args,
                "kwargs": kwargs,
                "ret": ret,
            }
        )
        if self.__file:
            self.__file.write(
                json.dumps({"fn": fn_name, "args": args, "kwargs": kwargs, "ret": ret})
                + "\n"
            )

    def query(self, fn_name, *args, **kwargs):
        for call in self.__calls:
            if (
                call["fn"] == fn_name
                and call["args"] == args
                and call["kwargs"] == kwargs
            ):
                return call["ret"]
        return None

    def match_and_pop(self, fn_name, *args, **kwargs):
        front = self.__calls[0]
        if (
            front["fn"] == fn_name
            and front["args"] == args
            and front["kwargs"] == kwargs
        ):
            self.__calls.pop(0)
            return front["ret"]
        return None


def _make_temp_file():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        return temp_file.name


class _TempFileMgr:
    def __init__(self, n=1, auto_delete=True):
        self.__n = n
        self.__files = None
        self.__auto_delete = auto_delete

    @property
    def files(self):
        return self.__files

    def __enter__(self):
        self.__files = [_make_temp_file() for _ in range(self.__n)]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.__auto_delete:
            for f in self.__files:
                os.remove(f)


def _make_temp_file_mgr(n=1, auto_delete=True):
    return _TempFileMgr(n, auto_delete)


def _git(command, repo_dir):
    return subprocess.check_output(
        ["git"] + command, stderr=subprocess.DEVNULL, cwd=repo_dir, text=True
    )


def _read_text_file(fn: str, encoding=None, return_encoding=False):
    import chardet

    with open(fn, "rb") as file:
        raw_data = file.read()

    if encoding is None:
        encoding = chardet.detect(raw_data)["encoding"]
        encoding = encoding or "utf-8"
    assert encoding is not None

    content = raw_data.decode(encoding)
    if not return_encoding:
        return content
    return content, encoding


def _parse_ints(s: str):
    # e.g. 1,2,3,4-6,7
    ints = []
    parts = s.split(",")
    for p in parts:
        p = p.split("-")
        if len(p) == 1:
            ints.append(int(p[0]))  # 1
        elif len(p) == 2:
            ints.extend(list(range(int(p[0]), int(p[1]) + 1)))  # 4-7
        else:
            raise ValueError(f"Invalid range in `{s}`: `{p}`")
    return ints


def _set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import random

        random.seed(seed)
    except:
        pass

    try:
        import numpy as np

        np.random.seed(seed)
    except:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except:
        pass


def _check_kw_arg(*, arg_name: str, required_keys: list):
    def inner(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if not isinstance(kwargs[arg_name], dict):
                raise ValueError(f"{arg_name} must be a dict")
            _required_keys = []
            for k in required_keys:
                if isinstance(k, tuple):
                    if k[1](kwargs[arg_name]):
                        _required_keys.append(k[0])
                elif isinstance(k, str):
                    _required_keys.append(k)
                else:
                    raise ValueError(f"Invalid key: {k}")
            missing_keys = set(_required_keys) - set(kwargs[arg_name].keys())
            if missing_keys:
                raise ValueError(f"{arg_name} is missing keys: {missing_keys}")
            return fn(*args, **kwargs)

        return wrapped

    return inner


def _extract_tar_gz(
    archive_path: str,
    destination_directory: str,
    files_or_dirs: str | list[str] | None = None,
) -> None:
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"Archive file not found: {archive_path}")
    if os.path.isdir(destination_directory):
        raise ValueError(
            f"Destination directory already exists: {destination_directory}"
        )

    os.makedirs(destination_directory, exist_ok=False)

    cmd = ["tar", "-zxf", archive_path, "-C", destination_directory]
    if files_or_dirs:
        if isinstance(files_or_dirs, str):
            cmd.append(files_or_dirs)
        else:
            cmd.extend(files_or_dirs)

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() or "Unknown error occurred"
        raise RuntimeError(f"Extraction failed: {error_msg}") from e


if __name__ == "__main__":
    import os, sys

    if len(sys.argv) < 2:
        print(f"Usage: {os.path.basename(__file__)} <subcmd> --arg0 <arg0> ...")
        sys.exit(0)

    fn, args, kwargs = _parse_cmd_as_call_like(sys.argv[1:])
    print(f'Call: {fn}({", ".join([*args, *[k+"="+v for k, v in kwargs.items()]])})')

    eval(f"_CMD_{fn}")(*args, **kwargs)
