import os
import time
import shutil
import subprocess as sp

from dataclasses import dataclass
from typing import Optional, Any, Tuple, List, Dict, Literal
from . import rs_utils as rsu
from .basic import BugInfo, CodeLocation
from .utils import _is_debug_mode, _get_java_method_code_lines


@dataclass
class RunTestResult:
    @dataclass
    class FailingTestInfo:
        md_path: str
        raised_exception: str
        line_raised_at: int
        file_raised_at: str
        traceback: str

    timeout: bool
    passed: bool
    failing_tests: List[FailingTestInfo]

    # Only for defects4j projects
    _stderr: Optional[str]
    _stdout: Optional[str]
    _full_output: Optional[str]


class Project:
    @rsu._abstractmethod
    def get_lang(self) -> str:
        pass

    @rsu._abstractmethod
    def get_itendifier(self, ignore_project_version: bool = True) -> str:
        pass

    @rsu._abstractmethod
    def get_root_path(self) -> str:
        pass

    @rsu._abstractmethod
    def get_bugs(self, _filter=None) -> List[BugInfo]:
        pass

    @rsu._abstractmethod
    def checkout(self, bug: BugInfo, b_mark: Literal["b", "f"]) -> "Project":
        pass

    @rsu._abstractmethod
    def run_compile(self) -> Tuple[int, str, str]:
        pass

    @rsu._abstractmethod
    def run_test(
        self,
        relevant_bug: BugInfo,
        strategy: Literal["trigger->all", "trigger", "all"],
    ) -> RunTestResult:
        pass

    @rsu._abstractmethod
    def get_method_code_lines(
        self, file_path: str, method_path: str, strict: bool
    ) -> List[Tuple[int, str]]:
        pass

    @rsu._abstractmethod
    def get_test_method_code_lines(
        self, test_method_full_path: str, return_file_path: bool = False
    ) -> List[Tuple[int, str]]:
        pass

    @rsu._abstractmethod
    def apply_patch(self, file_path: str, method_path: str, patch: str):
        pass


class JavaProject(Project):
    def get_lang(self) -> str:
        return "java"

    @rsu._abstractmethod
    def get_java_home(self) -> str:
        pass

    @rsu._abstractmethod
    def get_java_version(self) -> str:
        pass

    @rsu._abstractmethod
    def get_src_sourcepath(self) -> str:
        pass

    @rsu._abstractmethod
    def get_tests_sourcepath(self) -> str:
        pass

    @rsu._abstractmethod
    def get_test_classpath(self) -> str:
        pass


class Defects4JProject(JavaProject):
    """
    Defects4J: https://github.com/rjust/defects4j
    """

    def __init__(
        self,
        project_name: str,
        bug_ids: int,
        d4j_version: str,
        d4j_cli: str,
        *,
        project_path: str = None,
        project_source_path: str = None,
        tests_source_path: str = None,
        project_version: str = None,
        d4j_bugs_stat=None,
    ):
        if d4j_bugs_stat is not None:  # The repo-like project, HEAD=None
            assert project_path is None
            assert project_source_path is None
            assert project_version is None
        else:  # The version-like project, HEAD=@project_version
            assert project_path is not None
            assert project_source_path is not None
            assert project_version is not None

        self.__project_name = project_name
        self.__bug_ids = bug_ids
        self.__d4j_version = d4j_version
        self.__d4j_cli = os.path.abspath(d4j_cli)
        self.__project_path = project_path
        self.__project_version = project_version
        self.__d4j_bugs_stat = d4j_bugs_stat and {
            int(info["bid"]): info for info in d4j_bugs_stat
        }

        if self.__d4j_version == "v1.2.0":
            self.__java_home = os.environ["JAVA7_HOME"]
            self.__java_version = "java7"
        elif self.__d4j_version == "v2.0.0":
            self.__java_home = os.environ["JAVA8_HOME"]
            self.__java_version = "java8"
        else:
            raise ValueError(f"Unsupported defects4j version: {self.__d4j_version}")

        self.__d4j_cli_cmd = f"JAVA_HOME={self.__java_home} {self.__d4j_cli}"

        if self.__project_path:
            assert project_source_path is not None
            assert tests_source_path is not None
            self.__src_path = f"{self.__project_path}/{project_source_path}"
            self.__tests_src_path = f"{self.__project_path}/{tests_source_path}"
            self._test_classpath = None

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"Defects4JProject[{self.__d4j_version}]({self.__project_name}@{self.__project_version})"

    def _d4j_cli(self, *args, workdir: str = None, timeout: float = None):
        cmd = f"{self.__d4j_cli_cmd} {' '.join(args)}"
        if _is_debug_mode():
            start_time = time.time()
            rsu._ilog(
                f"Run d4j: `{cmd}`, workdir: `{workdir}`, start at: `{start_time}`"
            )
        ret, stdout, stderr = rsu._sp_run(cmd, cwd=workdir, timeout=timeout)
        if _is_debug_mode():
            end_time = time.time()
            rsu._ilog(f">>>> end at: {end_time}")
            rsu._ilog(f">>>> duration: {end_time - start_time}s`")
            rsu._ilog(f">>>> return code: {ret}")
            rsu._ilog(f">>>> stdout:`{stdout}`")
            rsu._ilog(f">>>> stderr:`{stderr}`")
        return ret, stdout, stderr

    def _get_property(self, key: str) -> str:
        ret, out, err = self._d4j_cli(f"export -p {key} -w {self.__project_path}")
        if ret != 0 and f"Unknown property {key}" in err:
            raise ValueError(f"Unknown property {key}")
        if ret != 0:
            raise ValueError(f"Failed to get property: {key}: {err}")
        return out.strip()

    def get_java_home(self) -> str:
        return self.__java_home

    def get_java_version(self) -> str:
        return self.__java_version

    def get_src_sourcepath(self) -> str:
        return self.__src_path

    def get_test_sourcepath(self) -> str:
        return self.__tests_src_path

    def get_test_classpath(self) -> str:
        if self._test_classpath is None:
            self._test_classpath = self._get_property("cp.test")
        return self._test_classpath

    def get_lang(self) -> str:
        return "java"

    def get_itendifier(self, ignore_project_version: bool = True) -> str:
        itendifier = f'd4j-{self.__d4j_version.replace(".", "")}-{self.__project_name}'
        if not ignore_project_version:
            itendifier += f"-{self.__project_version}"
        return itendifier

    def get_root_path(self) -> str:
        return self.__project_path

    def get_bugs(self, _filter=None) -> List[BugInfo]:
        _filter = _filter or (lambda b: True)
        assert callable(_filter)

        d4j_info = self.__d4j_bugs_stat

        return list(
            filter(
                _filter,
                (
                    BugInfo(
                        project=self,
                        bug_locations=[
                            CodeLocation(
                                file_path=line["file"],
                                lineno=line["lineno"],
                                method_path=line["method"],
                            )
                            for line in d4j_info[bid]["buggy_lines"]
                        ],
                        num_buggy_lines=d4j_info[bid]["num_buggy_lines"],
                        is_single_line_bug=d4j_info[bid]["is_single_line_bug"],
                        is_single_method_bug=d4j_info[bid]["is_single_method_bug"],
                        is_single_file_bug=d4j_info[bid]["is_single_file_bug"],
                        all_tests=d4j_info[bid]["tests.all"].splitlines(),
                        relevant_tests=d4j_info[bid]["tests.relevant"].splitlines(),
                        trigger_tests=d4j_info[bid]["tests.trigger"].splitlines(),
                        _interal_id=f"{self.__project_name}_{bid}",
                    )
                    for bid in self.__bug_ids
                    if bid in d4j_info
                ),
            )
        )

    def checkout(self, bug: BugInfo, b_mark: Literal["b", "f"]) -> "Defects4JProject":
        pid, bid = bug._interal_id.split("_")
        assert self.__project_name == pid
        assert self.__d4j_bugs_stat is not None

        # defects4j checkout -p Chart -v 1b -w ~/tmp/Chart_1b
        p_path = f"{os.environ['D4J_CHECKOUT_TMP_PATH']}/{pid}_{bid}{b_mark}"

        if os.path.exists(p_path):
            if os.getenv("CAPREPAIR_CLEAN_PREV_CHECKOUT", None) == "1":
                rsu._wlog(f"Enforce remove previous checkout: {p_path}")
                shutil.rmtree(p_path)
            elif not (
                rsu._sp_run(f"cd '{p_path}' && git checkout -- .")[0] == 0
                and rsu._sp_run(f"cd '{p_path}' && git clean -fd")[0] == 0
            ):
                rsu._wlog(f"Remove previous checkout: {p_path}, git clean failed")
                shutil.rmtree(p_path)

        if not os.path.exists(p_path):
            ret, out, err = self._d4j_cli(
                "checkout", "-p", pid, "-v", f"{bid}{b_mark}", "-w", p_path
            )
            if ret != 0:
                rsu._wlog(f"Checkout failed: out='{out}', err='{err}'")
                raise RuntimeError(f"Checkout failed: {pid}-{bid}{b_mark} {p_path}")

        # NOTE: Check git diff: no changes
        assert rsu._sp_run(f"git diff", cwd=p_path) == (0, "", "")

        if pid == "Mockito" and bid in ["1", "5", "7", "8", "18", "20"]:
            rsu._ilog(f"Fixing build scripts for: Mockito-{bid}")
            build_gradle_fn = f"{p_path}/build.gradle"
            init_gradle_fn = f"{p_path}/.gradle_local_home/init.gradle"
            assert os.path.isfile(build_gradle_fn)

            init_gradle = """\
allprojects{
    repositories {
        def REPOSITORY_URL = 'http://maven.aliyun.com/nexus/content/groups/public/'
        all { ArtifactRepository repo ->
            if(repo instanceof MavenArtifactRepository){
                def url = repo.url.toString()
                if (url.startsWith('https://repo1.maven.org/maven2') || url.startsWith('https://jcenter.bintray.com/')) {
                    project.logger.lifecycle "Repository ${repo.url} replaced by $REPOSITORY_URL."
                    remove repo
                }
            }
        }
        maven {
            url REPOSITORY_URL
        }
    }
}"""
            fix_jcenter = """\
jcenter { url 'http://maven.aliyun.com/nexus/content/repositories/jcenter' }
maven { url 'https://maven.aliyun.com/nexus/content/groups/public' }"""

            with open(build_gradle_fn, "r") as f:
                build_gradle = f.read().replace("jcenter()", f"\n\n{fix_jcenter}\n\n")
            rsu._save_txt(build_gradle, build_gradle_fn, auto_mkdir=True)
            rsu._save_txt(init_gradle, init_gradle_fn, auto_mkdir=True)
            rsu._ilog(f">>>> Patched build.gradle: {build_gradle_fn}")
            rsu._ilog(f">>>> Patched init.gradle: {init_gradle_fn}")

        return Defects4JProject(
            project_name=self.__project_name,
            bug_ids=self.__bug_ids,
            d4j_version=self.__d4j_version,
            d4j_cli=self.__d4j_cli,
            project_path=p_path,
            project_version=f"{bid}{b_mark}",
            project_source_path=self.__d4j_bugs_stat[int(bid)]["dir.src.classes"],
            tests_source_path=self.__d4j_bugs_stat[int(bid)]["dir.src.tests"],
        )

    def run_compile(self) -> Tuple[int, str, str]:
        """
        ret == 0: Compile Success, otherwise Compile Failed
        """
        assert self.__project_path is not None
        assert self.__project_version is not None
        return self._d4j_cli("compile", workdir=self.__project_path)

    def run_test(
        self,
        relevant_bug: BugInfo = None,
        strategy: Literal["trigger->all", "all"] = "trigger->all",
        test_case_method_full_path: str = None,
    ) -> RunTestResult:

        def _parse_failing_tests(
            stdout: str, f_output: str
        ) -> List[RunTestResult.FailingTestInfo]:
            # stdout: from stdout of `defects4j test`
            # f_output: from `failing_test` file saved by `defects4j test`
            stdout_lines = stdout.strip().splitlines()
            ft, ftc = stdout_lines[0].strip().split(": ")
            assert ft.strip() == "Failing tests"
            num_failing_tests = int(ftc.strip())
            failing_test_names = [
                line.removeprefix("  - ").strip() for line in stdout_lines[1:]
            ]
            assert len(failing_test_names) == num_failing_tests
            failing_tests = []
            for p in f_output.strip().split("--- ")[1:]:
                p_lines = p.strip().splitlines()
                md_path = p_lines[0].strip()
                possible_raise_lines = list(
                    filter(
                        lambda L: L.strip().startswith(
                            f"at {md_path.replace('::', '.')}"
                        ),
                        p_lines[2:],
                    )
                )
                if possible_raise_lines:
                    line = possible_raise_lines[0]
                    assert line.endswith(")")
                    filename, lineno = line[line.rfind("(") + 1 : -1].split(":")
                    del line
                else:
                    filename, lineno = "<unknown>", "-1"
                failing_tests.append(
                    RunTestResult.FailingTestInfo(
                        md_path=md_path,
                        raised_exception=p_lines[1].strip(),
                        line_raised_at=int(lineno),
                        file_raised_at=filename,
                        traceback="\n".join(p_lines[1:]),
                    )
                )
            assert num_failing_tests == len(failing_test_names) == len(failing_tests)
            if set(failing_test_names) != set(t.md_path for t in failing_tests):
                rsu._elog("failing_test_names:", failing_test_names)
                rsu._elog("failing_tests:", failing_tests)

            return failing_tests

        def _test(t: str) -> RunTestResult:
            # Run tests
            t_args = ["-t", t] if t else []
            timeout_sec = 10 * 60 if t else 60 * 60  # 10m for single test, 1h for all
            try:
                ret, stdout, stderr = self._d4j_cli(
                    "test", *t_args, workdir=self.__project_path, timeout=timeout_sec
                )
            except sp.TimeoutExpired:
                return RunTestResult(
                    timeout=True,
                    passed=False,
                    failing_tests=["<Timeout>"],
                    _stdout="<Timeout>",
                    _stderr="<Timeout>",
                    _full_output="<Timeout>",
                )
            assert ret == 0, "always 0 now"
            f_output = rsu._read_text_file(
                f"{self.__project_path}/failing_tests", encoding="UTF-8"
            )
            # Parse test results
            failing_tests = _parse_failing_tests(stdout, f_output)

            if _is_debug_mode():
                rsu._ilog("Result of testing:")
                # rsu._ilog(f">>>> stdout: `{stdout}`")  # log in self._d4j_cli(...)
                # rsu._ilog(f">>>> stderr: `{stderr}`")  # log in self._d4j_cli(...)
                rsu._ilog(f">>>> f_output: `{f_output}`")
                rsu._ilog(f">>>> num_failing_tests: {len(failing_tests)}")
                rsu._plog(f">>>> failing_tests", [t.md_path for t in failing_tests])

            return RunTestResult(
                timeout=False,
                failing_tests=failing_tests,
                passed=len(failing_tests) == 0,
                _stdout=stdout,
                _stderr=stderr,
                _full_output=f_output,
            )

        if test_case_method_full_path is not None:
            return _test(t=test_case_method_full_path)
        if strategy == "all":
            return _test(t=None)
        elif strategy == "trigger->all":
            assert relevant_bug is not None
            result = RunTestResult(
                timeout=False,
                passed=True,
                failing_tests=[],
                _stderr="",
                _stdout="",
                _full_output="",
            )

            # Run trigger tests
            trigger_tests = relevant_bug.trigger_tests
            for t in trigger_tests:
                single_test_result = _test(t=t)
                result.timeout = result.timeout or single_test_result.timeout
                result.passed = result.passed and single_test_result.passed
                result.failing_tests += single_test_result.failing_tests
                result._stderr += "\n" + single_test_result._stderr
                result._stdout += "\n" + single_test_result._stdout
                result._full_output += "\n" + single_test_result._full_output

                if not single_test_result.passed:
                    # Early return
                    return result

            # if not result.passed:
            #     # Early return
            #     return result

            # Run all tests
            return _test(t=None)

    def get_method_code_lines(
        self, file_path: str, method_path: str, strict=True
    ) -> List[Tuple[int, str]]:
        file_path = f"{self.__src_path}/{file_path}"
        return _get_java_method_code_lines(file_path, method_path, strict=strict)

    def get_test_method_code_lines(
        self, test_method_full_path: str, return_file_path: bool = False
    ) -> List[Tuple[int, str]]:
        test_method_full_path = test_method_full_path.replace("::", ".")
        test_method_fullp_parts = test_method_full_path.split(".")
        assert len(test_method_fullp_parts) >= 2
        file_path = (
            f"{self.__tests_src_path}/{'/'.join(test_method_fullp_parts[:-1])}.java"
        )
        rel_md_path = ".".join(test_method_fullp_parts[-2:])

        try:
            mds = _get_java_method_code_lines(file_path, rel_md_path, strict=False)
        except FileNotFoundError as ex:
            rsu._elog(f"File not found: {ex}")
            return None

        if len(mds) != 1:
            rsu._elog(f"Found {len(mds)} methods in {file_path}")
            rsu._elog(f"Method path: {test_method_full_path}")
            rsu._elog(f"Method code lines", mds)
        if len(mds) == 1:
            if return_file_path:
                return {"method": mds[0], "file_path": file_path}
            else:
                return mds[0]
        else:
            return None

    def apply_patch(self, file_path: str, method_path: str, patch: str):
        file_path = f"{self.__src_path}/{file_path}"

        class FilePatcher:
            def __init__(self, file_path, method_path, patch):
                self.file_path = file_path
                self.method_path = method_path
                self.patch = patch

            def __enter__(self):
                if _is_debug_mode():
                    rsu._ilog(f"Applying patch to: {self.file_path}")
                # Save the original file
                self.orig_file_content_lines, self.ori_file_encoding = (
                    rsu._read_text_file(fn=self.file_path, return_encoding=True)
                )
                self.orig_file_content_lines = self.orig_file_content_lines.splitlines()
                assert self.ori_file_encoding is not None
                # Get original method lines
                method_lines = _get_java_method_code_lines(
                    self.file_path, self.method_path, strict=True
                )
                # Construct patched file content
                self.patched_file_content = "\n".join(
                    [
                        *self.orig_file_content_lines[: method_lines[0][0] - 1],
                        self.patch.strip("\n"),
                        *self.orig_file_content_lines[(method_lines[-1][0] - 1) + 1 :],
                    ]
                )
                # Write patched file
                try:
                    with open(
                        self.file_path, "w", encoding=self.ori_file_encoding
                    ) as fp:
                        fp.write(self.patched_file_content)
                except UnicodeEncodeError as ex:
                    rsu._wlog(f"UnicodeEncodeError: {ex}, retry")
                    encodings = [
                        "utf-8",
                        "utf-16",
                        "iso-8859-1",
                        "windows-1252",
                        "ascii",
                        "latin-1",
                        "utf-8-sig",
                        "gbk",
                        "utf-16-le",
                        "utf-16-be",
                    ]
                    success, last_ex = False, None
                    for enc in encodings:
                        try:
                            with open(self.file_path, "w", encoding=enc) as fp:
                                fp.write(self.patched_file_content)
                            success = True
                            break
                        except UnicodeEncodeError as ex:
                            last_ex = ex
                            rsu._wlog(f"UnicodeEncodeError: {ex}, retry")
                    if not success:
                        raise last_ex
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                # Restore original file
                assert self.ori_file_encoding is not None
                with open(self.file_path, "w", encoding=self.ori_file_encoding) as fp:
                    fp.write("\n".join(self.orig_file_content_lines))
                if _is_debug_mode():
                    rsu._ilog(f"Restored file: {self.file_path}")
                return False

        return FilePatcher(file_path=file_path, method_path=method_path, patch=patch)


def get_defects4j_v1_2_0_projects(
    d4j_cli: str, d4j_bugs_stat_file: str
) -> List[Defects4JProject]:
    _projs = [
        ("Chart", 26),
        ("Closure", 133),
        ("Lang", 65),
        ("Math", 106),
        ("Mockito", 38),
        ("Time", 27),
    ]
    assert sum(bug_count for _, bug_count in _projs) == 395

    stat = rsu._load_json(d4j_bugs_stat_file)
    pid2stat = {}
    for s in stat:
        if s["pid"] not in pid2stat:
            pid2stat[s["pid"]] = []
        pid2stat[s["pid"]].append(s)

    return [
        Defects4JProject(
            project_name=proj_mame,
            bug_ids=list(range(1, bug_count + 1)),
            d4j_version="v1.2.0",
            d4j_cli=d4j_cli,
            d4j_bugs_stat=pid2stat[proj_mame],
        )
        for proj_mame, bug_count in _projs
    ]


def get_defects4j_v2_0_0_projects(
    d4j_cli: str, d4j_bugs_stat_file: str
) -> List[Defects4JProject]:
    _projs = [
        # Chart  # 1-26 (d4j v1.2.0)
        ("Cli", "1-5,7-40"),
        ("Closure", "134-176"),  # (1-62,64-92,94-133: d4j v1.2.0)
        ("Codec", "1-18"),
        ("Collections", "25-28"),
        ("Compress", "1-47"),
        ("Csv", "1-16"),
        ("Gson", "1-18"),
        ("JacksonCore", "1-26"),
        ("JacksonDatabind", "1-112"),
        ("JacksonXml", "1-6"),
        ("Jsoup", "1-93"),
        ("JxPath", "1-22"),
        # Lang  # 1,3-65 (d4j v1.2.0)
        # Math  # 1-106 (d4j v1.2.0)
        # Mockito  # 1-38 (d4j v1.2.0)
        # Time  # 1-20,22-27 (d4j v1.2.0)
    ]
    assert sum(len(rsu._parse_ints(bug_ids)) for _, bug_ids in _projs) == 444

    stat = rsu._load_json(d4j_bugs_stat_file)
    pid2stat = {}
    for s in stat:
        if s["pid"] not in pid2stat:
            pid2stat[s["pid"]] = []
        pid2stat[s["pid"]].append(s)

    return [
        Defects4JProject(
            project_name=proj_mame,
            bug_ids=rsu._parse_ints(bug_ids),
            d4j_version="v2.0.0",
            d4j_cli=d4j_cli,
            d4j_bugs_stat=pid2stat[proj_mame],
        )
        for proj_mame, bug_ids in _projs
    ]


class Vul4JProject(JavaProject):
    """
    Vul4J: https://github.com/tuhh-softsec/vul4j
        Note: Only support single-function bug
    """

    def __init__(
        self,
        project_name: str,
        *,
        # For the repo-like project, HEAD=None
        bug_ids: Optional[List[int]] = None,  # bids
        bug_infos: Optional[List[Dict[str, Any]]] = None,  # bid -> bug info
        # For the version-like project, HEAD=project_version
        version: Optional[str] = None,
        directory: Optional[str] = None,
        source_path: Optional[str] = None,
        tests_source_path: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ):
        if bug_ids is not None:  # The repo-like project, HEAD=None
            assert bug_ids is not None
            assert bug_infos is not None
            assert set(bug_ids) == set(int(i["bid"]) for i in bug_infos)
            assert len(bug_ids) == 1  # Vul4J do not group bugs by project
            assert version is None
            assert directory is None
            assert source_path is None
            assert tests_source_path is None
            assert info is None
        else:  # The version-like project, HEAD=project_version
            assert bug_ids is None
            assert bug_infos is None
            assert version is not None
            assert directory is not None
            assert source_path is not None
            assert tests_source_path is not None
            assert info is not None
            assert version == "0b"

        self.__project_name = project_name
        if bug_ids is not None:  # For the repo-like project, HEAD=None
            self.__version = None
            self.__bug_ids = bug_ids
            self.__bug_infos = {int(info["bid"]): info for info in bug_infos}
        else:  # For the version-like project, HEAD=project_version
            self.__version = version
            self.__directory = directory
            self.__source_path = f"{self.__directory}/{source_path}"
            self.__tests_source_path = f"{self.__directory}/{tests_source_path}"
            self.__info = info
            assert os.path.isdir(self.__directory)
            assert os.path.isdir(self.__source_path)
            assert os.path.isdir(self.__tests_source_path)

        self.__vul4j_cmd = f"vul4j"  # Find in PATH
        # NOTE: vul4j cli is not multi-process safe
        self.__vul4j_cli_lock = os.path.expanduser("~/.vul4j_cli.lock")

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"Vul4JProject({self.__project_name}@{self.__version})"

    def _vul4j_cli(
        self, *args, workdir: str = None, timeout: float = None
    ) -> Tuple[int, str]:
        cmd = f"{self.__vul4j_cmd} {' '.join(args)}"
        if _is_debug_mode():
            start_time = time.time()
            rsu._ilog(f"Run vul4j: `{cmd}`")
            rsu._ilog(f">>>> workdir: {workdir}")
            rsu._ilog(f">>>> start at: {start_time}")
        with rsu.FileLock(self.__vul4j_cli_lock, interval=0.5) as flock:
            if flock.locked:
                ret, stdout, stderr = rsu._sp_run(
                    cmd,
                    cwd=workdir,
                    timeout=timeout,
                )
        if _is_debug_mode():
            end_time = time.time()
            rsu._ilog(f">>>> end at: {end_time}")
            rsu._ilog(f">>>> duration: {end_time - start_time}s`")
            rsu._ilog(f">>>> return code: {ret}")
            rsu._ilog(f">>>> stdout: `{stdout}`")
            rsu._ilog(f">>>> stderr: `{stderr}`")
        return ret, stdout, stderr

    def get_java_home(self) -> str:
        raise NotImplementedError("Not supported yet")

    def get_java_version(self) -> str:
        raise NotImplementedError("Not supported yet")

    def get_src_sourcepath(self) -> str:
        assert self.__version is not None
        return self.__source_path

    def get_tests_sourcepath(self) -> str:
        assert self.__version is not None
        return self.__tests_source_path

    def get_test_classpath(self) -> str:
        assert self.__version is not None
        raise NotImplementedError("Not supported yet")

    def get_lang(self) -> str:
        return super().get_lang()

    def get_itendifier(self, ignore_project_version: bool = True) -> str:
        itendifier = f"vul4j-{self.__project_name}"
        if not ignore_project_version:
            itendifier += f"-{self.__version}"
        return itendifier

    def get_root_path(self) -> str:
        assert self.__version is not None
        return self.__directory

    def get_bugs(self, _filter=None) -> List[BugInfo]:
        _filter = _filter or (lambda _: True)
        assert self.__version is None
        assert callable(_filter)

        bug_ids = self.__bug_ids
        bugs_info = self.__bug_infos

        return list(
            filter(
                _filter,
                (
                    BugInfo(
                        project=self,
                        bug_locations=[
                            CodeLocation(
                                file_path=line["file"],
                                lineno=line["lineno"],
                                method_path=line["method"],
                            )
                            for line in bugs_info[bid]["buggy_lines"]
                        ],
                        num_buggy_lines=bugs_info[bid]["num_buggy_lines"],
                        is_single_line_bug=bugs_info[bid]["is_single_line_bug"],
                        is_single_method_bug=bugs_info[bid]["is_single_method_bug"],
                        is_single_file_bug=bugs_info[bid]["is_single_file_bug"],
                        all_tests=None,
                        relevant_tests=None,
                        trigger_tests=bugs_info[bid]["tests.trigger"],
                        _interal_id=f"{self.__project_name}_{bid}",
                    )
                    for bid in bug_ids
                ),
            )
        )

    def checkout(self, bug: BugInfo, b_mark: Literal["b", "f"]) -> "Project":
        assert self.__version is None
        assert b_mark in {"b", "f"}
        pid, bid = bug._interal_id.split("_")
        assert self.__project_name == pid

        if b_mark != "b":
            raise RuntimeError("Checkout fixed version is not supported yet")

        CHECKOUT_TMP_PATH = os.getenv("VUL4J_CHECKOUT_TMP_PATH", None)
        if CHECKOUT_TMP_PATH is None:
            raise RuntimeError(
                "Environment variable VUL4J_CHECKOUT_TMP_PATH is not set"
            )

        p_path = f"{CHECKOUT_TMP_PATH}/{pid}_{bid}{b_mark}"
        bug_info = self.__bug_infos[int(bid)]

        if os.path.exists(p_path):
            if _is_debug_mode():
                rsu._ilog(f"Removing old: {p_path}")
            shutil.rmtree(p_path)

        assert not os.path.exists(p_path)
        if _is_debug_mode():
            rsu._ilog(f"Checking out project {pid}_{bid} to: {p_path}")
        # vul4j checkout --id VUL4J-10 -d /tmp/vul4j/VUL4J-10
        ret, out, err = self._vul4j_cli("checkout", "--id", pid, "-d", p_path)
        if ret != 0:
            rsu._wlog(f"Checkout failed: out='{out}', err='{err}'")
            raise RuntimeError(f"Checkout failed: {pid}-{bid}{b_mark} {p_path}")

        # NOTE: Check git diff: no changes
        assert rsu._sp_run(f"git diff", cwd=p_path) == (0, "", "")

        if _is_debug_mode():
            rsu._ilog(f"Checked out project {pid}_{bid} to {p_path}")

        ###### HARD CODE [START] ######
        gradle_props_fn = f"{p_path}/gradle/wrapper/gradle-wrapper.properties"
        if os.path.isfile(gradle_props_fn):
            with open(gradle_props_fn, "r") as fp:
                gradle_props_content = fp.read()
            gradle_props_content = gradle_props_content.replace(
                "services.gradle.org/distributions/",  # Too slow
                "mirrors.cloud.tencent.com/gradle/",
            )
            with open(gradle_props_fn, "w") as fp:
                fp.write(gradle_props_content)
            if _is_debug_mode():
                rsu._ilog(f"Patched {gradle_props_fn}")
            del gradle_props_content
        del gradle_props_fn

        if bug.get_itendifier() in ["VUL4J-53_0", "VUL4J-55_0"]:
            bad_file = f"{p_path}/core/src/main/java/hudson/util/ProcessTree.java"
            assert os.path.isfile(bad_file)
            bad_code, encoding = rsu._read_text_file(bad_file, return_encoding=True)
            fixed_code = bad_code.replace(" _)", " _1)").replace(" _ ", " _1 ")
            assert bad_code != fixed_code
            with open(bad_file, "w", encoding=encoding) as fp:
                fp.write(fixed_code)
            del bad_code, encoding, fixed_code
        ###### HARD CODE [END] ######

        return Vul4JProject(
            project_name=self.__project_name,
            # For the version-like project, HEAD=project_version
            version=f"{bid}{b_mark}",
            directory=p_path,
            source_path=bug_info["dir.src.classes"],
            tests_source_path=bug_info["dir.src.tests"],
            info=bug_info.copy(),
        )

    def run_compile(self) -> Tuple[int, str, str]:
        useless_log = """\
[ERROR] -> [Help 1]
[ERROR] 
[ERROR] To see the full stack trace of the errors, re-run Maven with the -e switch.
[ERROR] Re-run Maven using the -X switch to enable full debug logging.
[ERROR] 
[ERROR] For more information about the errors and possible solutions, please read the following articles:
[ERROR] [Help 1] http://cwiki.apache.org/confluence/display/MAVEN/MojoFailureException""".strip()
        useless_log = "\n".join([L.rstrip() for L in useless_log.splitlines()]).strip()

        # vul4j compile -d /tmp/vul4j/VUL4J-10
        ret, _, _ = self._vul4j_cli("compile", "-d", self.get_root_path())
        compile_log_fn = f"{self.get_root_path()}/VUL4J/compile.log"
        assert os.path.isfile(compile_log_fn)
        compile_log = rsu._read_text_file(compile_log_fn)
        compile_log = "\n".join([L.rstrip() for L in compile_log.splitlines()]).strip()

        if ret == 0:
            return ret, compile_log, ""
        else:
            assert (
                "[ERROR] COMPILATION ERROR :" in compile_log
                or "BUILD FAILED" in compile_log
            )
            err_log = "\n".join(
                L
                for L in compile_log.splitlines()
                if L.strip().startswith("[ERROR]")  # Filter error log
            )
            err_log = err_log.replace(useless_log, "").strip()
            err_log = err_log.replace(self.get_root_path() + "/", "")
            return ret, "", err_log

    def run_test(
        self,
        relevant_bug: BugInfo = None,
        strategy: Literal["trigger->all", "all"] = "trigger->all",
        test_case_method_full_path: str = None,
    ) -> RunTestResult:
        if test_case_method_full_path and "::" not in test_case_method_full_path:
            raise ValueError("test method path should be: xx.xx.XxClass::xxMethod")

        def _parse_test_results(xml_files: List[str]) -> list:
            import re
            import xml.etree.ElementTree as ET

            failing_tests = []
            passed = True
            stack_pattern = re.compile(
                r"at\s+([\w.$]+)\.([\w$]+)\(([\w$]+\.java):(\d+)\)"
            )

            for xml_file in xml_files:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                for testcase in root.findall("testcase"):
                    classname = testcase.get("classname", "").strip()
                    name = testcase.get("name", "").strip()
                    error = testcase.find("error")
                    failure = testcase.find("failure")

                    if error is None and failure is None:
                        continue

                    passed = False

                    element = error if error is not None else failure
                    traceback = element.text.strip() if element.text else ""

                    line_raised_at = -1
                    file_raised_at = "unknown"
                    for line in traceback.split("\n"):
                        match = stack_pattern.match(line.strip())
                        if match:
                            trace_class = match.group(1).strip()
                            trace_method = match.group(2).strip()
                            if trace_class == classname and trace_method == name:
                                file_path = match.group(3)
                                file_raised_at = os.path.basename(file_path)
                                line_raised_at = int(match.group(4))
                                break

                    # NOTE: Hard code but enough
                    if " on instance " in name:
                        name_parts = name.split(" on instance ")
                        assert len(name_parts) == 2
                        assert name_parts[0].strip() == name_parts[1].strip()
                        name = name_parts[0].strip()
                    elif "[" in name:
                        assert name.strip()[-1] == "]"
                        name = name[: name.index("[")].strip()
                    elif name.strip().endswith("()"):
                        name = name.strip().removesuffix("()").strip()

                    md_path = f"{classname}::{name}"
                    failing_test = RunTestResult.FailingTestInfo(
                        md_path=md_path,
                        raised_exception=traceback.split("\n")[0].strip(),
                        line_raised_at=line_raised_at,
                        file_raised_at=file_raised_at,
                        traceback=traceback,
                    )
                    failing_tests.append(failing_test)

            dedup_failing_tests = []
            failing_test_keys = set()
            for t in failing_tests:
                if t.md_path not in failing_test_keys:
                    failing_test_keys.add(t.md_path)
                    dedup_failing_tests.append(t)
            failing_tests = sorted(dedup_failing_tests, key=lambda x: x.md_path)

            assert passed or len(failing_tests) > 0
            return failing_tests

        def _test(batch_type: str) -> RunTestResult:
            timeout_sec = 60 * 60  # 1h for all or povs
            try:
                # vul4j test -b all -d /tmp/vul4j/VUL4J-10
                ret, _, _ = self._vul4j_cli(
                    "test",
                    "-b",
                    batch_type,
                    "-d",
                    self.get_root_path(),
                    timeout=timeout_sec,
                )
                assert ret == 0
            except sp.TimeoutExpired:
                return RunTestResult(
                    timeout=True,
                    passed=False,
                    failing_tests=["<Timeout>"],
                    _stdout=None,
                    _stderr=None,
                    _full_output=None,
                )

            testing_result_jf = f"{self.get_root_path()}/VUL4J/testing_results.json"
            testing_result_j = rsu._load_json(testing_result_jf)
            passed = 0 == (
                testing_result_j["tests"]["overall_metrics"]["number_error"]
                + testing_result_j["tests"]["overall_metrics"]["number_failing"]
            )

            if passed:
                assert not testing_result_j["tests"]["failures"]
                if _is_debug_mode():
                    rsu._ilog(f"Passed testing of: {batch_type}")
                return RunTestResult(
                    timeout=False,
                    passed=True,
                    failing_tests=[],
                    _stdout=None,
                    _stderr=None,
                    _full_output=None,
                )

            def _find_test_reports(project_dir: str) -> List[str]:
                report_files = []
                for r, dirs, files in os.walk(project_dir):
                    for file in files:
                        file_path = os.path.join(r, file)
                        if (
                            (
                                "target/surefire-reports" in file_path
                                or "target/failsafe-reports" in file_path
                                or "build/test-results" in file_path  # gradle
                            )
                            and file.endswith(".xml")
                            and file.startswith("TEST-")
                        ):
                            report_files.append(file_path)
                return sorted(report_files)

            # Parse test results
            test_result_files = _find_test_reports(self.get_root_path())
            failing_tests = _parse_test_results(xml_files=test_result_files)
            assert len(test_result_files) > 0
            assert len(failing_tests) > 0

            if _is_debug_mode():
                rsu._ilog("Result of testing:")
                rsu._ilog(f">>>> num_failing_tests: {len(failing_tests)}")
                rsu._plog(f">>>> failing_tests", [t.md_path for t in failing_tests])

            return RunTestResult(
                timeout=False,
                passed=False,
                failing_tests=failing_tests,
                _stdout=None,
                _stderr=None,
                _full_output=None,
            )

        if test_case_method_full_path is not None:
            test_result = (
                _test(batch_type="povs")
                if test_case_method_full_path
                in (
                    relevant_bug
                    and relevant_bug.trigger_tests
                    or self.__info["tests.trigger"]
                )
                else _test(batch_type="all")
            )
            if test_result.timeout:
                return test_result
            elif test_result.passed:
                assert not test_result.failing_tests
                return test_result
            else:
                test_result_of_spec_t = next(
                    (
                        t
                        for t in test_result.failing_tests
                        if t.md_path == test_case_method_full_path
                    ),
                    None,
                )
                if test_result_of_spec_t is None:
                    return RunTestResult(
                        timeout=False,
                        passed=True,
                        failing_tests=[],
                        _stdout=None,
                        _stderr=None,
                        _full_output=None,
                    )
                else:
                    return RunTestResult(
                        timeout=False,
                        passed=False,
                        failing_tests=[test_result_of_spec_t],
                        _stdout=None,
                        _stderr=None,
                        _full_output=None,
                    )
        elif strategy == "all":
            return _test(batch_type="all")
        elif strategy == "trigger->all":
            assert relevant_bug is not None

            # Run PoV tests
            pov_t_result = _test(batch_type="povs")

            if not pov_t_result.passed:
                # Early return
                return pov_t_result

            # Run all tests
            assert pov_t_result.passed
            return _test(batch_type="all")

    def get_method_code_lines(
        self, file_path: str, method_path: str, strict: bool
    ) -> List[Tuple[int, str]]:
        file_path = f"{self.__source_path}/{file_path}"
        start_l, end_l = map(int, method_path.split("[")[-1].split("]")[0].split("-"))
        file_lines = rsu._read_text_file(file_path).splitlines()
        method_lines = file_lines[start_l - 1 : (end_l - 1) + 1]
        return [
            (lineno, line) for lineno, line in enumerate(method_lines, start=start_l)
        ]

    def get_test_method_code_lines(
        self, test_method_full_path: str, return_file_path: bool = False
    ) -> List[Tuple[int, str]]:
        test_method_full_path = test_method_full_path.replace("::", ".")
        test_method_fullp_parts = test_method_full_path.split(".")
        assert len(test_method_fullp_parts) >= 2
        file_path = f"{'/'.join(test_method_fullp_parts[:-1])}.java"
        file_path = f"{self.__tests_source_path}/{file_path}"
        rel_md_path = ".".join(test_method_fullp_parts[-2:])

        try:
            mds = _get_java_method_code_lines(file_path, rel_md_path, strict=False)
        except FileNotFoundError as ex:
            rsu._elog(f"File not found: {ex}")
            return None

        if len(mds) != 1:
            rsu._elog(f"[{self.__version}] Found {len(mds)} methods in {file_path}")
            rsu._elog(f"Method path: {test_method_full_path}")
            rsu._elog(f"Method code lines", mds)
        if len(mds) == 1:
            if return_file_path:
                return {"method": mds[0], "file_path": file_path}
            else:
                return mds[0]
        else:
            return None

    def apply_patch(self, file_path: str, method_path: str, patch: str):
        file_path = f"{self.__source_path}/{file_path}"
        _bug_info = self.__info

        class FilePatcher:
            def __init__(self, file_path, method_path, patch):
                self.file_path = file_path
                self.method_path = method_path
                self.patch = patch

            def __enter__(self):
                if _is_debug_mode():
                    rsu._ilog(f"Applying patch to: {self.file_path}")
                    rsu._ilog(f">>>> Bug Method: {self.method_path}")
                # Save the original file
                self.orig_file_content_lines, self.ori_file_encoding = (
                    rsu._read_text_file(fn=self.file_path, return_encoding=True)
                )
                self.orig_file_content_lines = self.orig_file_content_lines.splitlines()
                assert self.ori_file_encoding is not None
                # Get method start & end line
                assert "[" in self.method_path and "]" in self.method_path
                md_start_l, md_end_l = map(  # 1-based
                    int,
                    self.method_path.strip().split("[")[-1].split("]")[0].split("-"),
                )
                # Construct patched file content
                patched_file_content_lines = [
                    *self.orig_file_content_lines[: md_start_l - 1],
                    self.patch.strip("\n"),
                    *self.orig_file_content_lines[(md_end_l - 1) + 1 :],
                ]
                if addi_imports := _bug_info.get("addi_imports_in_fixed_version", None):
                    assert isinstance(addi_imports, str)
                    pkg_del_L_idx = next(
                        i
                        for i, line in enumerate(patched_file_content_lines)
                        if line.strip().startswith("package ")
                    )
                    pkg_del_L = patched_file_content_lines[pkg_del_L_idx]
                    pkg_del_next_L = patched_file_content_lines[pkg_del_L_idx + 1]
                    patched_file_content_lines = [
                        # ... package xxx;
                        *patched_file_content_lines[: pkg_del_L_idx + 1],
                        # import xxx; ... import zzz;
                        addi_imports,
                        # ...
                        *patched_file_content_lines[pkg_del_L_idx + 1 :],
                    ]
                    if _is_debug_mode():
                        rsu._ilog(f"Inserted addi_imports: {repr(addi_imports)}")
                        rsu._ilog(f">>>> Before: {repr(pkg_del_L)}")
                        rsu._ilog(f">>>> After: {repr(pkg_del_next_L)}")
                self.patched_file_content = "\n".join(patched_file_content_lines)
                # Write patched file
                try:
                    with open(
                        file=self.file_path,
                        mode="w",
                        encoding=self.ori_file_encoding,
                    ) as fp:
                        fp.write(self.patched_file_content)
                except UnicodeEncodeError as ex:
                    rsu._wlog(f"UnicodeEncodeError: {ex}, retry")
                    encodings = [
                        "utf-8",
                        "utf-16",
                        "iso-8859-1",
                        "windows-1252",
                        "ascii",
                        "latin-1",
                        "utf-8-sig",
                        "gbk",
                        "utf-16-le",
                        "utf-16-be",
                    ]
                    success, last_ex = False, None
                    for enc in encodings:
                        try:
                            with open(self.file_path, "w", encoding=enc) as fp:
                                fp.write(self.patched_file_content)
                            success = True
                            break
                        except UnicodeEncodeError as ex:
                            last_ex = ex
                            rsu._wlog(f"UnicodeEncodeError: {ex}, retry")
                    if not success:
                        raise last_ex
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                # Restore original file
                assert self.ori_file_encoding is not None
                with open(self.file_path, "w", encoding=self.ori_file_encoding) as fp:
                    fp.write("\n".join(self.orig_file_content_lines))
                if _is_debug_mode():
                    rsu._ilog(f"Restored file: {self.file_path}")
                return False

        return FilePatcher(file_path=file_path, method_path=method_path, patch=patch)


def get_vul4j_projects(bug_info_file: str) -> List[Vul4JProject]:
    if not os.path.exists(bug_info_file) or not bug_info_file.endswith(".json"):
        raise ValueError(f"Invalid bug_info_file: {bug_info_file}")

    bug_info_file = os.path.abspath(bug_info_file)
    bug_info = rsu._load_json(bug_info_file)
    proj2bugs = {}
    for b in bug_info:
        if b["pid"] not in proj2bugs:
            proj2bugs[b["pid"]] = []
        proj2bugs[b["pid"]].append(b)

    return [
        Vul4JProject(
            project_name=proj_name,
            # For the repo-like project, HEAD=None
            bug_ids=[0],
            bug_infos=bugs,
        )
        for proj_name, bugs in sorted(
            # proj_name: VUL4J-{id: int}
            proj2bugs.items(),
            key=lambda x: int(x[0].split("-")[-1]),
        )
    ]
