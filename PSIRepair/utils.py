import os
import io
import re
import javalang
from collections import namedtuple
from typing import List, Tuple, Dict, Union, Optional, Literal
from . import rs_utils as rsu

_COMMENT_MARK_TABLE = {"c": "//", "cpp": "//", "java": "//", "python": "#"}
_TextLines = List[List[Tuple[int, str]]]
_Codeblock = namedtuple("Codeblock", ["lang", "content", "start_index", "end_index"])


def _get_comment_mark(lang: str) -> str:
    lang = lang.lower()
    assert lang in _COMMENT_MARK_TABLE
    return _COMMENT_MARK_TABLE[lang]


def _get_text_range_lines(
    file_path: str, start_position, end_position, file_content=None
) -> _TextLines:
    """
    [start, end]
    """
    if isinstance(start_position, int):
        start_position = javalang.tokenizer.Position(line=start_position, column=None)
    if isinstance(end_position, int):
        end_position = javalang.tokenizer.Position(line=end_position, column=None)

    assert file_path or file_content
    # TODO: Consiter *_position.column if needed
    file_content = (file_content or rsu._read_text_file(file_path)).splitlines()[
        start_position.line - 1 : (end_position.line - 1) + 1
    ]
    lines = []
    for lineno, line in enumerate(
        file_content,
        start=start_position.line,
    ):
        lines.append((lineno, line))
    assert len(lines) == end_position.line - start_position.line + 1
    assert (
        not lines
        or lines[0][0] == start_position.line
        and lines[-1][0] == end_position.line
    )
    return lines


def _patch_javalang(javalang):
    @javalang.parser.parse_debug
    def parse_member_declaration(self):
        modifiers, annotations, javadoc = self.parse_modifiers()
        member = None

        token = self.tokens.look()
        if self.try_accept("void"):
            method_name = self.parse_identifier()
            member = self.parse_void_method_declarator_rest()
            member.name = method_name

        elif token.value == "<":
            member = self.parse_generic_method_or_constructor_declaration()

        elif token.value == "class":
            member = self.parse_normal_class_declaration()

        elif token.value == "enum":
            member = self.parse_enum_declaration()

        elif token.value == "interface":
            member = self.parse_normal_interface_declaration()

        elif self.is_annotation_declaration():
            member = self.parse_annotation_type_declaration()

        elif self.would_accept(javalang.tokenizer.Identifier, "("):
            constructor_name = self.parse_identifier()
            member = self.parse_constructor_declarator_rest()
            member.name = constructor_name

        else:
            member = self.parse_method_or_field_declaraction()

        member._position = token.position
        member._pg_start_position = token.position
        member._pg_end_position = self.tokens.list[self.tokens.marker - 1].position
        member.modifiers = modifiers
        member.annotations = annotations
        member.documentation = javadoc

        return member

    @javalang.parser.parse_debug
    def parse_generic_method_or_constructor_declaration(self):
        type_parameters = self.parse_type_parameters()
        method = None

        token = self.tokens.look()
        if self.would_accept(javalang.tokenizer.Identifier, "("):
            constructor_name = self.parse_identifier()
            method = self.parse_constructor_declarator_rest()
            method.name = constructor_name
        elif self.try_accept("void"):
            method_name = self.parse_identifier()
            method = self.parse_void_method_declarator_rest()
            method.name = method_name

        else:
            method_return_type = self.parse_type()
            method_name = self.parse_identifier()

            method = self.parse_method_declarator_rest()

            method_return_type.dimensions += method.return_type.dimensions
            method.return_type = method_return_type
            method.name = method_name

        method._position = token.position
        method._pg_start_position = token.position
        method._pg_end_position = self.tokens.list[self.tokens.marker - 1].position
        method.type_parameters = type_parameters
        return method

    @javalang.parser.parse_debug
    def parse_interface_member_declaration(self):
        declaration = None

        token = self.tokens.look()
        if self.would_accept("class"):
            declaration = self.parse_normal_class_declaration()
        elif self.would_accept("interface"):
            declaration = self.parse_normal_interface_declaration()
        elif self.would_accept("enum"):
            declaration = self.parse_enum_declaration()
        elif self.is_annotation_declaration():
            declaration = self.parse_annotation_type_declaration()
        elif self.would_accept("<"):
            declaration = self.parse_interface_generic_method_declarator()
        elif self.try_accept("void"):
            method_name = self.parse_identifier()
            declaration = self.parse_void_interface_method_declarator_rest()
            declaration.name = method_name
        else:
            declaration = self.parse_interface_method_or_field_declaration()

        declaration._position = token.position
        declaration._pg_start_position = token.position
        declaration._pg_end_position = self.tokens.list[self.tokens.marker - 1].position

        return declaration

    javalang.parser.Parser.parse_member_declaration = parse_member_declaration
    javalang.parser.Parser.parse_generic_method_or_constructor_declaration = (
        parse_generic_method_or_constructor_declaration
    )
    javalang.parser.Parser.parse_interface_member_declaration = (
        parse_interface_member_declaration
    )


_patch_javalang(javalang)  # Patch `javalang` to support end positions for members


def _get_java_method_code_lines(
    file_path: str, method_path: str, strict: bool
) -> Union[List[_TextLines], _TextLines]:
    # file_path: "/path/to/some.java"
    # method_path: "...someMethod[startline:endline]" or "...someMethod"

    def _get_path_string(path):
        return ".".join(
            map(
                lambda x: x.name,
                filter(
                    lambda x: isinstance(
                        x,
                        (
                            javalang.parser.tree.MethodDeclaration,
                            javalang.parser.tree.ConstructorDeclaration,
                            javalang.parser.tree.ClassDeclaration,
                        ),
                    ),
                    path,
                ),
            )
        )

    method = []
    file_content = rsu._read_text_file(file_path)
    tree = javalang.parse.parse(file_content)
    for path, node in tree:
        if isinstance(
            node,
            (
                javalang.parser.tree.MethodDeclaration,
                javalang.parser.tree.ConstructorDeclaration,
            ),
        ):
            if strict:
                if (
                    f"{_get_path_string(path)}.{node.name}[{node._pg_start_position.line}-{node._pg_end_position.line}]"
                    == method_path
                ):
                    method.append(node)
            else:
                try:
                    left_idx = method_path.index("[")
                    method_path = method_path[:left_idx]
                except ValueError:
                    pass
                if f"{_get_path_string(path)}.{node.name}" == method_path:
                    method.append(node)

    if strict:
        assert len(method) == 1
        method = method[0]
        return _get_text_range_lines(
            file_path=file_path,
            start_position=method._pg_start_position,
            end_position=method._pg_end_position,
        )
    else:
        return [
            _get_text_range_lines(
                file_path=file_path,
                start_position=md._pg_start_position,
                end_position=md._pg_end_position,
            )
            for md in method
        ]


def _is_debug_mode() -> bool:
    return os.environ.get("DEBUG", "0") == "1"


def _is_single_method_bug(bug) -> bool:
    return bug.is_single_method_bug and bug.bug_locations[0].method_path is not None


def _make_func_body_with_bug_location(
    bug,
    with_line_numbers=False,
    additional_comment_lines: Optional[Dict[int, str]] = None,
    with_bug_location=True,
) -> str:
    def _get_addi_comment(lineno: int) -> Optional[str]:
        if additional_comment_lines is None:
            return None
        return additional_comment_lines.get(lineno, None)

    if os.getenv("FL_NO_BUG_LINE", "0") == "1":
        rsu._wlog("!!!FL_NO_BUG_LINE: is set!!!")
        if with_bug_location:
            rsu._wlog("!!!Enforce with_bug_location to be False!!!")
            with_bug_location = False

    project = bug.project
    file_path = bug.bug_locations[0].file_path
    method_path = bug.bug_locations[0].method_path
    comment_mark = _get_comment_mark(project.get_lang())
    bug_linenos = [b.lineno for b in bug.bug_locations]
    bug_md_code_lines = project.get_method_code_lines(
        file_path, method_path, strict=True
    )
    result_sb = io.StringIO()
    for lineno, line in bug_md_code_lines:
        # Add line number
        if with_line_numbers:
            result_sb.write(f"{lineno}: ")
        # Add code Line
        result_sb.write(line.rstrip())
        # Add bug location comment
        if with_bug_location and lineno in bug_linenos:
            result_sb.write(f"  {comment_mark} bug line")
        # Add additional comment
        addi_comment = _get_addi_comment(lineno)
        if addi_comment is not None:
            result_sb.write(f"  {comment_mark} {addi_comment}")
        # Add newline
        result_sb.write("\n")
    return result_sb.getvalue()


def _get_test_method_info(
    bug,
    trigger_test_index=0,
    with_line_numbers=False,
    additional_comment_lines: Optional[Dict[int, str]] = None,
) -> Optional[Dict[Literal["method", "method_path", "file_path"], str]]:
    def _get_addi_comment(lineno: int) -> Optional[str]:
        if additional_comment_lines is None:
            return None
        return additional_comment_lines.get(lineno, None)

    project = bug.project
    comment_mark = _get_comment_mark(project.get_lang())
    test_md_path = bug.trigger_tests[trigger_test_index]
    ret = project.get_test_method_code_lines(
        test_method_full_path=test_md_path,
        return_file_path=True,
    )
    if ret is None:
        return None

    result_sb = io.StringIO()
    for lineno, line in ret["method"]:
        # Add line number
        if with_line_numbers:
            result_sb.write(f"{lineno}: ")
        # Add code Line
        result_sb.write(line.rstrip())
        # Add additional comment
        addi_comment = _get_addi_comment(lineno)
        if addi_comment is not None:
            result_sb.write(f"  {comment_mark} {addi_comment}")
        # Add newline
        result_sb.write("\n")
    return {
        "method": result_sb.getvalue(),
        "method_path": test_md_path,
        "file_path": ret["file_path"],
    }


def _add_code_block(string: str, lang: str) -> str:
    assert string is not None
    assert lang is not None
    return f"```{lang}\n{string}\n```"


def _parse_code_block(string: str, lang: str, strict: bool = True) -> str:
    code_pattern = rf"```{lang}\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)

    if match:
        return match.group(1)

    generic_code_pattern = r"```\n(.*?)\n```"
    match = re.search(generic_code_pattern, string, re.DOTALL)

    if match:
        return match.group(1)

    if not strict:
        return string

    raise ValueError(f"No code block found: `{string}`")


def _parse_multi_code_blocks(string: str) -> List[_Codeblock]:
    pattern = re.compile(r"\n\s*```([^\n]*)\n(.*?)\n```", re.DOTALL)
    return [
        _Codeblock(
            lang=match.group(1).strip(),
            content=match.group(2),
            start_index=match.start(),  # NOTE: has leading spaces
            end_index=match.end(),
        )
        for match in pattern.finditer(string)
    ]


def _parse_java_multiline_comment(string: str, strict: bool = False) -> str:
    pattern = r"/\*(.*?)\*/"
    match = re.search(pattern, string, re.DOTALL)
    if match:
        return f"/*{match.group(1)}*/"
    if strict:
        raise ValueError(f"No comment found: `{string}`")
    return string


def _get_symbol_definition_content(file_text, symbol_name, start, end) -> _TextLines:
    """
    NOTE: start and end are 0-based, returns are 1-based
    """
    symbol_start_line = start["line"] + 1
    symbol_end_line = end["line"] + 1
    tree = javalang.parse.parse(file_text)
    nodes = []
    for path, node in tree:
        if isinstance(node, javalang.parser.tree.Declaration):
            if hasattr(node, "_pg_start_position") and hasattr(
                node, "_pg_end_position"
            ):
                if not (
                    symbol_start_line >= node._pg_start_position.line
                    and symbol_end_line <= node._pg_end_position.line
                ):
                    continue

            if node.position is not None:
                if not (node.position.line == symbol_start_line):
                    continue

            if hasattr(node, "name"):
                if not (node.name == symbol_name):
                    continue

            nodes.append((path, node))

    if nodes:
        max_path_node = max(nodes, key=lambda x: len(x[0]))[1]
        if hasattr(max_path_node, "_pg_start_position") and hasattr(
            max_path_node, "_pg_end_position"
        ):
            start_line = max_path_node._pg_start_position.line
            end_line = max_path_node._pg_end_position.line
            # print("XXXX", start_line, end_line)
            # print(max_path_node)
            return _get_text_range_lines(
                file_path=None,
                start_position=javalang.tokenizer.Position(
                    line=start_line, column=None
                ),
                end_position=javalang.tokenizer.Position(line=end_line, column=None),
                file_content=file_text,
            )
        else:
            start_line = max_path_node.position.line
            return [(start_line, file_text.splitlines()[start_line - 1])]
    else:
        start_line = start["line"] + 1
        return [(start_line, file_text.splitlines()[start_line - 1])]


class SkipException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _make_apr_task_input(
    bug,
    with_line_numbers: bool,
    inline_call_chain_from_test_to_junit_assertions: bool = False,
    simple_style: bool = False,
    only_include_bug_method: bool = False,
    only_return_bug_method: bool = False,
    only_return_failing_message: bool = False,
) -> str:

    def _ensure_compiled() -> bool:
        ret, stdout, stderr = bug.project.run_compile()
        if ret != 0:
            rsu._elog(f"Failed to compile, error:\n`{stderr}`")
            if (
                "Could not GET 'https://jcenter.bintray.com/com/googlecode/json-simple/json-simple/1.1.1/json-simple-1.1.1.pom'"
                in stderr
            ):
                raise SkipException("The project is bad (failed to compile), skip it")
            raise RuntimeError(f"Failed to compile, error:\n`{stderr}`")

    def _clip_traceback(md_path: str, full_traceback: str) -> str:
        _md_path = md_path.replace("::", ".")
        # Remove the last unnecessary lines
        result_sb = io.StringIO()
        for line in full_traceback.splitlines():
            result_sb.write(f"{line}\n")
            if _md_path in line:
                break
        return result_sb.getvalue()

    assert bug.project.get_lang() == "java", "PSIRepair only support Java now"
    assert bug.is_single_method_bug, "PSIRepair only support single method bug"
    assert bug.is_single_file_bug, "PSIRepair only support single file bug"
    assert len(set(f"{i.file_path}:{i.method_path}" for i in bug.bug_locations)) == 1
    assert (
        set((i.file_path, i.method_path) for i in bug.bug_locations).pop()[1]
        is not None
    )

    def _make_result(
        with_line_number_prompt: str,
        bug_md_path: str,
        rel_bug_file_path: str,
        lang: str,
        bug_md_with_linenos: str,
        test_md_info: str,
        failing_message: str,
    ) -> str:
        if not simple_style:
            result = f"""\
Buggy Method {with_line_number_prompt}(the bug lines are marked with '// bug line', and the optional exceptions are marked with '// Exception: ...'):
- Method Path: {bug_md_path}
- File Path: {rel_bug_file_path}
- Method Body:
```{lang}
{bug_md_with_linenos}
```
{test_md_info}
{failing_message}
"""
        else:
            result = f"""\
Buggy Method (the bug lines are marked with '// bug line', and the optional exceptions are marked with '// Exception: ...'):
```{lang}
// method path: {bug_md_path.replace('::', '.')}
{bug_md_with_linenos}
```
{test_md_info}
{failing_message}
"""

        if os.getenv("FL_NO_BUG_LINE", "0") == "1":
            rsu._wlog("!!!FL_NO_BUG_LINE: is set!!!")
            result = result.replace("the bug lines are marked with '// bug line', and ", "")

        return result

    def _make_test_md_info(
        with_line_number_prompt: str,
        test_md_path: str,
        rel_test_file_path: str,
        lang: str,
        test_md_with_linenos: str,
    ) -> str:
        if not simple_style:
            return f"""\
    
Triggered Failed Test Method {with_line_number_prompt}(the optional exceptions are marked with '// Exception:...'):
- Test Method Path: {test_md_path}
- Test File Path: {rel_test_file_path}
- Test Method Body:
```{lang}
{test_md_with_linenos}
```"""
        else:
            return f"""\

Triggered Failed Test Method (the optional exceptions are marked with '// Exception:...'):
```{lang}
// method path: {test_md_path.replace('::', '.')}
{test_md_with_linenos}
```"""

    if with_line_numbers:
        with_line_number_prompt = "with line numbers"
    else:
        with_line_number_prompt = ""

    # Prepare test result
    _ensure_compiled()
    trigger_test_index = 0  # Just select the first one NOW
    selected_test_md_path = bug.trigger_tests[trigger_test_index]
    test_result = bug.project.run_test(test_case_method_full_path=selected_test_md_path)
    assert not test_result.timeout
    assert (
        len(test_result.failing_tests) <= 1
        or len(set(t.md_path.split("(")[0].strip() for t in test_result.failing_tests))
        <= 1
    )
    if len(test_result.failing_tests) == 0:
        rsu._elog("Found a test method that does not fail")
        rsu._elog(f">>>> Bug: {bug.get_itendifier()}")
        rsu._elog(f">>>> Test: {selected_test_md_path}")
        rsu._elog(f">>>> Timeout: {test_result.timeout}")
        additional_comment_lines_for_bug_md = None
        additional_comment_lines_for_test_md = None
        failing_message = ""
        raise RuntimeError("Found a test method that does not fail")
    else:
        assert (
            len(set(t.md_path.split("(")[0].strip() for t in test_result.failing_tests))
            == 1
        )
        assert (
            test_result.failing_tests[0].md_path.split("(")[0].strip()
            == selected_test_md_path
        )
        line_raised_at = test_result.failing_tests[0].line_raised_at
        raised_exception = test_result.failing_tests[0].raised_exception
        traceback = test_result.failing_tests[0].traceback
        # Make addi comments for bug method
        additional_comment_lines_for_bug_md = {}
        ## at com.google.javascript.jscomp.TypeCheckTest.testTypes(TypeCheckTest.java:9816)
        bug_md_path = bug.bug_locations[0].method_path
        bug_md_start_l, bug_md_end_l = map(
            int, bug_md_path.split("[")[-1].split("]")[0].split("-")
        )
        bug_file_path = bug.bug_locations[0].file_path
        bug_md_full_path = bug_file_path.replace("/", ".").replace(".java", "")
        bug_md_full_path += "." + ".".join(bug_md_path.split("[")[0].split(".")[1:])
        possible_traceback_lines = [
            L
            for L in traceback.splitlines()
            if L.strip().startswith(f"at {bug_md_full_path}")
        ]
        possible_traceback_lines = [
            int(L[L.rfind("(") + 1 : -1].split(":")[1])
            for L in possible_traceback_lines
            if bug_md_start_l
            <= int(L[L.rfind("(") + 1 : -1].split(":")[1])
            <= bug_md_end_l
        ]
        if len(possible_traceback_lines) == 1:
            additional_comment_lines_for_bug_md[possible_traceback_lines[0]] = (
                f"Exception: {raised_exception}"
            )
        # Make addi comments for test method
        additional_comment_lines_for_test_md = {}
        if line_raised_at != -1:
            additional_comment_lines_for_test_md[line_raised_at] = (
                f"Exception: {raised_exception}"
            )

        # Make failing message
        failing_message = f"""\

Full traceback of the triggered failed test method:
{_clip_traceback(selected_test_md_path, traceback)}
"""

        # NOTE: I know this is ugly, but the time is limited, I will not fix.
        if only_return_failing_message:
            return _clip_traceback(selected_test_md_path, traceback)

        del (
            line_raised_at,
            raised_exception,
            traceback,
            bug_md_path,
            bug_md_start_l,
            bug_md_end_l,
            bug_file_path,
            bug_md_full_path,
            possible_traceback_lines,
        )

    # Prepare bug method
    src_sourcepath = os.path.relpath(
        bug.project.get_src_sourcepath(),
        start=bug.project.get_root_path(),
    )
    bug_file_path = bug.bug_locations[0].file_path
    rel_bug_file_path = f"{src_sourcepath}/{bug_file_path}"
    bug_md_path = bug.bug_locations[0].method_path
    bug_md_full_path = bug_file_path.replace("/", ".").replace(".java", "")
    bug_md_full_path += "::" + ".".join(bug_md_path.split("[")[0].split(".")[1:])
    bug_md_with_linenos = _make_func_body_with_bug_location(
        bug=bug,
        with_line_numbers=with_line_numbers,
        additional_comment_lines=additional_comment_lines_for_bug_md,
    )
    if only_return_bug_method:
        return f"""\
// method path: {bug_md_full_path.replace('::', '.')}
{bug_md_with_linenos}"""

    # Prepare test method
    test_md_info = _get_test_method_info(
        bug=bug,
        trigger_test_index=trigger_test_index,
        with_line_numbers=with_line_numbers,
        additional_comment_lines=additional_comment_lines_for_test_md,
    )
    if test_md_info is None:
        test_md_info = ""
    else:
        test_md_path = test_md_info["method_path"]
        abs_test_file_path = test_md_info["file_path"]
        rel_test_file_path = os.path.relpath(
            abs_test_file_path,
            start=bug.project.get_root_path(),
        )
        if inline_call_chain_from_test_to_junit_assertions:
            from .tools.inline_call_chain import _inline_call_chain

            if with_line_numbers:
                rsu._elog("inline... && with_line_numbers is not implemented")
                raise NotImplementedError("inline && w/ lineno is not implemented")

            inline_r = _inline_call_chain(
                source_roots=[
                    bug.project.get_src_sourcepath(),
                    bug.project.get_test_sourcepath(),
                ],
                class_roots=bug.project.get_test_classpath().split(":"),
                f_path=test_md_path.replace("::", "#"),
                # p_path_or_names=_get_junit_assertion_api_paths(),
                p_path_or_names=_get_junit_assertion_api_names(),
            )

            original_md = inline_r["original_method"]
            inlined_md = inline_r["inlined_method"]
            processed_inlined_md = inline_r["processed_inlined_method"]

            if os.getenv("SKIP_IF_MD_IS_SAME_AS_INLINED", "0") == "1":
                if original_md.strip() == inlined_md.strip():
                    raise SkipException(
                        "The inlined method is the same as the original method, skip it"
                    )

            test_md_with_linenos = processed_inlined_md

        else:
            test_md_with_linenos = test_md_info["method"]
        test_md_info = _make_test_md_info(
            lang=bug.project.get_lang(),
            test_md_path=test_md_path,
            rel_test_file_path=rel_test_file_path,
            test_md_with_linenos=test_md_with_linenos,
            with_line_number_prompt=with_line_number_prompt,
        )

    assert not only_return_failing_message

    if only_include_bug_method:
        test_md_info = ""
        failing_message = ""

    return _make_result(
        lang=bug.project.get_lang(),
        bug_md_path=bug_md_full_path,
        rel_bug_file_path=rel_bug_file_path,
        bug_md_with_linenos=bug_md_with_linenos,
        test_md_info=test_md_info,
        failing_message=failing_message,
        with_line_number_prompt=with_line_number_prompt,
    ).strip()


def _get_java_func_signature(full_method_code: str) -> str:
    full_method_code = full_method_code.strip()
    if not full_method_code:
        raise ValueError("Empty method code")
    if full_method_code.startswith("//") or full_method_code.startswith("/*"):
        raise ValueError("Do not support method with comment")
    try:
        tree = javalang.parse.parse(f"class ABC {{ {full_method_code}\n }}")
        assert len(tree.types) == 1 and tree.types[0].name == "ABC"
        if len(tree.types[0].methods) + len(tree.types[0].constructors) != 1:
            raise ValueError(
                f"Invalid method code, has multiple methods or no method: `{full_method_code}`"
            )
        if len(tree.types[0].fields) != 0:
            raise ValueError(
                f"Invalid method code, has fields but no method: `{full_method_code}`"
            )
        if len(tree.types[0].methods) == 1 and tree.types[0].methods[0].body is None:
            raise ValueError(f"Invalid method code, no body: `{full_method_code}`")
        is_method = len(tree.types[0].methods) == 1
        is_constructor = len(tree.types[0].constructors) == 1
        assert int(is_method) + int(is_constructor) == 1
    except javalang.parser.JavaSyntaxError as e:
        raise ValueError(
            f"Invalid method code, syntax error: `{full_method_code}`"
        ) from e

    sig = full_method_code[: full_method_code.index("{")].strip()
    try:  # Check syntax correctness
        if is_method:
            javalang.parse.parse(f"class ABC {{ {sig}; }}")
        elif is_constructor:
            javalang.parse.parse(f"class ABC {{ {sig} {{ }} }}")
    except javalang.parser.JavaSyntaxError as ex:
        raise ValueError(f"Invalid got method signature: `{sig}`") from ex
    return sig


def _get_simple_signature_for_java_method(
    method: str,
    include_return_type: bool = False,
) -> str:
    try:
        dummy_class_tokens = list(javalang.parse.tokenize("class ABC { }"))
        method_tokens = list(javalang.parse.tokenize(method))
        left_brace_index = next(  # the first {
            index_of_token
            for index_of_token, tk in enumerate(method_tokens)
            if isinstance(tk, javalang.tokenizer.Separator) and tk.value == "{"
        )
        right_brace_index = next(  # the last }
            index_of_token
            for index_of_token, tk in reversed(list(enumerate(method_tokens)))
            if isinstance(tk, javalang.tokenizer.Separator) and tk.value == "}"
        )
        method_tokens[left_brace_index + 1 : right_brace_index] = []
        tokens = [*dummy_class_tokens[:3], *method_tokens, *dummy_class_tokens[3:]]
        tree = javalang.parse.Parser(tokens).parse()
        assert len(tree.types) == 1 and tree.types[0].name == "ABC"
        if len(tree.types[0].methods) + len(tree.types[0].constructors) != 1:
            raise ValueError(
                f"Invalid method code, has multiple methods or no method: `{method}`"
            )
        if len(tree.types[0].fields) != 0:
            raise ValueError(
                f"Invalid method code, has fields but no method: `{method}`"
            )
        if len(tree.types[0].methods) == 1 and tree.types[0].methods[0].body is None:
            raise ValueError(f"Invalid method code, no body: `{method}`")
        is_method = len(tree.types[0].methods) == 1
        is_constructor = len(tree.types[0].constructors) == 1
        assert int(is_method) + int(is_constructor) == 1
    except (javalang.parser.JavaSyntaxError, StopIteration) as e:
        raise ValueError(f"Invalid method code, syntax error: `{method}`") from e

    def _get_param_type_name(param):
        name = param.type.name
        if param.type.dimensions:
            name += "[]" * len(param.type.dimensions)
        if param.varargs:
            name += "[]"
        return name

    def _get_ret_type_name(return_type):
        if return_type is None:
            return "void"
        name = return_type.name
        if return_type.dimensions:
            name += "[]" * len(return_type.dimensions)
        return name

    md = [*tree.types[0].methods, *tree.types[0].constructors][0]
    md_sign_parameters = ",".join(_get_param_type_name(p) for p in md.parameters)
    if include_return_type:
        md_sign_ret = (
            hasattr(md.return_type, "return_type")
            and _get_ret_type_name(md.return_type)
            or ""
        )
        return f"{md_sign_ret} {md.name}({md_sign_parameters})"
    return f"{md.name}({md_sign_parameters})"


def _replace_java_method_body(method: str, replacement: str) -> str:
    PLACEHOLDER_STMT = "__this_is_placeholder__ = __this_is_also_placeholder__;"
    placeholder_tokens = list(javalang.parse.tokenize(PLACEHOLDER_STMT))
    method_tokens = list(javalang.parse.tokenize(method))
    left_brace_index = next(  # the first {
        index_of_token
        for index_of_token, token in enumerate(method_tokens)
        if isinstance(token, javalang.tokenizer.Separator) and token.value == "{"
    )
    right_brace_index = next(  # the last }
        index_of_token
        for index_of_token, token in reversed(list(enumerate(method_tokens)))
        if isinstance(token, javalang.tokenizer.Separator) and token.value == "}"
    )
    method_tokens[left_brace_index + 1 : right_brace_index] = list(placeholder_tokens)
    return javalang.tokenizer.reformat_tokens(method_tokens).replace(
        PLACEHOLDER_STMT, replacement.strip()
    )


def _get_junit_assertion_api_paths() -> List[str]:
    return [
        # Junit 3
        "junit.framework.Assert#assertEquals",
        "junit.framework.Assert#assertFalse",
        "junit.framework.Assert#assertNotNull",
        "junit.framework.Assert#assertNotSame",
        "junit.framework.Assert#assertNull",
        "junit.framework.Assert#assertSame",
        "junit.framework.Assert#assertTrue",
        "junit.framework.Assert#fail",
        # Junit 4
        "org.junit.Assert#assertArrayEquals",
        "org.junit.Assert#assertEquals",
        "org.junit.Assert#assertFalse",
        "org.junit.Assert#assertNotEquals",
        "org.junit.Assert#assertNotNull",
        "org.junit.Assert#assertNotSame",
        "org.junit.Assert#assertNull",
        "org.junit.Assert#assertSame",
        "org.junit.Assert#assertThat",
        "org.junit.Assert#assertThrows",
        "org.junit.Assert#assertTrue",
        "org.junit.Assert#fail",
        # Junit 5
        "org.junit.jupiter.api.Assertions#assertAll",
        "org.junit.jupiter.api.Assertions#assertArrayEquals",
        "org.junit.jupiter.api.Assertions#assertEquals",
        "org.junit.jupiter.api.Assertions#assertFalse",
        "org.junit.jupiter.api.Assertions#assertIterableEquals",
        "org.junit.jupiter.api.Assertions#assertLinesMatch",
        "org.junit.jupiter.api.Assertions#assertNotEquals",
        "org.junit.jupiter.api.Assertions#assertNotNull",
        "org.junit.jupiter.api.Assertions#assertNotSame",
        "org.junit.jupiter.api.Assertions#assertNull",
        "org.junit.jupiter.api.Assertions#assertSame",
        "org.junit.jupiter.api.Assertions#assertThrows",
        "org.junit.jupiter.api.Assertions#assertTimeout",
        "org.junit.jupiter.api.Assertions#assertTimeoutPreemptively",
        "org.junit.jupiter.api.Assertions#assertTrue",
        "org.junit.jupiter.api.Assertions#fail",
    ]


def _get_junit_assertion_api_names() -> List[str]:
    return [p.split("#")[-1] for p in _get_junit_assertion_api_paths()]


def _assert_bug_is_single_method(bug):
    assert _is_single_method_bug(bug)
    assert bug.is_single_method_bug
    assert bug.is_single_file_bug
    assert len(set(f"{i.file_path}:{i.method_path}" for i in bug.bug_locations)) == 1
    return bug


def _assert_method_path_is_valid(path: str) -> str:
    assert "::" in path
    return path


def _generate_sha256_key(obj):
    import hashlib

    return hashlib.sha256(str(obj).encode()).hexdigest()


def _generate_linear_sequence(M: int, D: int) -> List[int]:
    """Generates a linearly interpolated integer sequence with defined endpoints.

    Args:
        M: Maximum index (sequence length will be M+1). Must be integer >= 0
        D: Terminal value at index M (f(M) = D). Must be integer >= 1

    Returns:
        List of integers where:
        - f(0) = 1
        - f(M) = D
        - Intermediate values calculated as round(1 + (D-1)*(i/M))

    Raises:
        ValueError: For invalid input parameters
        ZeroDivisionError: If M=0 and D != 1

    Examples:
        >>> generate_linear_sequence(3, 10)
        [1, 4, 7, 10]

        >>> generate_linear_sequence(5, 5)
        [1, 1, 2, 3, 4, 5]

        >>> generate_linear_sequence(0, 1)
        [1]
    """
    if not isinstance(M, int) or M < 0:
        raise ValueError(
            f"M must be non-negative integer, got {M} (type: {type(M).__name__})"
        )

    if not isinstance(D, int) or D < 1:
        raise ValueError(
            f"D must be positive integer >=1, got {D} (type: {type(D).__name__})"
        )

    if M == 0:
        if D != 1:
            raise ValueError(f"When M=0, D must be 1. Got D={D}")
        return [1]

    return [round(1 + (D - 1) * (i / M)) for i in range(M + 1)]


def _generate_exponential_sequence(M: int, D: int) -> List[int]:
    """Generates an exponentially increasing integer sequence with defined endpoints.

    Args:
        M: Maximum index (sequence length will be M+1). Must be integer >= 0
        D: Terminal value at index M (f(M) = D). Must be integer >= 1

    Returns:
        List of integers where:
        - f(0) = 1
        - f(M) = D
        - Intermediate values follow D^(i/M) rounded to nearest integer

    Raises:
        ValueError: For invalid input parameters
        ZeroDivisionError: If M=0 and D != 1

    Examples:
        >>> generate_exponential_sequence(3, 10)
        [1, 2, 5, 10]

        >>> generate_exponential_sequence(0, 1)
        [1]
    """

    if not isinstance(M, int) or M < 0:
        raise ValueError(
            f"M must be non-negative integer, got {M} (type: {type(M).__name__})"
        )

    if not isinstance(D, int) or D < 1:
        raise ValueError(
            f"D must be positive integer >=1, got {D} (type: {type(D).__name__})"
        )

    if M == 0:
        if D != 1:
            raise ValueError(f"When M=0, D must be 1. Got D={D}")
        return [1]

    return [round(D ** (i / M)) for i in range(M + 1)]


def _generate_alternate_sequence(list1: list, list2: list) -> list:
    result = []
    for a, b in zip(list1, list2):
        result.append(a)
        result.append(b)
    result += list1[len(list2) :] + list2[len(list1) :]
    return result
