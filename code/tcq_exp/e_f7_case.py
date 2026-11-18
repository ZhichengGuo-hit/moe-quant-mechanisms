"""E-F7 单例执行器: 把 HumanEval 的 check() 拆成逐条 assert 计通过数。

输入: 一个 JSON 文件 {"code","test","entry_point"}
输出: 一行 JSON {"n","passed","flags","error"}
用法: python -I e_f7_case.py payload.json   (由 e_f7_tests_passed.py 在临时目录里调用)
"""
import ast
import json
import sys

RESULTS = []


def _safe(fn):
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001  单条 assert 的异常只判 False
        return False


class AssertToCount(ast.NodeTransformer):
    def visit_Assert(self, node):
        lam = ast.Lambda(
            args=ast.arguments(posonlyargs=[], args=[], vararg=None,
                               kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=self.generic_visit(node.test))
        call = ast.Call(func=ast.Name(id="_safe", ctx=ast.Load()), args=[lam], keywords=[])
        app = ast.Call(func=ast.Attribute(value=ast.Name(id="RESULTS", ctx=ast.Load()),
                                          attr="append", ctx=ast.Load()),
                       args=[call], keywords=[])
        return ast.copy_location(ast.Expr(value=app), node)


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        p = json.load(f)
    out = {"n": 0, "passed": 0, "flags": [], "error": None}
    try:
        tree = ast.parse(p["test"])
        tree = AssertToCount().visit(tree)
        ast.fix_missing_locations(tree)
        ns = {"RESULTS": RESULTS, "_safe": _safe, "__name__": "he_case"}
        exec(compile(p["code"], "<model_code>", "exec"), ns)  # noqa: S102
        exec(compile(tree, "<test_code>", "exec"), ns)        # noqa: S102
        ns["check"](ns.get(p["entry_point"]))
    except BaseException as e:  # noqa: BLE001  模型代码本身可能 raise
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    out["flags"] = list(RESULTS)
    out["n"] = len(RESULTS)
    out["passed"] = sum(1 for x in RESULTS if x)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
