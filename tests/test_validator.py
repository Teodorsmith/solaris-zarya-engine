import pytest

from agent.engine.validator import SecurityError, SkillValidator, validate_ast


def test_ast_allowlist():
    # Valid imports
    validate_ast("import json\nimport re\nfrom dataclasses import dataclass")

    # Invalid import
    with pytest.raises(SecurityError, match="forbidden"):
        validate_ast("import os")

    with pytest.raises(SecurityError, match="forbidden"):
        validate_ast("from subprocess import run")


def test_ast_banned_primitives():
    with pytest.raises(SecurityError, match="forbidden"):
        validate_ast("eval('1+1')")

    with pytest.raises(SecurityError, match="forbidden"):
        validate_ast("exec('print(1)')")

    with pytest.raises(SecurityError, match="forbidden"):
        validate_ast("getattr(obj, 'attr')")


def test_ast_banned_attributes():
    with pytest.raises(SecurityError, match="forbidden"):
        validate_ast("x.__class__")

    with pytest.raises(SecurityError, match="forbidden"):
        validate_ast("().__class__.__bases__[0].__subclasses__()")


def test_validator_success():
    validator = SkillValidator()
    code = """
def execute():
    return "hello"
"""
    test_code = """
import unittest
class TestSkill(unittest.TestCase):
    def test_hello(self):
        from test_skill import execute
        self.assertEqual(execute(), "hello")
"""
    result = validator.validate_and_run("test_skill", code, test_code)
    assert result.status == "ok"
    assert result.result == "hello"


def test_validator_timeout():
    validator = SkillValidator(timeout_seconds=1)
    code = """
def execute():
    while True:
        pass
"""
    test_code = """
import unittest
class TestSkill(unittest.TestCase):
    def test_timeout(self):
        from test_skill import execute
        execute()
"""
    with pytest.raises(SecurityError, match="timed out"):
        validator.validate_and_run("test_skill", code, test_code)


def test_validator_test_failure():
    validator = SkillValidator()
    code = """
def execute():
    return "wrong"
"""
    test_code = """
import unittest
class TestSkill(unittest.TestCase):
    def test_fail(self):
        from test_skill import execute
        self.assertEqual(execute(), "right")
"""
    with pytest.raises(SecurityError, match="Unit tests failed"):
        validator.validate_and_run("test_skill", code, test_code)
