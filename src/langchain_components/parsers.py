"""Output parsers for code generation."""
import ast
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any

from langchain.schema import BaseOutputParser
import libcst as cst


@dataclass
class CodeParseResult:
    """Result of parsing generated code."""
    code: str
    imports: Set[str]
    boto3_services: Set[str]
    syntax_valid: bool
    errors: List[str]
    security_risks: List[str]
    dependencies: Set[str]


class PythonCodeParser(BaseOutputParser[CodeParseResult]):
    """Parser for generated Python code."""

    def parse(self, text: str) -> CodeParseResult:
        """Parse the generated code and extract metadata."""
        code = self._extract_code_block(text)
        syntax_valid, errors = self._validate_syntax(code)
        imports = self._extract_imports(code)
        boto3_services = self._extract_boto3_services(code)
        security_risks = self._check_security_risks(code)
        dependencies = self._extract_dependencies(imports)

        return CodeParseResult(
            code=code,
            imports=imports,
            boto3_services=boto3_services,
            syntax_valid=syntax_valid,
            errors=errors,
            security_risks=security_risks,
            dependencies=dependencies
        )

    def _extract_code_block(self, text: str) -> str:
        """Extract code block from the text."""
        # Look for code between triple backticks or after 'Code:' marker
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            return text[start:end].strip()
        elif "Code:" in text:
            return text[text.find("Code:") + 5:].strip()
        return text.strip()

    def _validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            return False, [str(e)]
        except Exception as e:
            return False, [f"Unexpected error: {str(e)}"]

    def _extract_imports(self, code: str) -> Set[str]:
        """Extract import statements."""
        imports = set()
        try:
            tree = cst.parse_module(code)
            for node in tree.body:
                if isinstance(node, (cst.Import, cst.ImportFrom)):
                    if isinstance(node, cst.Import):
                        for name in node.names:
                            imports.add(name.name.value)
                    else:  # ImportFrom
                        if node.module:
                            module_name = node.module.value
                            for name in node.names:
                                imports.add(f"{module_name}.{name.name.value}")
        except:
            # Fallback to basic string parsing if CST parsing fails
            for line in code.split("\n"):
                if line.startswith("import "):
                    imports.add(line[7:].strip())
                elif line.startswith("from "):
                    parts = line.split(" import ")
                    if len(parts) == 2:
                        module = parts[0][5:]
                        for name in parts[1].split(","):
                            imports.add(f"{module}.{name.strip()}")
        return imports

    def _extract_boto3_services(self, code: str) -> Set[str]:
        """Extract boto3 services used in the code."""
        services = set()
        try:
            tree = cst.parse_module(code)
            for node in tree.body:
                if isinstance(node, cst.Assign):
                    if isinstance(node.value, cst.Call):
                        if isinstance(node.value.func, cst.Attribute):
                            if isinstance(node.value.func.value, cst.Name):
                                if node.value.func.value.value == "boto3":
                                    if node.value.func.attr.value == "client":
                                        for arg in node.value.args:
                                            if isinstance(arg.value, cst.SimpleString):
                                                services.add(arg.value.value.strip("'\""))
        except:
            # Fallback to basic string parsing
            for line in code.split("\n"):
                if "boto3.client(" in line:
                    start = line.find("boto3.client(") + 13
                    end = line.find(")", start)
                    service = line[start:end].strip("'\"")
                    if service:
                        services.add(service)
        return services

    def _check_security_risks(self, code: str) -> List[str]:
        """Check for common security risks."""
        risks = []
        risk_patterns = {
            "Hardcoded credentials": [
                "aws_access_key_id",
                "aws_secret_access_key",
                "password",
                "secret",
            ],
            "Insecure SSL": [
                "verify=False",
                "check_hostname=False",
            ],
            "Dangerous file operations": [
                "os.system(",
                "subprocess.call(",
                "eval(",
                "exec(",
            ],
        }

        for risk_type, patterns in risk_patterns.items():
            for pattern in patterns:
                if pattern.lower() in code.lower():
                    risks.append(f"{risk_type} detected: {pattern}")

        return risks

    def _extract_dependencies(self, imports: Set[str]) -> Set[str]:
        """Extract external package dependencies from imports."""
        standard_libs = {
            "os", "sys", "json", "datetime", "time", "logging",
            "typing", "collections", "pathlib", "uuid", "random"
        }
        return {imp.split(".")[0] for imp in imports} - standard_libs


class ErrorRecoveryParser(BaseOutputParser[Dict[str, str]]):
    """Parser for error recovery suggestions."""

    def parse(self, text: str) -> Dict[str, str]:
        """Parse error recovery suggestions."""
        suggestions = {}
        current_error = None

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("Error:"):
                current_error = line[6:].strip()
            elif line.startswith("Fix:") and current_error:
                suggestions[current_error] = line[4:].strip()
                current_error = None

        return suggestions


class MetadataParser(BaseOutputParser[Dict[str, Any]]):
    """Parser for code metadata."""

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse metadata from generated code."""
        metadata = {
            "resources": [],
            "permissions": [],
            "estimated_cost": "low",
            "complexity": "low",
            "tags": []
        }

        # Extract AWS resources
        resources = []
        for line in text.split("\n"):
            if "boto3.client(" in line or "boto3.resource(" in line:
                service = line.split("(")[1].split(")")[0].strip("'\"")
                resources.append(service)
        metadata["resources"] = list(set(resources))

        # Estimate permissions needed
        permissions = []
        for resource in resources:
            for line in text.split("\n"):
                if f".{resource}." in line:
                    action = line.split(f".{resource}.")[1].split("(")[0]
                    permissions.append(f"{resource}:{action}")
        metadata["permissions"] = list(set(permissions))

        # Estimate cost based on services used
        high_cost_services = {"lambda", "ec2", "rds", "eks"}
        if any(service in high_cost_services for service in resources):
            metadata["estimated_cost"] = "high"

        # Estimate complexity
        if len(text.split("\n")) > 50 or len(resources) > 2:
            metadata["complexity"] = "high"

        # Extract tags
        metadata["tags"] = self._extract_tags(text)

        return metadata

    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from the code."""
        tags = set()
        if "async" in text or "await" in text:
            tags.add("async")
        if "try" in text and "except" in text:
            tags.add("error-handling")
        if "logging" in text:
            tags.add("logging")
        if "pytest" in text or "unittest" in text:
            tags.add("testing")
        return list(tags) 