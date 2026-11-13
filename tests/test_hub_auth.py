# SPDX-License-Identifier: Apache-2.0
import ast
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch


class HubCredentialTests(unittest.TestCase):
    def test_only_exact_https_hub_origin_receives_token(self):
        source = Path(__file__).resolve().parents[1] / 'runtime-source/gguf-reader/gguf/utility.py'
        tree = ast.parse(source.read_text())
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == 'SafetensorRemote')
        method = copy.deepcopy(next(node for node in original.body if isinstance(node, ast.FunctionDef)
                                    and node.name == '_get_request_headers'))
        minimal = ast.ClassDef(name='Transport', bases=[], keywords=[], body=[method], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[minimal], type_ignores=[]))
        namespace = {'os': os}
        exec(compile(module, str(source), 'exec'), namespace)
        with patch.dict(os.environ, {'HF_TOKEN': 'dummy-test-token'}):
            for url in ('https://huggingface.co/owner/repo/resolve/main/model',
                        'https://huggingface.co:443/owner/repo'):
                self.assertEqual(namespace['Transport']._get_request_headers(url)['Authorization'],
                                 'Bearer dummy-test-token')
            for url in ('http://huggingface.co/model', 'https://evil.example/model',
                        'https://huggingface.co.evil.example/model',
                        'https://huggingface.co@evil.example/model',
                        'https://user@huggingface.co/model',
                        'https://huggingface.co:444/model'):
                with self.subTest(url=url):
                    self.assertNotIn('Authorization', namespace['Transport']._get_request_headers(url))


if __name__ == '__main__':
    unittest.main()
