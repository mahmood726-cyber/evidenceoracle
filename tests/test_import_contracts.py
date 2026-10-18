"""Imports are inert; source discovery belongs to individual loader calls."""
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_pipeline_imports_preserve_redirected_stdout():
    code = r"""
import contextlib, io, sys
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    import assemble_features
    import train_model
    assert sys.stdout is stream
    print('still usable')
assert stream.getvalue() == 'still usable\n'
"""
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_one_loader_needs_only_its_own_source(tmp_path, monkeypatch):
    import assemble_features as assembly
    def forbid_discovery(*args):
        raise AssertionError('explicit input must not discover other sources')
    monkeypatch.setattr(assembly, '_candidate_roots', forbid_discovery)
    path = tmp_path / 'fragility.csv'
    # Deliberately synthetic software fixture; no research result or study ID.
    pd.DataFrame([dict(review_id='TEST_REVIEW', robustness_score=0.3,
        classification='Fragile', frac_significant=0.5, frac_reversed=0.2,
        iqr_theta=0.1, eta2_estimator=0.2, top_dimension='Estimator')]).to_csv(path, index=False)
    result = assembly.load_fragility(path)
    assert result['review_id'].tolist() == ['TEST_REVIEW']
    assert result['fragile_or_unstable'].tolist() == [1]


def test_missing_source_is_reported_when_loader_runs(tmp_path, monkeypatch):
    import assemble_features as assembly
    monkeypatch.setenv('FRAGILITY_ROOT', str(tmp_path / 'missing'))
    with pytest.raises(FileNotFoundError, match='FRAGILITY_ROOT'):
        assembly.load_fragility()


@pytest.mark.parametrize('kind', ['file', 'dir'])
def test_explicit_missing_root_cannot_fall_back(tmp_path, monkeypatch, kind):
    import assemble_features as assembly
    fallback = tmp_path / 'fallback'
    fallback.mkdir()
    (fallback / 'input.csv').write_text('value\n1\n', encoding='utf-8')
    monkeypatch.setenv('ORACLE_TEST_ROOT', str(tmp_path / 'missing'))
    monkeypatch.setattr(assembly, '_candidate_roots', lambda *args: [fallback])
    with pytest.raises(FileNotFoundError, match='ORACLE_TEST_ROOT'):
        if kind == 'file':
            assembly.resolve_existing_file('ORACLE_TEST_ROOT', Path('input.csv'), 'unused')
        else:
            assembly.resolve_existing_dir('ORACLE_TEST_ROOT', 'unused')


@pytest.mark.parametrize('drive', ['C:/', 'D:/'])
def test_candidate_roots_include_both_drives(drive):
    import assemble_features as assembly
    assert Path(drive) / 'Example' in assembly._candidate_roots('Example')
