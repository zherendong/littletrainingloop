"""
Data loader for Stack v2 dataset turning stackv2 dictionaries into text to train on.
"""

import json
from stackv2_dataloader import preamble, extract

example_data = r"""{"repo_name": "LLLLLLCCCCCC/aaaaaaaa", "repo_url": "https://github.com/LLLLLLCCCCCC/aaaaaaaa", "snapshot_id": "ac8d43dbbcd9c748e4f8e769e267076b86440324", "revision_id": "52616a66af37ceb2ec5cc403168f5076da4ffc53", "directory_id": "3a825c97460f1b85965b2ab41545b7f29e4a8a93", "branch_name": "refs/heads/master", "visit_date": 1610297472535967000, "revision_date": 1459921149000000000, "committer_date": 1459921149000000000, "github_id": 55578705, "star_events_count": 0, "fork_events_count": 0, "gha_license_id": null, "gha_created_at": null, "gha_updated_at": null, "gha_pushed_at": null, "gha_language": null, "files": [{"blob_id": "522d5f7839fa28747309c6be133d8a63c2ebfabb", "path": "/README.md", "content_id": "05ee8db65dd113c163da97ae6c8e0130a13d1123", "language": "Markdown", "length_bytes": 18, "detected_licenses": [], "license_type": "no_license", "src_encoding": "UTF-8", "is_vendor": false, "is_generated": false, "alphanum_fraction": 0.7777777910232544, "alpha_fraction": 0.4444444477558136, "num_lines": 2, "avg_line_length": 8.0, "max_line_length": 10, "content": "# aaaaaaaa\n000000\n"}], "num_files": 1}"""


def test_preamble():
    """Test the preamble function"""
    row = json.loads(example_data)
    assert (
        preamble(row)
        == "=== Repository LLLLLLCCCCCC/aaaaaaaa, branch refs/heads/master with 1 files. ==="
    )


def test_extract():
    """Test the extract function"""
    row = json.loads(example_data)
    assert (
        extract(row)
        == "=== Repository LLLLLLCCCCCC/aaaaaaaa, branch refs/heads/master with 1 files. ===\n=== File /README.md (Markdown) ===\n# aaaaaaaa\n000000\n\n\n"
    )
