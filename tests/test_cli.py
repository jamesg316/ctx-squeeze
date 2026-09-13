import json

from ctx_squeeze.cli import main


def _write(path, content):
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_document_mode_writes_squeezed_text_to_stdout(tmp_path, capsys):
    text = "\n\n".join(f"Paragraph {i} about something entirely different each time." for i in range(20))
    path = _write(tmp_path / "doc.md", text)
    main([path, "--budget", "40", "--strategy", "score"])
    out = capsys.readouterr().out
    assert "segments elided" in out


def test_stats_flag_prints_summary_to_stderr(tmp_path, capsys):
    text = "\n\n".join(f"Paragraph {i} about something entirely different each time." for i in range(20))
    path = _write(tmp_path / "doc.md", text)
    main([path, "--budget", "40", "--strategy", "score", "--stats"])
    err = capsys.readouterr().err
    assert "kept" in err and "segments" in err and "budget 40" in err


def test_json_flag_emits_report_object(tmp_path, capsys):
    path = _write(tmp_path / "doc.md", "One short paragraph.")
    main([path, "--budget", "1000", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == "One short paragraph."
    assert payload["segments_in"] == 1


def test_no_marker_omits_elision_text(tmp_path, capsys):
    text = "\n\n".join(f"Paragraph {i} about something entirely different each time." for i in range(20))
    path = _write(tmp_path / "doc.md", text)
    main([path, "--budget", "40", "--strategy", "score", "--no-marker"])
    out = capsys.readouterr().out
    assert "segments elided" not in out


def test_output_flag_writes_to_file_instead_of_stdout(tmp_path, capsys):
    src = _write(tmp_path / "doc.md", "One short paragraph.")
    dest = tmp_path / "out.md"
    main([src, "--budget", "1000", "-o", str(dest)])
    assert capsys.readouterr().out == ""
    assert dest.read_text(encoding="utf-8") == "One short paragraph."


def test_messages_mode_emits_json_array_of_dicts(tmp_path, capsys):
    history = [
        {"role": "system", "content": "be careful"},
        {"role": "user", "content": "hello"},
    ]
    path = _write(tmp_path / "chat.json", json.dumps(history))
    main([path, "--messages", "--budget", "1000"])
    dicts = json.loads(capsys.readouterr().out)
    assert dicts == history


def test_messages_mode_json_flag_emits_report_object(tmp_path, capsys):
    history = [{"role": "user", "content": f"filler message number {i} with some words"} for i in range(10)]
    path = _write(tmp_path / "chat.json", json.dumps(history))
    main([path, "--messages", "--budget", "10", "--recent-turns", "1", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["messages_in"] == 10
    assert payload["messages_out"] < payload["messages_in"]
    assert "pinned_tool_results" in payload


def test_reads_from_stdin_when_input_is_dash(tmp_path, capsys, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("One short paragraph."))
    main(["-", "--budget", "1000"])
    out = capsys.readouterr().out
    assert out.strip() == "One short paragraph."
