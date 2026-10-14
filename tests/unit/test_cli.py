from typer.testing import CliRunner

from paperwatch.cli import app


def test_ingest_is_a_subcommand_and_requires_live() -> None:
    result = CliRunner().invoke(app, ["ingest"])
    assert result.exit_code == 2 and "pass --live" in result.output
