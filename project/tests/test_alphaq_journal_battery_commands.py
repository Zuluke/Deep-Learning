from __future__ import annotations

from scripts.build_alphaq_journal_battery_commands import command_blocks


def test_command_blocks_group_targets_by_stage_and_chunk() -> None:
    rows = [
        make_row("a", "full-action-expansion"),
        make_row("b", "full-action-expansion"),
        make_row("c", "full-action-expansion"),
        make_row("repair", "full-action-repair"),
        make_row("screen", "tensor-v3-screen"),
        make_row("restricted", "restricted-action-pilot"),
    ]

    blocks = command_blocks(rows, max_targets_per_job=2)

    commands = {block["stage"]: [] for block in blocks}
    for block in blocks:
        commands[block["stage"]].append(block)

    assert commands["full-action-repair"][0]["targets"] == ["repair"]
    assert commands["full-action-expansion"][0]["targets"] == ["a", "b"]
    assert commands["full-action-expansion"][1]["targets"] == ["c"]
    assert commands["tensor-v3-screen"][0]["targets"] == ["screen"]
    assert commands["tensor-v3-screen"][0]["command"] == ""
    assert "tensor-v3/profile screening" in commands["tensor-v3-screen"][0]["blocked_reason"]
    assert commands["restricted-action-pilot"][0]["targets"] == ["restricted"]
    assert "--output-suffix journal_full_1" in commands["full-action-expansion"][0]["command"]
    assert commands["restricted-action-pilot"][0]["command"] == ""
    assert "restricted-action" in commands["restricted-action-pilot"][0]["blocked_reason"]


def test_command_blocks_use_target_named_suffixes_for_single_target_chunks() -> None:
    rows = [
        make_row("mod_mult_55", "full-action-expansion"),
        make_row("cuccaro_adder_n4", "full-action-expansion"),
    ]

    blocks = command_blocks(rows, max_targets_per_job=1)

    assert "--output-suffix journal_full_mod_mult_55" in blocks[0]["command"]
    assert "--job-name alphaq_journal_full_mod_mult_55" in blocks[0]["command"]
    assert "--output-suffix journal_full_cuccaro_adder_n4" in blocks[1]["command"]


def test_command_blocks_ignore_unknown_stage() -> None:
    assert command_blocks([make_row("x", "unknown")], max_targets_per_job=2) == []


def make_row(target: str, stage: str) -> dict[str, str]:
    return {
        "target": target,
        "recommended_stage": stage,
    }
