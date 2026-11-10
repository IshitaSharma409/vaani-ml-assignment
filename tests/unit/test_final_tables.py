from scripts.build_final_tables import _mean_tokens, _percentile, build


def test_table_helpers_use_linear_percentiles_and_token_totals():
    assert _percentile([100.0, 200.0, 300.0], 0.95) == "290.0"
    assert _percentile([], 0.50) == "not measured"
    assert _mean_tokens([{"tokens_in": "100", "tokens_out": "20"},
                         {"tokens_in": "200", "tokens_out": "30"}]) == "175.0"


def test_generated_tables_cover_all_required_scenarios_and_settings():
    tables = build()
    for scenario in ("steady", "burst", "sustained_overload", "per_minute_limit_halved", "daily_budget_exhausted"):
        assert f"| {scenario} |" in tables
    for setting in ("small_fast", "medium", "medium_reranked"):
        assert setting in tables
    assert "not measured" in tables
    assert "8,000 observed" in tables
