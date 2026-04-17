from __future__ import annotations

import astro_planner
from PySide6.QtWidgets import QApplication


def test_same_runtime_site_matches_same_coordinates_and_timezone() -> None:
    first = astro_planner.Site(name="WRO", latitude=51.1, longitude=17.0, elevation=120.0)
    second = astro_planner.Site(name="WRO", latitude=51.1, longitude=17.0, elevation=120.0)

    assert astro_planner._same_runtime_site(first, second) is True


def test_should_apply_observatory_change_skips_same_active_site() -> None:
    current = astro_planner.Site(name="WRO", latitude=51.1, longitude=17.0, elevation=120.0)
    target = astro_planner.Site(name="WRO", latitude=51.1, longitude=17.0, elevation=120.0)

    assert astro_planner._should_apply_observatory_change("WRO", current, "WRO", target) is False


def test_should_apply_observatory_change_detects_real_site_switch() -> None:
    current = astro_planner.Site(name="OCM", latitude=-24.59, longitude=-70.19, elevation=2800.0)
    target = astro_planner.Site(name="WRO", latitude=51.1, longitude=17.0, elevation=120.0)

    assert astro_planner._should_apply_observatory_change("OCM", current, "WRO", target) is True


def test_qcolor_from_token_parses_rgba_css() -> None:
    color = astro_planner._qcolor_from_token("rgba(255, 95, 135, 0.560)")

    assert color.isValid()
    assert color.red() == 255
    assert color.green() == 95
    assert color.blue() == 135
    assert color.alpha() == 143


def test_bhtom_suggestion_source_message_mentions_cache_ttl() -> None:
    message = astro_planner._bhtom_suggestion_source_message("cache")

    assert "cached BHTOM target list" in message
    assert "TTL 1h" in message


def test_bhtom_suggestion_source_message_mentions_fresh_fetch() -> None:
    message = astro_planner._bhtom_suggestion_source_message("network")

    assert "fresh BHTOM target list" in message
    assert "cached for 1h" in message


def test_distribute_extra_table_width_fills_available_space() -> None:
    widths = {1: 180, 2: 90, 3: 70}
    fitted = astro_planner._distribute_extra_table_width(
        widths,
        available_width=420,
        stretch_weights={1: 4.0, 2: 1.0, 3: 1.0},
    )

    assert sum(fitted.values()) == 420
    assert fitted[1] > widths[1]
    assert fitted[1] > fitted[2]
    assert fitted[2] >= widths[2]
    assert fitted[3] >= widths[3]


def test_distribute_extra_table_width_keeps_widths_when_space_is_tight() -> None:
    widths = {1: 180, 2: 90, 3: 70}
    fitted = astro_planner._distribute_extra_table_width(
        widths,
        available_width=300,
        stretch_weights={1: 4.0, 2: 1.0, 3: 1.0},
    )

    assert fitted == widths


def test_target_table_model_hover_row_emits_row_refresh() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    model = astro_planner.TargetTableModel(
        [astro_planner.Target(name="M31", ra=10.684, dec=41.269)]
    )
    emissions: list[tuple[int, int, tuple[int, ...]]] = []

    def _capture(top_left, bottom_right, roles) -> None:
        emissions.append(
            (
                top_left.row(),
                bottom_right.row(),
                tuple(int(role) for role in roles),
            )
        )

    model.dataChanged.connect(_capture)
    model.set_hover_row(0)

    assert model._hover_row == 0
    assert emissions
    assert emissions[-1][0] == 0
    assert emissions[-1][1] == 0
    assert int(astro_planner.Qt.BackgroundRole) in emissions[-1][2]


def test_plot_font_css_stack_avoids_generic_sans_serif_fallback() -> None:
    stack = astro_planner._plot_font_css_stack({"display_font_family": "Rajdhani"})

    assert "sans-serif" not in stack
    assert '"Rajdhani"' in stack
