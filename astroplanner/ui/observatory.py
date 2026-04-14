from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from astroplanner.i18n import current_language, localize_widget_tree
from astroplanner.models import DEFAULT_LIMITING_MAGNITUDE, Site
from astroplanner.ui.common import _fit_dialog_to_screen
from astroplanner.ui.theme_utils import (
    _set_button_icon_kind,
    _set_button_variant,
    _set_label_tone,
    _style_dialog_button_box,
)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ObservatoryLookupWorker(QThread):
    """Resolve observatory coordinates in background to keep dialog responsive."""

    completed = Signal(float, float, object, str, str)

    def __init__(
        self,
        query: str,
        resolver: Callable[[str], tuple[float, float, Optional[float], str]],
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.query = str(query)
        self.resolver = resolver

    def run(self):
        try:
            lat, lon, elev, display_name = self.resolver(self.query)
            self.completed.emit(float(lat), float(lon), elev, str(display_name), "")
        except Exception as exc:  # noqa: BLE001
            self.completed.emit(0.0, 0.0, None, "", str(exc))


class ObservatoryManagerDialog(QDialog):
    """Manage observatories with search and inline configuration editing."""

    def __init__(
        self,
        observatories: dict[str, Site],
        parent=None,
        preset_keys: Optional[dict[str, str]] = None,
        selected_name: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("Observatory Manager")
        self.resize(1140, 700)

        self._sites: dict[str, Site] = {
            name: Site(**site.model_dump())
            for name, site in observatories.items()
        }
        preferred_name = str(selected_name or "").strip()
        self._selected_name: Optional[str] = preferred_name if preferred_name in self._sites else None
        self._loading_fields = False
        self._list_sync = False
        self._preset_sync = False
        preset_keys = preset_keys or {}
        self._site_preset_keys: dict[str, str] = {
            name: str(preset_keys.get(name, "custom") or "custom")
            for name in self._sites
        }
        self._bhtom_preset_map: dict[str, dict[str, object]] = {}
        self._lookup_worker: Optional[ObservatoryLookupWorker] = None
        self._preset_loading = False
        self._preset_status_default = "BHTOM presets load in background when token is configured."

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QLabel(
            "Manage observatories (coordinates, limiting magnitude, telescope/camera profile, and optional custom weather endpoint).",
            self,
        )
        header.setObjectName("SectionHint")
        header.setWordWrap(True)
        root.addWidget(header)

        left_col = QWidget(self)
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(QLabel("Observatories", left_col))
        self.search_edit = QLineEdit(left_col)
        self.search_edit.setPlaceholderText("Search observatory...")
        self.search_edit.setMinimumHeight(34)
        left_layout.addWidget(self.search_edit)
        self.obs_list = QListWidget(left_col)
        self.obs_list.setMinimumWidth(340)
        self.obs_list.setUniformItemSizes(True)
        left_layout.addWidget(self.obs_list, 1)
        list_actions = QHBoxLayout()
        list_actions.setContentsMargins(0, 0, 0, 0)
        list_actions.setSpacing(6)
        self.add_btn = QPushButton("New", left_col)
        self.remove_btn = QPushButton("Remove", left_col)
        self.save_cfg_btn = QPushButton("Save Config", left_col)
        self.add_btn.setMinimumHeight(34)
        self.remove_btn.setMinimumHeight(34)
        self.save_cfg_btn.setMinimumHeight(34)
        self.add_btn.setMinimumWidth(92)
        self.remove_btn.setMinimumWidth(92)
        self.save_cfg_btn.setMinimumWidth(128)
        _set_button_variant(self.add_btn, "primary")
        _set_button_variant(self.remove_btn, "ghost")
        _set_button_variant(self.save_cfg_btn, "secondary")
        _set_button_icon_kind(self.add_btn, "new")
        _set_button_icon_kind(self.remove_btn, "remove")
        _set_button_icon_kind(self.save_cfg_btn, "save")
        list_actions.addWidget(self.add_btn)
        list_actions.addWidget(self.remove_btn)
        list_actions.addWidget(self.save_cfg_btn)
        list_actions.addStretch(1)
        left_layout.addLayout(list_actions)

        right_col = QWidget(self)
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(QLabel("Configuration", right_col))

        top_form = QFormLayout()
        top_form.setContentsMargins(0, 0, 0, 0)
        top_form.setHorizontalSpacing(12)
        top_form.setVerticalSpacing(8)
        top_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.name_edit = QLineEdit(right_col)
        self.name_edit.setPlaceholderText("Observatory name")
        self.name_edit.setMinimumWidth(200)
        self.name_edit.setMaximumWidth(300)
        self.name_edit.setMinimumHeight(34)
        self.lat_edit = QLineEdit(right_col)
        self.lat_edit.setValidator(QDoubleValidator(-90.0, 90.0, 6, right_col))
        self.lat_edit.setMinimumHeight(34)
        self.lon_edit = QLineEdit(right_col)
        self.lon_edit.setValidator(QDoubleValidator(-180.0, 180.0, 6, right_col))
        self.lon_edit.setMinimumHeight(34)
        self.elev_edit = QLineEdit(right_col)
        self.elev_edit.setValidator(QDoubleValidator(-1000.0, 20000.0, 2, right_col))
        self.elev_edit.setMinimumHeight(34)
        self.lim_mag_spin = QDoubleSpinBox(right_col)
        self.lim_mag_spin.setRange(-5.0, 30.0)
        self.lim_mag_spin.setDecimals(1)
        self.lim_mag_spin.setSingleStep(0.1)
        self.lim_mag_spin.setMinimumHeight(34)
        self.telescope_diameter_spin = QDoubleSpinBox(right_col)
        self.telescope_diameter_spin.setRange(0.0, 50.0)
        self.telescope_diameter_spin.setDecimals(3)
        self.telescope_diameter_spin.setSingleStep(0.05)
        self.telescope_diameter_spin.setSuffix(" m")
        self.telescope_diameter_spin.setMinimumHeight(34)
        self.focal_length_spin = QDoubleSpinBox(right_col)
        self.focal_length_spin.setRange(0.0, 20000.0)
        self.focal_length_spin.setDecimals(1)
        self.focal_length_spin.setSingleStep(10.0)
        self.focal_length_spin.setSuffix(" mm")
        self.focal_length_spin.setMinimumHeight(34)
        self.pixel_size_spin = QDoubleSpinBox(right_col)
        self.pixel_size_spin.setRange(0.0, 100.0)
        self.pixel_size_spin.setDecimals(3)
        self.pixel_size_spin.setSingleStep(0.1)
        self.pixel_size_spin.setSuffix(" µm")
        self.pixel_size_spin.setMinimumHeight(34)
        self.detector_width_spin = QSpinBox(right_col)
        self.detector_width_spin.setRange(0, 20000)
        self.detector_width_spin.setSingleStep(16)
        self.detector_width_spin.setSuffix(" px")
        self.detector_width_spin.setMinimumHeight(34)
        self.detector_height_spin = QSpinBox(right_col)
        self.detector_height_spin.setRange(0, 20000)
        self.detector_height_spin.setSingleStep(16)
        self.detector_height_spin.setSuffix(" px")
        self.detector_height_spin.setMinimumHeight(34)
        self.pixel_scale_display = QLineEdit(right_col)
        self.pixel_scale_display.setReadOnly(True)
        self.pixel_scale_display.setPlaceholderText("auto")
        self.pixel_scale_display.setMinimumHeight(34)
        self.fov_display = QLineEdit(right_col)
        self.fov_display.setReadOnly(True)
        self.fov_display.setPlaceholderText("auto")
        self.fov_display.setMinimumHeight(34)
        self.custom_conditions_url_edit = QLineEdit(right_col)
        self.custom_conditions_url_edit.setPlaceholderText("https://example.com/station.json")
        self.custom_conditions_url_edit.setToolTip(
            "Optional endpoint used when Weather source = Custom URL. Supports AstroPlanner JSON and "
            "Weather.com PWS observations/all/1day feeds."
        )
        self.custom_conditions_url_edit.setMinimumHeight(34)
        self.lookup_btn = QPushButton("Lookup Coordinates", right_col)
        self.lookup_btn.setToolTip("Resolve latitude/longitude/elevation from observatory name")
        self.lookup_btn.setMinimumHeight(34)
        self.lookup_btn.setMinimumWidth(170)
        _set_button_variant(self.lookup_btn, "secondary")
        _set_button_icon_kind(self.lookup_btn, "lookup")
        self.preset_combo = QComboBox(right_col)
        self.preset_combo.addItem("Custom (editable)", "custom")
        self.preset_combo.setMinimumHeight(34)
        self.preset_combo.setMaximumWidth(300)
        self.preset_info = QLabel("", right_col)
        self.preset_info.setObjectName("SectionHint")
        self.preset_info.setWordWrap(True)
        _set_label_tone(self.preset_info, "muted")
        self.lookup_info = QLabel("", right_col)
        self.lookup_info.setObjectName("SectionHint")
        self.lookup_info.setWordWrap(True)
        _set_label_tone(self.lookup_info, "info")

        preset_actions = QWidget(right_col)
        preset_actions_layout = QHBoxLayout(preset_actions)
        preset_actions_layout.setContentsMargins(0, 0, 0, 0)
        preset_actions_layout.setSpacing(8)
        preset_actions_layout.addWidget(self.lookup_btn)
        preset_actions_layout.addStretch(1)
        self.preset_progress = QProgressBar(right_col)
        self.preset_progress.setTextVisible(False)
        self.preset_progress.setMinimumWidth(140)
        self.preset_progress.setMaximumWidth(220)
        self.preset_progress.setRange(0, 0)
        self.preset_progress.hide()
        self.preset_status_label = QLabel(self._preset_status_default, right_col)
        self.preset_status_label.setObjectName("SectionHint")
        self.preset_status_label.setWordWrap(True)
        _set_label_tone(self.preset_status_label, "muted")
        self.lookup_progress = QProgressBar(right_col)
        self.lookup_progress.setTextVisible(False)
        self.lookup_progress.setMinimumWidth(140)
        self.lookup_progress.setMaximumWidth(220)
        self.lookup_progress.setRange(0, 0)
        self.lookup_progress.hide()

        for widget in (
            self.name_edit,
            self.lat_edit,
            self.lon_edit,
            self.elev_edit,
            self.lim_mag_spin,
            self.telescope_diameter_spin,
            self.focal_length_spin,
            self.pixel_size_spin,
            self.detector_width_spin,
            self.detector_height_spin,
            self.pixel_scale_display,
            self.fov_display,
            self.custom_conditions_url_edit,
            self.preset_combo,
        ):
            widget.setMinimumWidth(170)
            widget.setMaximumWidth(300)

        top_form.addRow("Preset:", self.preset_combo)
        top_form.addRow("", preset_actions)
        top_form.addRow("", self.preset_progress)
        top_form.addRow("", self.lookup_progress)
        top_form.addRow("", self.preset_status_label)
        top_form.addRow("", self.lookup_info)
        right_layout.addLayout(top_form)

        site_panel = QWidget(right_col)
        site_layout = QVBoxLayout(site_panel)
        site_layout.setContentsMargins(0, 0, 0, 0)
        site_layout.setSpacing(6)
        site_title = QLabel("Site", site_panel)
        site_title.setObjectName("SectionHint")
        site_layout.addWidget(site_title)
        site_form = QFormLayout()
        site_form.setContentsMargins(0, 0, 0, 0)
        site_form.setHorizontalSpacing(12)
        site_form.setVerticalSpacing(8)
        site_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        site_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        site_form.addRow("Name:", self.name_edit)
        site_form.addRow("Latitude:", self.lat_edit)
        site_form.addRow("Longitude:", self.lon_edit)
        site_form.addRow("Elevation (m):", self.elev_edit)
        site_form.addRow("Limiting Mag:", self.lim_mag_spin)
        site_form.addRow("Custom conditions URL:", self.custom_conditions_url_edit)
        site_layout.addLayout(site_form)
        site_layout.addStretch(1)

        optics_panel = QWidget(right_col)
        optics_layout = QVBoxLayout(optics_panel)
        optics_layout.setContentsMargins(0, 0, 0, 0)
        optics_layout.setSpacing(6)
        optics_title = QLabel("Optics & Camera", optics_panel)
        optics_title.setObjectName("SectionHint")
        optics_layout.addWidget(optics_title)
        optics_form = QFormLayout()
        optics_form.setContentsMargins(0, 0, 0, 0)
        optics_form.setHorizontalSpacing(12)
        optics_form.setVerticalSpacing(8)
        optics_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        optics_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        optics_form.addRow("Telescope Ø (m):", self.telescope_diameter_spin)
        optics_form.addRow("Focal length:", self.focal_length_spin)
        optics_form.addRow("Pixel size:", self.pixel_size_spin)
        optics_form.addRow("Detector width:", self.detector_width_spin)
        optics_form.addRow("Detector height:", self.detector_height_spin)
        optics_form.addRow("Pixel scale:", self.pixel_scale_display)
        optics_form.addRow("FOV:", self.fov_display)
        optics_layout.addLayout(optics_form)
        optics_layout.addStretch(1)

        fields_row = QHBoxLayout()
        fields_row.setContentsMargins(0, 0, 0, 0)
        fields_row.setSpacing(16)
        fields_row.addWidget(site_panel, 1)
        fields_row.addWidget(optics_panel, 1)
        right_layout.addLayout(fields_row, 1)
        right_layout.addWidget(self.preset_info)
        right_layout.addStretch(1)

        body_splitter = QSplitter(Qt.Horizontal, self)
        body_splitter.addWidget(left_col)
        body_splitter.addWidget(right_col)
        body_splitter.setHandleWidth(1)
        body_splitter.setStretchFactor(0, 2)
        body_splitter.setStretchFactor(1, 3)
        body_splitter.setSizes([460, 700])
        body_splitter.setChildrenCollapsible(False)
        root.addWidget(body_splitter, 1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self._accept_dialog)
        self.button_box.rejected.connect(self.reject)
        _style_dialog_button_box(self.button_box)
        self.save_cfg_btn.clicked.connect(self._save_to_config)
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 2, 0, 0)
        bottom_bar.setSpacing(8)
        bottom_bar.addStretch(1)
        bottom_bar.addWidget(self.button_box, 0, Qt.AlignRight)
        root.addLayout(bottom_bar)

        self.search_edit.textChanged.connect(self._refresh_list)
        self.obs_list.currentItemChanged.connect(self._on_list_selection_changed)
        self.add_btn.clicked.connect(self._add_observatory)
        self.remove_btn.clicked.connect(self._remove_observatory)
        self.lookup_btn.clicked.connect(self._lookup_coordinates_for_current)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.name_edit.editingFinished.connect(self._store_current_site_from_editors)
        self.lat_edit.editingFinished.connect(self._store_current_site_from_editors)
        self.lon_edit.editingFinished.connect(self._store_current_site_from_editors)
        self.elev_edit.editingFinished.connect(self._store_current_site_from_editors)
        self.custom_conditions_url_edit.editingFinished.connect(self._store_current_site_from_editors)
        self.lim_mag_spin.valueChanged.connect(self._on_limiting_mag_changed)
        self.telescope_diameter_spin.valueChanged.connect(self._on_optics_fields_changed)
        self.focal_length_spin.valueChanged.connect(self._on_optics_fields_changed)
        self.pixel_size_spin.valueChanged.connect(self._on_optics_fields_changed)
        self.detector_width_spin.valueChanged.connect(self._on_optics_fields_changed)
        self.detector_height_spin.valueChanged.connect(self._on_optics_fields_changed)

        parent = self.parent()
        cached_loader = getattr(parent, "_cached_bhtom_observatory_presets", None)
        if callable(cached_loader):
            try:
                cached = cached_loader()
            except Exception:
                cached = None
            if cached:
                self._set_bhtom_presets(cached)
                self.preset_status_label.setText(f"Loaded {len(cached)} cached BHTOM presets.")
        parent_changed_sig = getattr(parent, "bhtom_observatory_presets_changed", None)
        if parent_changed_sig is not None and hasattr(parent_changed_sig, "connect"):
            parent_changed_sig.connect(self._on_parent_bhtom_presets_changed)
        parent_loading_sig = getattr(parent, "bhtom_observatory_presets_loading", None)
        if parent_loading_sig is not None and hasattr(parent_loading_sig, "connect"):
            parent_loading_sig.connect(self._on_parent_bhtom_presets_loading)
        status_getter = getattr(parent, "_bhtom_observatory_prefetch_status", None)
        if callable(status_getter):
            try:
                is_loading, status_txt = status_getter()
            except Exception:
                is_loading, status_txt = False, ""
            self._set_preset_loading_state(bool(is_loading), str(status_txt or ""))
        launcher = getattr(parent, "_ensure_bhtom_observatory_prefetch", None)
        if callable(launcher):
            try:
                launcher(force_refresh=False)
            except Exception:
                pass
        self._rebuild_preset_combo()
        self._refresh_list()
        _fit_dialog_to_screen(
            self,
            preferred_width=1460,
            preferred_height=860,
            min_width=1120,
            min_height=720,
        )
        if self.obs_list.count() == 0:
            self._set_editors_enabled(False)
        localize_widget_tree(self, current_language())

    def observatories(self) -> dict[str, Site]:
        return {
            name: Site(**site.model_dump())
            for name, site in sorted(self._sites.items(), key=lambda item: item[0].lower())
        }

    def preset_keys(self) -> dict[str, str]:
        return {
            name: str(self._site_preset_keys.get(name, "custom") or "custom")
            for name in sorted(self._sites.keys(), key=str.lower)
        }

    def _manual_editor_widgets(self) -> tuple[QWidget, ...]:
        return (
            self.lat_edit,
            self.lon_edit,
            self.elev_edit,
            self.lim_mag_spin,
            self.custom_conditions_url_edit,
            self.telescope_diameter_spin,
            self.focal_length_spin,
            self.pixel_size_spin,
            self.detector_width_spin,
            self.detector_height_spin,
            self.lookup_btn,
        )

    def _apply_custom_mode_ui(self) -> None:
        if self._selected_name is None:
            return
        preset_key = str(self._site_preset_keys.get(self._selected_name, "custom") or "custom")
        editable = preset_key == "custom"
        if editable:
            self.preset_info.setText("Custom preset: manual editing enabled.")
            return
        preset = self._bhtom_preset_map.get(preset_key)
        if isinstance(preset, dict):
            label = str(preset.get("label", "BHTOM preset")).strip() or "BHTOM preset"
            self.preset_info.setText(f"Using preset: {label} (values are editable).")
            return
        self.preset_info.setText(f"Saved preset key '{preset_key}' (values are editable).")

    def _rebuild_preset_combo(self) -> None:
        selected_key = "custom"
        if self._selected_name is not None:
            selected_key = str(self._site_preset_keys.get(self._selected_name, "custom") or "custom")
        self._preset_sync = True
        try:
            self.preset_combo.clear()
            self.preset_combo.addItem("Custom (editable)", "custom")
            for key, preset in sorted(
                self._bhtom_preset_map.items(),
                key=lambda item: str(item[1].get("label", "")).lower(),
            ):
                self.preset_combo.addItem(str(preset.get("label", key)), key)
            idx = self.preset_combo.findData(selected_key)
            if idx < 0:
                idx = 0
            self.preset_combo.setCurrentIndex(idx)
        finally:
            self._preset_sync = False
        self._apply_custom_mode_ui()

    def _set_editors_enabled(self, enabled: bool) -> None:
        for widget in (
            self.name_edit,
            self.preset_combo,
            self.pixel_scale_display,
            self.fov_display,
            self.remove_btn,
        ):
            widget.setEnabled(enabled)
        for widget in self._manual_editor_widgets():
            widget.setEnabled(enabled)
        if not enabled:
            self.lookup_info.setText("")
            self.preset_info.setText("")
            self.pixel_scale_display.clear()
            self.fov_display.clear()
            self.custom_conditions_url_edit.clear()
            return
        self._apply_custom_mode_ui()

    def _set_bhtom_presets(self, presets: object) -> None:
        self._bhtom_preset_map = {}
        if not isinstance(presets, list):
            self._rebuild_preset_combo()
            return
        for preset in presets:
            if not isinstance(preset, dict):
                continue
            key = str(preset.get("key", "")).strip()
            label = str(preset.get("label", "")).strip()
            site = preset.get("site")
            if not key or not label or not isinstance(site, Site):
                continue
            self._bhtom_preset_map[key] = {
                "key": key,
                "label": label,
                "site": Site(**site.model_dump()),
            }
        self._rebuild_preset_combo()

    def _set_preset_loading_state(self, loading: bool, message: str = "") -> None:
        self._preset_loading = bool(loading)
        if self._preset_loading:
            self.preset_progress.setRange(0, 0)
            self.preset_progress.show()
            self.preset_status_label.setText(message.strip() or "Loading BHTOM presets...")
            _set_label_tone(self.preset_status_label, "info")
            return
        self.preset_progress.hide()
        self.preset_progress.setRange(0, 1)
        self.preset_progress.setValue(0)
        text = message.strip()
        if text:
            self.preset_status_label.setText(text)
            _set_label_tone(self.preset_status_label, "success")
            return
        if self._bhtom_preset_map:
            self.preset_status_label.setText(f"Loaded {len(self._bhtom_preset_map)} BHTOM presets.")
            _set_label_tone(self.preset_status_label, "success")
        else:
            self.preset_status_label.setText(self._preset_status_default)
            _set_label_tone(self.preset_status_label, "muted")

    @Slot(list, str)
    def _on_parent_bhtom_presets_changed(self, presets: list, message: str) -> None:
        self._set_bhtom_presets(presets)
        final_message = str(message or "").strip()
        if not final_message:
            final_message = f"Loaded {len(self._bhtom_preset_map)} BHTOM presets."
        self._set_preset_loading_state(False, final_message)

    @Slot(bool, str)
    def _on_parent_bhtom_presets_loading(self, loading: bool, message: str) -> None:
        self._set_preset_loading_state(bool(loading), str(message or ""))

    def _set_lookup_loading_state(self, loading: bool) -> None:
        if loading:
            self.lookup_progress.setRange(0, 0)
            self.lookup_progress.show()
            self.lookup_btn.setEnabled(False)
            self.lookup_btn.setText("...")
            return
        self.lookup_progress.hide()
        self.lookup_progress.setRange(0, 1)
        self.lookup_progress.setValue(0)
        self.lookup_btn.setEnabled(self._selected_name is not None)
        self.lookup_btn.setText("Lookup Coordinates")

    @Slot(float, float, object, str, str)
    def _on_lookup_worker_completed(self, lat: float, lon: float, elev: object, display_name: str, err: str) -> None:
        self._set_lookup_loading_state(False)
        if err:
            QMessageBox.warning(self, "Lookup Failed", err)
            return
        self._loading_fields = True
        try:
            self.lat_edit.setText(f"{float(lat):.6f}")
            self.lon_edit.setText(f"{float(lon):.6f}")
            elev_value = _safe_float(elev)
            if elev_value is not None:
                self.elev_edit.setText(f"{elev_value:.1f}")
            elif not self.elev_edit.text().strip():
                self.elev_edit.setText("0")
            self.lookup_info.setText(str(display_name or "").strip())
            _set_label_tone(self.lookup_info, "info")
        finally:
            self._loading_fields = False
        self._store_current_site_from_editors(show_errors=False)

    def closeEvent(self, event):
        worker = self._lookup_worker
        if worker is not None:
            try:
                if worker.isRunning():
                    worker.requestInterruption()
                    worker.quit()
                    worker.wait(1000)
            except Exception:
                pass
            self._lookup_worker = None
        super().closeEvent(event)

    @Slot(int)
    def _on_preset_changed(self, _index: int) -> None:
        if self._preset_sync or self._loading_fields or self._selected_name is None:
            return
        key = str(self.preset_combo.currentData() or "custom")
        if key not in self._bhtom_preset_map:
            key = "custom"
        self._site_preset_keys[self._selected_name] = key
        if key != "custom":
            preset = self._bhtom_preset_map.get(key)
            if isinstance(preset, dict):
                preset_site = preset.get("site")
            else:
                preset_site = None
            if isinstance(preset_site, Site):
                site = self._sites.get(self._selected_name)
                if site is not None:
                    site.latitude = float(preset_site.latitude)
                    site.longitude = float(preset_site.longitude)
                    site.elevation = float(preset_site.elevation)
                    site.limiting_magnitude = float(preset_site.limiting_magnitude)
                    site.telescope_diameter_mm = float(preset_site.telescope_diameter_mm)
                    site.focal_length_mm = float(preset_site.focal_length_mm)
                    if float(preset_site.pixel_size_um) > 0.0:
                        site.pixel_size_um = float(preset_site.pixel_size_um)
                    if int(preset_site.detector_width_px) > 0:
                        site.detector_width_px = int(preset_site.detector_width_px)
                    if int(preset_site.detector_height_px) > 0:
                        site.detector_height_px = int(preset_site.detector_height_px)
                    self._loading_fields = True
                    try:
                        self.lat_edit.setText(f"{site.latitude}")
                        self.lon_edit.setText(f"{site.longitude}")
                        self.elev_edit.setText(f"{site.elevation}")
                        self.lim_mag_spin.setValue(float(site.limiting_magnitude))
                        self.telescope_diameter_spin.setValue(float(site.telescope_diameter_mm) / 1000.0)
                        self.focal_length_spin.setValue(float(site.focal_length_mm))
                        self.pixel_size_spin.setValue(float(site.pixel_size_um))
                        self.detector_width_spin.setValue(int(site.detector_width_px))
                        self.detector_height_spin.setValue(int(site.detector_height_px))
                    finally:
                        self._loading_fields = False
                    self._update_optics_summary(site)
        self._apply_custom_mode_ui()

    def _read_float(self, edit: QLineEdit, label: str) -> float:
        raw = edit.text().strip().replace(",", ".")
        if not raw:
            raise ValueError(f"{label} is required.")
        return float(raw)

    def _update_optics_summary(self, site: Optional[Site] = None) -> None:
        site_obj = site
        if site_obj is None and self._selected_name is not None:
            site_obj = self._sites.get(self._selected_name)
        if site_obj is None:
            self.pixel_scale_display.clear()
            self.fov_display.clear()
            return
        scale = site_obj.pixel_scale_arcsec_per_px
        if scale is None:
            self.pixel_scale_display.setText("-")
        else:
            self.pixel_scale_display.setText(f"{scale:.3f} arcsec/px")
        fov = site_obj.fov_arcmin
        if fov is None:
            self.fov_display.setText("-")
        else:
            self.fov_display.setText(f"{fov[0]:.1f} x {fov[1]:.1f} arcmin")

    def _store_current_site_from_editors(self, show_errors: bool = False) -> bool:
        if self._loading_fields or self._selected_name is None:
            return True
        old_name = self._selected_name
        new_name = self.name_edit.text().strip()
        if not new_name:
            if show_errors:
                QMessageBox.warning(self, "Invalid Observatory", "Name cannot be empty.")
            return False
        if new_name != old_name and new_name in self._sites:
            if show_errors:
                QMessageBox.warning(self, "Invalid Observatory", f"Observatory '{new_name}' already exists.")
            return False
        try:
            latitude = self._read_float(self.lat_edit, "Latitude")
            longitude = self._read_float(self.lon_edit, "Longitude")
            elevation = self._read_float(self.elev_edit, "Elevation")
            if not -90.0 <= latitude <= 90.0:
                raise ValueError("Latitude must be within [-90, 90].")
            if not -180.0 <= longitude <= 180.0:
                raise ValueError("Longitude must be within [-180, 180].")
        except Exception as exc:
            if show_errors:
                QMessageBox.warning(self, "Invalid Coordinates", str(exc))
            return False

        site = Site(
            name=new_name,
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            limiting_magnitude=float(self.lim_mag_spin.value()),
            custom_conditions_url=self.custom_conditions_url_edit.text().strip(),
            telescope_diameter_mm=float(self.telescope_diameter_spin.value()) * 1000.0,
            focal_length_mm=float(self.focal_length_spin.value()),
            pixel_size_um=float(self.pixel_size_spin.value()),
            detector_width_px=int(self.detector_width_spin.value()),
            detector_height_px=int(self.detector_height_spin.value()),
        )
        if new_name != old_name:
            self._sites.pop(old_name, None)
            self._sites[new_name] = site
            preset_key = str(self._site_preset_keys.pop(old_name, "custom") or "custom")
            self._site_preset_keys[new_name] = preset_key
            self._selected_name = new_name
            self._refresh_list()
        else:
            self._sites[old_name] = site
        self._update_optics_summary(site)
        return True

    def _on_limiting_mag_changed(self, _value: float) -> None:
        if self._loading_fields or self._selected_name is None:
            return
        site = self._sites.get(self._selected_name)
        if site is None:
            return
        site.limiting_magnitude = float(self.lim_mag_spin.value())
        self._update_optics_summary(site)

    @Slot()
    def _on_optics_fields_changed(self, _value: object = None) -> None:
        if self._loading_fields or self._selected_name is None:
            return
        site = self._sites.get(self._selected_name)
        if site is None:
            return
        site.telescope_diameter_mm = float(self.telescope_diameter_spin.value()) * 1000.0
        site.focal_length_mm = float(self.focal_length_spin.value())
        site.pixel_size_um = float(self.pixel_size_spin.value())
        site.detector_width_px = int(self.detector_width_spin.value())
        site.detector_height_px = int(self.detector_height_spin.value())
        self._update_optics_summary(site)

    def _on_list_selection_changed(self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]) -> None:
        if self._list_sync:
            return
        if current is None:
            self._selected_name = None
            self._set_editors_enabled(False)
            return
        name = str(current.data(Qt.UserRole) or current.text()).strip()
        site = self._sites.get(name)
        if site is None:
            self._selected_name = None
            self._set_editors_enabled(False)
            return
        self._selected_name = name
        self._loading_fields = True
        try:
            self._set_editors_enabled(True)
            self.name_edit.setText(site.name)
            self.lat_edit.setText(f"{site.latitude}")
            self.lon_edit.setText(f"{site.longitude}")
            self.elev_edit.setText(f"{site.elevation}")
            self.lim_mag_spin.setValue(float(site.limiting_magnitude))
            self.custom_conditions_url_edit.setText(str(site.custom_conditions_url or "").strip())
            self.telescope_diameter_spin.setValue(float(site.telescope_diameter_mm) / 1000.0)
            self.focal_length_spin.setValue(float(site.focal_length_mm))
            self.pixel_size_spin.setValue(float(site.pixel_size_um))
            self.detector_width_spin.setValue(int(site.detector_width_px))
            self.detector_height_spin.setValue(int(site.detector_height_px))
            self.lookup_info.setText("")
            _set_label_tone(self.lookup_info, "info")
            preset_key = str(self._site_preset_keys.get(name, "custom") or "custom")
            displayed_preset_key = preset_key if preset_key in self._bhtom_preset_map else "custom"
            self._preset_sync = True
            try:
                idx = self.preset_combo.findData(displayed_preset_key)
                if idx < 0:
                    idx = 0
                self.preset_combo.setCurrentIndex(idx)
            finally:
                self._preset_sync = False
            self._update_optics_summary(site)
        finally:
            self._loading_fields = False
        self._apply_custom_mode_ui()

    def _refresh_list(self) -> None:
        query = self.search_edit.text().strip().lower()
        previous = self._selected_name
        names = sorted(self._sites.keys(), key=str.lower)
        if query:
            names = [name for name in names if query in name.lower()]

        self._list_sync = True
        try:
            self.obs_list.clear()
            for name in names:
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, name)
                self.obs_list.addItem(item)
        finally:
            self._list_sync = False

        target_name = previous if previous in names else (names[0] if names else None)
        if target_name is None:
            self.obs_list.setCurrentItem(None)
            self._selected_name = None
            self._set_editors_enabled(False)
            return
        for row in range(self.obs_list.count()):
            item = self.obs_list.item(row)
            if item is None:
                continue
            if str(item.data(Qt.UserRole) or item.text()) == target_name:
                self.obs_list.setCurrentRow(row)
                break

    @Slot()
    def _add_observatory(self) -> None:
        if not self._store_current_site_from_editors(show_errors=True):
            return
        base = "New Observatory"
        candidate = base
        suffix = 2
        while candidate in self._sites:
            candidate = f"{base} {suffix}"
            suffix += 1
        template = self._sites.get(self._selected_name or "")
        if template is None:
            template = Site(
                name=candidate,
                latitude=0.0,
                longitude=0.0,
                elevation=0.0,
                limiting_magnitude=DEFAULT_LIMITING_MAGNITUDE,
                custom_conditions_url="",
                telescope_diameter_mm=0.0,
                focal_length_mm=0.0,
                pixel_size_um=0.0,
                detector_width_px=0,
                detector_height_px=0,
            )
        self._sites[candidate] = Site(
            name=candidate,
            latitude=float(template.latitude),
            longitude=float(template.longitude),
            elevation=float(template.elevation),
            limiting_magnitude=float(template.limiting_magnitude),
            custom_conditions_url=str(template.custom_conditions_url or "").strip(),
            telescope_diameter_mm=float(template.telescope_diameter_mm),
            focal_length_mm=float(template.focal_length_mm),
            pixel_size_um=float(template.pixel_size_um),
            detector_width_px=int(template.detector_width_px),
            detector_height_px=int(template.detector_height_px),
        )
        self._site_preset_keys[candidate] = "custom"
        self._selected_name = candidate
        self.search_edit.clear()
        self._refresh_list()
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    @Slot()
    def _remove_observatory(self) -> None:
        name = self._selected_name
        if not name:
            return
        if len(self._sites) <= 1:
            QMessageBox.warning(self, "Cannot Remove", "At least one observatory must remain.")
            return
        confirm = QMessageBox.question(
            self,
            "Remove Observatory",
            f"Remove observatory '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        self._sites.pop(name, None)
        self._site_preset_keys.pop(name, None)
        self._selected_name = None
        self._refresh_list()

    @Slot()
    def _lookup_coordinates_for_current(self) -> None:
        query = self.name_edit.text().strip()
        if not query:
            QMessageBox.warning(self, "Missing Name", "Enter observatory name first.")
            return
        if self._lookup_worker is not None and self._lookup_worker.isRunning():
            return
        parent = self.parent()
        resolver = getattr(parent, "_lookup_observatory_coordinates", None)
        if not callable(resolver):
            QMessageBox.warning(self, "Lookup Unavailable", "Coordinate lookup is unavailable in this context.")
            return
        self._set_lookup_loading_state(True)
        worker = ObservatoryLookupWorker(query=query, resolver=resolver, parent=self)
        worker.completed.connect(self._on_lookup_worker_completed)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(lambda: setattr(self, "_lookup_worker", None))
        self._lookup_worker = worker
        worker.start()

    @Slot()
    def _accept_dialog(self) -> None:
        if not self._store_current_site_from_editors(show_errors=True):
            return
        if not self._sites:
            QMessageBox.warning(self, "No Observatories", "Add at least one observatory.")
            return
        self.accept()

    @Slot()
    def _save_to_config(self) -> None:
        if not self._store_current_site_from_editors(show_errors=True):
            return
        if not self._sites:
            QMessageBox.warning(self, "No Observatories", "Add at least one observatory.")
            return
        parent = self.parent()
        saver = getattr(parent, "_save_custom_observatories", None)
        if not callable(saver):
            QMessageBox.warning(self, "Save Failed", "Saving observatory config is unavailable in this context.")
            return
        try:
            saver(self.observatories(), preset_keys=self.preset_keys())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save Failed", str(exc))
            return
        if parent is not None and hasattr(parent, "_observatory_preset_keys"):
            try:
                parent._observatory_preset_keys = self.preset_keys()
            except Exception:
                pass
        self.lookup_info.setText("Configuration saved.")
        _set_label_tone(self.lookup_info, "success")


__all__ = [
    "ObservatoryLookupWorker",
    "ObservatoryManagerDialog",
]
