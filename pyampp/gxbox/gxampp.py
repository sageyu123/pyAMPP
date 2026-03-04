import sys
import os
import select
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QComboBox,
                             QRadioButton,
                             QCheckBox, QGridLayout, QGroupBox, QButtonGroup, QVBoxLayout, QHBoxLayout, QDateTimeEdit,
                             QCalendarWidget, QTextEdit, QMessageBox, QDockWidget, QToolButton, QMenu,
                             QFileDialog)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5 import uic

from pyampp.util.config import *
import pyampp
from pathlib import Path
from pyampp.gxbox.boxutils import read_b3d_h5, validate_number
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from sunpy.coordinates import get_earth, HeliographicStonyhurst, HeliographicCarrington, Helioprojective
from sunpy.sun import constants as sun_consts
import numpy as np
import typer
import subprocess

app = typer.Typer(help="Launch the PyAmpp application.")

base_dir = Path(pyampp.__file__).parent


class CustomQLineEdit(QLineEdit):
    def setTextL(self, text):
        """
        Sets the text of the QLineEdit and moves the cursor to the beginning.

        :param text: str
            The text to set.
        """
        self.setText(text)
        self.setCursorPosition(0)


class PyAmppGUI(QMainWindow):
    """
    Main application GUI for the Solar Data Model.

    This class creates the main window and sets up the user interface for managing solar data and model configurations.

    Attributes
    ----------
    central_widget : QWidget
        The central widget of the main window.
    main_layout : QVBoxLayout
        The main layout for the central widget.

    Methods
    -------
    initUI():
        Initializes the user interface.
    add_data_repository_section():
        Adds the data repository section to the UI.
    update_sdo_data_dir():
        Updates the SDO data directory path.
    update_gxmodel_dir():
        Updates the GX model directory path.
    update_external_box_dir():
        Updates the external box directory path.
    update_dir(new_path, default_path):
        Updates the specified directory path.
    open_sdo_file_dialog():
        Opens a file dialog for selecting the SDO data directory.
    open_gx_file_dialog():
        Opens a file dialog for selecting the GX model directory.
    open_external_file_dialog():
        Opens a file dialog for selecting the external box directory.
    add_model_configuration_section():
        Adds the model configuration section to the UI.
    add_options_section():
        Adds the options section to the UI.
    add_cmd_display():
        Adds the command display section to the UI.
    add_cmd_buttons():
        Adds command buttons to the UI.
    add_status_log():
        Adds the status log section to the UI.
    update_command_display():
        Updates the command display with the current command.
    update_hpc_state(checked):
        Updates the UI when Helioprojective coordinates are selected.
    update_hgc_state(checked):
        Updates the UI when Heliographic Carrington coordinates are selected.
    get_command():
        Constructs the command based on the current UI settings.
    execute_command():
        Executes the constructed command.
    save_command():
        Saves the current command.
    refresh_command():
        Refreshes the current session.
    clear_command():
        Clears the status log.
    """

    def __init__(self):
        """
        Initializes the PyAmppGUI class.
        """
        super().__init__()
        self._gxbox_proc = None
        self._proc_partial_line = ""
        self.info_only_box = None
        self._last_model_path = None
        self._proc_timer = QTimer(self)
        self._proc_timer.setInterval(500)
        self._proc_timer.timeout.connect(self._check_gxbox_process)
        self.model_time_orig = None
        # self.rotate_to_time_button = None
        self.rotate_revert_button = None
        self.coords_center = None
        self.coords_center_orig = None
        self.initUI()

    def initUI(self):
        """
        Sets up the initial user interface for the main window.
        """
        # Main widget and layout
        uic.loadUi(Path(__file__).parent / "UI" / "gxampp.ui", self)
        self.setWindowTitle("GX Automatic Production Pipeline Interface")

        # Adding different sections
        self.add_data_repository_section()
        self.add_model_configuration_section()
        self.add_options_section()
        self.add_cmd_display()
        self.add_cmd_buttons()
        self.add_status_log()
        self.update_coords_center()

        self.update_command_display()
        self.show()

    def add_data_repository_section(self):
        self.sdo_data_edit.setText(DOWNLOAD_DIR)
        self.sdo_data_edit.returnPressed.connect(self.update_sdo_data_dir)
        self.sdo_browse_button.clicked.connect(self.open_sdo_file_dialog)

        self.gx_model_edit.setText(GXMODEL_DIR)
        self.gx_model_edit.returnPressed.connect(self.update_gxmodel_dir)
        self.gx_browse_button.clicked.connect(self.open_gx_file_dialog)

        notify_email = os.environ.get("PYAMPP_JSOC_NOTIFY_EMAIL", JSOC_NOTIFY_EMAIL)
        self.jsoc_notify_email_edit.setText(notify_email)
        self.jsoc_notify_email_edit.returnPressed.connect(self.update_jsoc_notify_email)

        self.external_box_edit.returnPressed.connect(self.update_external_box_dir)
        self.external_browse_button.clicked.connect(self.open_external_file_dialog)

    def update_sdo_data_dir(self):
        """
        Updates the SDO data directory path based on the user input.
        """
        new_path = self.sdo_data_edit.text()
        self.update_dir(new_path, DOWNLOAD_DIR)
        self.update_command_display()

    def update_gxmodel_dir(self):
        """
        Updates the GX model directory path based on the user input.
        """
        new_path = self.gx_model_edit.text()
        self.update_dir(new_path, GXMODEL_DIR)
        self.update_command_display()

    def update_jsoc_notify_email(self):
        """
        Updates the JSOC notify email via the PYAMPP_JSOC_NOTIFY_EMAIL environment variable.
        """
        new_email = self.jsoc_notify_email_edit.text().strip()
        if new_email:
            os.environ["PYAMPP_JSOC_NOTIFY_EMAIL"] = new_email
        else:
            os.environ.pop("PYAMPP_JSOC_NOTIFY_EMAIL", None)
            self.jsoc_notify_email_edit.setText(JSOC_NOTIFY_EMAIL)

    def read_external_box(self):
        """
        Reads the external box path based on the user input.
        """
        import pickle

        boxfile = self.external_box_edit.text()
        if boxfile.endswith('.h5'):
            boxdata = read_b3d_h5(boxfile)
            corona = boxdata.get("corona")
            if corona and "bx" in corona:
                nx, ny, nz = corona["bx"].shape
                self.grid_x_edit.setText(f'{nx}')
                self.grid_y_edit.setText(f'{ny}')
                self.grid_z_edit.setText(f'{nz}')
            if corona and "dr" in corona:
                dr0 = float(corona["dr"][0])
                rsun_km = sun_consts.radius.to(u.km).value
                self.res_edit.setText(f'{dr0 * rsun_km:.3f}')
            self.update_command_display()
            return

        with open(boxfile, 'rb') as f:
            boxdata = pickle.load(f)
            map_bottom = boxdata['map_bottom']
            self.model_time_orig = map_bottom.date
            if "corona" in boxdata.get("b3d", {}):
                bx = boxdata['b3d']['corona']['bx']
            else:
                bx = boxdata['b3d']['nlfff']['bx']
            nx, ny, nz = bx.shape
            box_res = map_bottom.rsun_meters.to(u.Mm) * ((map_bottom.scale[0] * 1. * u.pix).to(u.rad) / u.rad)
            center = map_bottom.center.transform_to(
                HeliographicStonyhurst(obstime=self.model_time_orig))
        self.model_time_edit.setDateTime(QDateTime(self.model_time_orig.to_datetime()))
        self.hgs_radio_button.setChecked(True)
        self.coord_x_edit.setText(f'{center.lon.to(u.deg).value}')
        self.coord_y_edit.setText(f'{center.lat.to(u.deg).value}')
        self.grid_x_edit.setText(f'{nx}')
        self.grid_y_edit.setText(f'{ny}')
        self.grid_z_edit.setText(f'{nz}')
        self.res_edit.setText(f'{box_res.to(u.km).value}')
        self.update_coords_center()
        self.coords_center_orig = self.coords_center
        self.update_command_display()

    def update_external_box_dir(self):
        """
        Updates the external box directory path based on the user input.
        """
        new_path = self.external_box_edit.text()
        self.update_dir(new_path, os.getcwd())
        if os.path.isfile(self.external_box_edit.text()):
            self.read_external_box()
        self.update_command_display()

    def update_dir(self, new_path, default_path):
        """
        Updates the specified directory path.

        :param new_path: The new directory path.
        :type new_path: str
        :param default_path: The default directory path.
        :type default_path: str
        """
        if new_path != default_path:
            # Normalize the path whether it's absolute or relative
            if not os.path.isabs(new_path):
                new_path = os.path.abspath(new_path)

            if not os.path.exists(new_path):  # Checks if the path does not exist
                # Ask user if they want to create the directory
                reply = QMessageBox.question(self, 'Create Directory?',
                                             "The directory does not exist. Do you want to create it?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

                if reply == QMessageBox.Yes:
                    try:
                        os.makedirs(new_path)
                        # QMessageBox.information(self, "Directory Created", "The directory was successfully created.")
                    except PermissionError:
                        QMessageBox.critical(self, "Permission Denied",
                                             "You do not have permission to create this directory.")
                    except OSError as e:
                        QMessageBox.critical(self, "Error", f"Failed to create directory: {str(e)}")
                else:
                    # User chose not to create the directory, revert to the original path
                    self.sdo_data_edit.setText(DOWNLOAD_DIR)
        # else:
        #     QMessageBox.warning(self, "Invalid Path", "The specified path is not a valid absolute path.")

    def open_sdo_file_dialog(self):
        """
        Opens a file dialog for selecting the SDO data directory.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name = QFileDialog.getExistingDirectory(self, "Select Directory", DOWNLOAD_DIR)
        if file_name:
            self.sdo_data_edit.setText(file_name)

    def open_gx_file_dialog(self):
        """
        Opens a file dialog for selecting the GX model directory.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name = QFileDialog.getExistingDirectory(self, "Select Directory", GXMODEL_DIR)
        if file_name:
            self.gx_model_edit.setText(file_name)

    def open_external_file_dialog(self):
        """
        Opens a file dialog for selecting the external box directory.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File", os.getcwd(), "Model Files (*.h5 *.gxbox)")
        # file_name = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        if file_name:
            self.external_box_edit.setText(file_name)
            self.read_external_box()

    def add_model_configuration_section(self):
        # Replace combo with radio style jump controls (IDL-like UX).
        self.jump_to_action_combo.setVisible(False)
        self.label_jumpToAction.setText("Jump-to Action")
        self.jump_none_radio = QRadioButton("none")
        self.jump_potential_radio = QRadioButton("potential")
        self.jump_nlfff_radio = QRadioButton("nlfff")
        self.jump_lines_radio = QRadioButton("lines")
        self.jump_chromo_radio = QRadioButton("chromo")
        self.jump_none_radio.setChecked(True)
        self.jump_button_group = QButtonGroup(self)
        for rb in [
            self.jump_none_radio,
            self.jump_potential_radio,
            self.jump_nlfff_radio,
            self.jump_lines_radio,
            self.jump_chromo_radio,
        ]:
            self.jump_button_group.addButton(rb)
            insert_idx = max(0, self.jumpToActionLayout.count() - 1)
            self.jumpToActionLayout.insertWidget(insert_idx, rb)
            rb.toggled.connect(self._sync_pipeline_options)
        self.model_time_edit.setDateTime(QDateTime.currentDateTimeUtc())
        self.model_time_edit.setDateTimeRange(QDateTime(2010, 1, 1, 0, 0, 0), QDateTime(QDateTime.currentDateTimeUtc()))
        self.model_time_edit.dateTimeChanged.connect(self.on_time_input_changed)

        self.coord_x_edit.returnPressed.connect(lambda: self.on_coord_x_input_return_pressed(self.coord_x_edit))
        self.coord_y_edit.returnPressed.connect(lambda: self.on_coord_y_input_return_pressed(self.coord_y_edit))

        self.hpc_radio_button.toggled.connect(self.update_hpc_state)
        self.hgc_radio_button.toggled.connect(self.update_hgc_state)
        self.hgs_radio_button.toggled.connect(self.update_hgs_state)

        self.grid_x_edit.returnPressed.connect(lambda: self.on_grid_x_input_return_pressed(self.grid_x_edit))
        self.grid_y_edit.returnPressed.connect(lambda: self.on_grid_y_input_return_pressed(self.grid_y_edit))
        self.grid_z_edit.returnPressed.connect(lambda: self.on_grid_z_input_return_pressed(self.grid_z_edit))
        self.res_edit.returnPressed.connect(lambda: self.on_res_input_return_pressed(self.res_edit))
        self.padding_size_edit.returnPressed.connect(
            lambda: self.on_padding_size_input_return_pressed(self.padding_size_edit))
        self.hpc_radio_button.setText("Heliocentric")
        self.hgc_radio_button.setText("Carrington")
        self.hgs_radio_button.setText("Stonyhurst")

        self.proj_group = QGroupBox("Geometrical Projection")
        self.proj_cea_radio = QRadioButton("CEA")
        self.proj_top_radio = QRadioButton("TOP")
        self.proj_cea_radio.setChecked(True)
        self.proj_button_group = QButtonGroup(self.proj_group)
        self.proj_button_group.addButton(self.proj_cea_radio)
        self.proj_button_group.addButton(self.proj_top_radio)
        proj_layout = QHBoxLayout()
        proj_layout.addWidget(self.proj_cea_radio)
        proj_layout.addWidget(self.proj_top_radio)
        proj_layout.addStretch()
        self.proj_group.setLayout(proj_layout)
        self.verticalLayout_2.addWidget(self.proj_group)
        self.proj_cea_radio.toggled.connect(self.update_command_display)
        self.proj_top_radio.toggled.connect(self.update_command_display)

        # Standalone disambiguation group in model configuration (not part of workflow options).
        self.disambig_group = QGroupBox("Pi-disambiguation")
        self.disambig_hmi_radio = QRadioButton("HMI")
        self.disambig_sfq_radio = QRadioButton("SFQ")
        self.disambig_hmi_radio.setChecked(True)
        self.disambig_button_group = QButtonGroup(self.disambig_group)
        self.disambig_button_group.addButton(self.disambig_hmi_radio)
        self.disambig_button_group.addButton(self.disambig_sfq_radio)
        disambig_layout = QHBoxLayout()
        disambig_layout.addWidget(self.disambig_hmi_radio)
        disambig_layout.addWidget(self.disambig_sfq_radio)
        disambig_layout.addStretch()
        self.disambig_group.setLayout(disambig_layout)
        self.verticalLayout_2.addWidget(self.disambig_group)
        self.disambig_hmi_radio.toggled.connect(self.update_command_display)
        self.disambig_sfq_radio.toggled.connect(self.update_command_display)

    def _get_jump_action(self):
        if self.jump_potential_radio.isChecked():
            return "potential"
        if self.jump_nlfff_radio.isChecked():
            return "nlfff"
        if self.jump_lines_radio.isChecked():
            return "lines"
        if self.jump_chromo_radio.isChecked():
            return "chromo"
        return "none"

    def _set_jump_action(self, action):
        target = (action or "none").lower()
        mapping = {
            "none": self.jump_none_radio,
            "potential": self.jump_potential_radio,
            "nlfff": self.jump_nlfff_radio,
            "lines": self.jump_lines_radio,
            "chromo": self.jump_chromo_radio,
        }
        rb = mapping.get(target, self.jump_none_radio)
        rb.blockSignals(True)
        rb.setChecked(True)
        rb.blockSignals(False)

    def add_options_section(self):
        """
        Adds the options section to the main layout.
        """
        self.optionsGroupBox.setTitle("Pipeline Workflow")
        self.download_aia_euv.setChecked(True)
        self.download_aia_uv.setChecked(True)
        self.save_empty_box.setChecked(False)
        self.save_potential_box.setChecked(False)
        self.save_bounds_box.setChecked(False)
        self.skip_nlfff_extrapolation.setChecked(False)
        self.stop_after_potential_box.setChecked(False)
        self.stop_after_potential_box.setVisible(True)
        self.stop_after_potential_box.setEnabled(True)
        self.stop_after_potential_box.setText("Stop after the potential box is generated")
        self.skip_nlfff_extrapolation.setText("Skip NLFFF extrapolation")
        self.download_aia_uv.setText("Download AIA/UV contextual maps")
        self.download_aia_euv.setText("Download AIA/EUV contextual maps")
        self.save_empty_box.setText("Save Empty Box")
        self.save_potential_box.setText("Save Potential Box")
        self.save_bounds_box.setText("Save Bounds Box")

        # Additional CLI parity controls (added programmatically to preserve .ui compatibility)
        options_layout = self.optionsGroupBox.layout()
        while options_layout.count():
            item = options_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        self.save_nas_box = QCheckBox("Save NAS")
        self.save_gen_box = QCheckBox("Save GEN")
        self.save_chr_box = QCheckBox("Save CHR")
        self.empty_box_only_box = QCheckBox("Empty Box Only")
        self.potential_only_box = QCheckBox("Potential Only")
        self.nlfff_only_box = QCheckBox("Stop after the NLFFF box is generated")
        self.generic_only_box = QCheckBox("Do not add Fontenla chromosphere model")
        self.center_vox_box = QCheckBox("Center voxel magnetic field line tracing")

        # Match legacy GX two-column workflow layout.
        options_layout.addWidget(self.download_aia_uv, 0, 0)
        options_layout.addWidget(self.download_aia_euv, 1, 0)
        options_layout.addWidget(self.save_empty_box, 2, 0)
        options_layout.addWidget(self.save_potential_box, 3, 0)
        options_layout.addWidget(self.save_bounds_box, 4, 0)
        options_layout.addWidget(self.stop_after_potential_box, 0, 1)
        options_layout.addWidget(self.skip_nlfff_extrapolation, 1, 1)
        options_layout.addWidget(self.nlfff_only_box, 2, 1)
        options_layout.addWidget(self.center_vox_box, 3, 1)
        options_layout.addWidget(self.generic_only_box, 4, 1)

        # Keep these available for CLI parity but out of this legacy-like workflow layout.
        self.save_nas_box.setVisible(False)
        self.save_gen_box.setVisible(False)
        self.save_chr_box.setVisible(False)
        self.empty_box_only_box.setVisible(False)
        self.potential_only_box.setVisible(False)

        # Update command/state when options change
        dynamic_widgets = [
            self.download_aia_euv,
            self.download_aia_uv,
            self.save_empty_box,
            self.save_potential_box,
            self.save_bounds_box,
            self.stop_after_potential_box,
            self.skip_nlfff_extrapolation,
            self.save_nas_box,
            self.save_gen_box,
            self.save_chr_box,
            self.nlfff_only_box,
            self.generic_only_box,
            self.center_vox_box,
        ]
        for w in dynamic_widgets:
            w.toggled.connect(self._sync_pipeline_options)
        self.external_box_edit.textChanged.connect(self.update_command_display)
        self.sdo_data_edit.textChanged.connect(self.update_command_display)
        self.gx_model_edit.textChanged.connect(self.update_command_display)
        self._sync_pipeline_options()

    def _set_checkbox_state(self, box, enabled):
        box.blockSignals(True)
        if not enabled and box.isChecked():
            box.setChecked(False)
        box.setEnabled(enabled)
        box.blockSignals(False)

    def _sync_pipeline_options(self, *_):
        """
        Enforce linear pipeline behavior:
        - jump disables controls for preceding stages
        - *only disables controls for following stages
        - use-potential disables NAS-stage controls
        """
        jump_stage_map = {
            "none": None,
            "potential": 1,
            "nlfff": 3,
            "lines": 4,
            "chromo": 5,
        }
        save_stage_boxes = [
            (self.save_empty_box, 0),
            (self.save_potential_box, 1),
            (self.save_bounds_box, 2),
            (self.save_nas_box, 3),
            (self.save_gen_box, 4),
            (self.save_chr_box, 5),
        ]
        only_stage_boxes = [
            (self.stop_after_potential_box, 2),
            (self.nlfff_only_box, 3),
            (self.generic_only_box, 4),
        ]

        jump_action = self._get_jump_action()
        jump_stage = jump_stage_map.get(jump_action)

        # --use-potential means NAS stage is skipped.
        if self.skip_nlfff_extrapolation.isChecked() and jump_action == "nlfff":
            self._set_jump_action("lines")
            jump_stage = 4

        use_potential = self.skip_nlfff_extrapolation.isChecked()

        # Disable NLFFF jump target when NAS is skipped.
        self.jump_nlfff_radio.setEnabled(not use_potential)

        def only_allowed(stage):
            if jump_stage is not None and stage < jump_stage:
                return False
            if use_potential and stage == 3:
                return False
            return True

        # Clear checked *only options that are no longer allowed.
        for box, stage in only_stage_boxes:
            if box.isChecked() and not only_allowed(stage):
                box.blockSignals(True)
                box.setChecked(False)
                box.blockSignals(False)

        stop_stage = None
        for box, stage in only_stage_boxes:
            if box.isChecked():
                stop_stage = stage
                break

        # --use-potential is irrelevant when stop is before NAS.
        skip_enabled = stop_stage is None or stop_stage >= 3
        self._set_checkbox_state(self.skip_nlfff_extrapolation, skip_enabled)
        use_potential = self.skip_nlfff_extrapolation.isChecked()

        # Stops constrain allowable jump destinations too.
        jump_option_stage = [
            (self.jump_none_radio, None),
            (self.jump_potential_radio, 1),
            (self.jump_lines_radio, 4),
            (self.jump_chromo_radio, 5),
        ]
        for rb, stage in jump_option_stage:
            enabled = True
            if stage is not None and stop_stage is not None and stage > stop_stage:
                enabled = False
            if stage == 3 and use_potential:
                enabled = False
            rb.setEnabled(enabled)
        self.jump_nlfff_radio.setEnabled(not use_potential and not (stop_stage is not None and 3 > stop_stage))

        selected_jump = self._get_jump_action()
        selected_stage = {"none": None, "potential": 1, "nlfff": 3, "lines": 4, "chromo": 5}.get(selected_jump)
        invalid_selected = False
        if selected_jump == "nlfff":
            invalid_selected = use_potential or (stop_stage is not None and 3 > stop_stage)
        elif selected_stage is not None:
            invalid_selected = stop_stage is not None and selected_stage > stop_stage
        if invalid_selected:
            self._set_jump_action("none")

        for box, stage in only_stage_boxes:
            enabled = only_allowed(stage)
            if stop_stage is not None and stage > stop_stage:
                enabled = False
            self._set_checkbox_state(box, enabled)

        for box, stage in save_stage_boxes:
            enabled = True
            if jump_stage is not None and stage < jump_stage:
                enabled = False
            if stop_stage is not None and stage > stop_stage:
                enabled = False
            if use_potential and stage == 3:
                enabled = False
            self._set_checkbox_state(box, enabled)

        # center-vox matters only when lines are computed (stage >= NAS.GEN and not jumping directly to CHR).
        center_vox_enabled = True
        if stop_stage is not None and stop_stage < 4:
            center_vox_enabled = False
        if jump_stage == 5:
            center_vox_enabled = False
        self._set_checkbox_state(self.center_vox_box, center_vox_enabled)

        self.update_command_display()

    def add_cmd_display(self):
        """
        Adds the command display section to the main layout.
        """
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(11)
        self.cmd_display_edit.setFont(mono)

    def add_cmd_buttons(self):
        """
        Adds the command buttons to the main layout.
        """
        self.info_only_box = QCheckBox("Info Only")
        self.info_only_box.toggled.connect(self.update_command_display)
        # Place utility toggle with execution controls, not pipeline-flow options.
        if self.cmd_button_layout is not None:
            spacer_idx = max(0, self.cmd_button_layout.count() - 1)
            self.cmd_button_layout.insertWidget(spacer_idx, self.info_only_box)
        self.execute_button.clicked.connect(self.execute_command)
        self.stop_button.clicked.connect(self.stop_command)
        self.stop_button.setEnabled(False)
        self.save_button.setVisible(False)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_command)
        self.send_to_viewer_button = QPushButton("Send to gxbox-view")
        self.send_to_viewer_button.setToolTip("Open latest generated model in gxbox-view")
        self.send_to_viewer_button.setEnabled(False)
        if self.cmd_button_layout is not None:
            spacer_idx = max(0, self.cmd_button_layout.count() - 1)
            self.cmd_button_layout.insertWidget(spacer_idx, self.send_to_viewer_button)
        self.send_to_viewer_button.clicked.connect(self.send_to_gxbox_view)
        self.clear_button_refresh.clicked.connect(self.refresh_command)
        self.clear_button_clear.setVisible(False)
        self.clear_button_clear.setEnabled(False)
        self.clear_button_clear.clicked.connect(self.clear_command)

    def add_status_log(self):
        """
        Adds the status log section to the main layout.
        """
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(10)
        self.status_log_edit.setFont(mono)
        # Move console to a right-side dock panel to keep the main workflow visible.
        for i in range(self.main_layout.count()):
            item = self.main_layout.itemAt(i)
            if item is not None and item.widget() is self.status_log_edit:
                self.main_layout.takeAt(i)
                break
        self.console_dock = QDockWidget("Console", self)
        self.console_dock.setObjectName("consoleDock")
        self.console_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.console_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.console_dock.setMinimumWidth(340)
        self.console_dock.setMaximumWidth(520)
        self.console_dock.setWidget(self.status_log_edit)
        title_bar = QWidget(self.console_dock)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(6, 2, 6, 2)
        title_layout.setSpacing(4)
        title_label = QLabel("Console", title_bar)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        self.console_menu_button = QToolButton(title_bar)
        self.console_menu_button.setText("⋮")
        self.console_menu_button.setToolTip("Console options")
        self.console_menu_button.setPopupMode(QToolButton.InstantPopup)
        console_menu = QMenu(self.console_menu_button)
        console_menu.addAction("Clear", self.clear_console)
        console_menu.addAction("Copy All", self.copy_console)
        console_menu.addAction("Save As...", self.save_console)
        self.console_menu_button.setMenu(console_menu)
        title_layout.addWidget(self.console_menu_button)
        self.console_dock.setTitleBarWidget(title_bar)
        self.addDockWidget(Qt.RightDockWidgetArea, self.console_dock)

    @validate_number
    def on_coord_x_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_coord_y_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_grid_x_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_grid_y_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_grid_z_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_res_input_return_pressed(self, widget):
        self.update_command_display(widget)

    def on_padding_size_input_return_pressed(self, widget):
        self.update_command_display(widget)

    def _remove_stretch_from_layout(self, layout):
        """
        Removes the last stretch item from the given layout if it exists.

        This method checks the last item in the layout and removes it if it is a spacer item.
        It is useful for dynamically managing layout items, especially when adding or removing widgets.

        Parameters
        ----------
        layout : QLayout
            The layout from which the stretch item should be removed.
        """
        count = layout.count()
        if count > 0 and layout.itemAt(count - 1).spacerItem():
            layout.takeAt(count - 1)

    def on_time_input_changed(self):
        self.coords_center = self._coords_center
        if self.model_time_orig is not None:
            time = Time(self.model_time_edit.dateTime().toPyDateTime()).mjd
            model_time_orig = self.model_time_orig.mjd
            time_sec_diff = (time - model_time_orig) * 24 * 3600
            print(time_sec_diff)
            if np.abs(time_sec_diff) >= 0.5:
                self.on_rotate_model_to_time()
                if self.rotate_revert_button is None:
                    self._remove_stretch_from_layout(self.model_time_layout)
                    self.rotate_revert_button = QPushButton("Revert")
                    self.rotate_revert_button.setToolTip("Revert the model to the original time")
                    self.rotate_revert_button.clicked.connect(self.on_rotate_revert_button_clicked)
                    self.model_time_layout.addWidget(self.rotate_revert_button)
                    self.model_time_layout.addStretch()
            else:
                if self.rotate_revert_button is not None:
                    self.update_coords_center(revert=True)
                    self.rotate_revert()
                    self._remove_stretch_from_layout(self.model_time_layout)
                    self.model_time_layout.removeWidget(self.rotate_revert_button)
                    self.rotate_revert_button.deleteLater()
                    self.rotate_revert_button = None
                    self.model_time_layout.addStretch()
        self.update_command_display()

    def on_rotate_revert_button_clicked(self):
        self.model_time_edit.setDateTime(QDateTime(self.model_time_orig.to_datetime()))
        self.rotate_revert()

    def rotate_revert(self):
        if self.hpc_radio_button.isChecked():
            self.update_hpc_state(True)
        elif self.hgc_radio_button.isChecked():
            self.update_hgc_state(True)
        elif self.hgs_radio_button.isChecked():
            self.update_hgs_state(True)

    def on_rotate_model_to_time(self):
        """
        Rotates the model to the specified time.
        """
        from sunpy.coordinates import RotatedSunFrame
        point = self.coords_center_orig
        time = Time(self.model_time_edit.dateTime().toPyDateTime()).mjd
        model_time_orig = self.model_time_orig.mjd
        time_sec_diff = (time - model_time_orig) * 24 * 3600
        diffrot_point = SkyCoord(RotatedSunFrame(base=point, duration=time_sec_diff * u.s))
        self.coords_center = diffrot_point.transform_to(self._coords_center.frame)
        print(self.coords_center_orig, self.coords_center)
        # self.status_log_edit.append("Model rotated to the specified time")
        if self.hpc_radio_button.isChecked():
            self.update_hpc_state(True, self.coords_center)
        elif self.hgc_radio_button.isChecked():
            self.update_hgc_state(True, self.coords_center)
        elif self.hgs_radio_button.isChecked():
            self.update_hgs_state(True, self.coords_center)
        self.update_command_display()

    def update_command_display(self, widget=None):
        """
        Updates the command display with the current command.
        """
        # print(widget)
        # if widget:
        #     if isinstance(widget, QDateTimeEdit):
        #         current_value = widget.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        #     else:
        #         current_value = widget.text()
        #     self.previous_params[widget] = self.current_params.get(widget, current_value)
        #     self.current_params[widget] = current_value
        self.coords_center = self._coords_center
        command = self.get_command()
        self.cmd_display_edit.clear()
        self.cmd_display_edit.append(" ".join(command))

    def update_hpc_state(self, checked, coords_center=None):
        """
        Updates the UI when Helioprojective coordinates are selected.

        :param checked: Whether the Helioprojective radio button is checked.
        :type checked: bool
        """
        if checked:
            self.coord_x_edit.setToolTip("Solar X coordinate of the model center in arcsec")
            self.coord_y_edit.setToolTip("Solar Y coordinate of the model center in arcsec")
            self.coord_label.setText("Center Coords  in arcsec")
            self.coord_x_label.setText("X:")
            self.coord_y_label.setText("Y:")
            if coords_center is None:
                obstime = Time(self.model_time_edit.dateTime().toPyDateTime())
                observer = get_earth(obstime)
                coords_center = self.coords_center.transform_to(Helioprojective(obstime=obstime, observer=observer))
            self.coord_x_edit.setText(f'{coords_center.Tx.to(u.arcsec).value}')
            self.coord_y_edit.setText(f'{coords_center.Ty.to(u.arcsec).value}')
            self.update_command_display()

    def update_hgc_state(self, checked, coords_center=None):
        """
        Updates the UI when Heliographic Carrington coordinates are selected.

        :param checked: Whether the Heliographic Carrington radio button is checked.
        :type checked: bool
        """
        if checked:
            self.coord_x_edit.setToolTip("Heliographic Carrington Longitude of the model center in deg")
            self.coord_y_edit.setToolTip("Heliographic Carrington Latitude of the model center in deg")
            self.coord_label.setText("Center Coords in deg")
            self.coord_x_label.setText("lon:")
            self.coord_y_label.setText("lat:")
            if coords_center is None:
                print(f'coords_center: {self.coords_center}')
                obstime = Time(self.model_time_edit.dateTime().toPyDateTime())
                observer = get_earth(obstime)
                coords_center = self.coords_center.transform_to(
                    HeliographicCarrington(obstime=obstime, observer=observer))
            print(f'new coords_center: {coords_center}')
            self.coord_x_edit.setText(f'{coords_center.lon.to(u.deg).value}')
            self.coord_y_edit.setText(f'{coords_center.lat.to(u.deg).value}')
            self.update_command_display()

    def update_hgs_state(self, checked, coords_center=None):
        """
        Updates the UI when Heliographic Stonyhurst coordinates are selected.

        :param checked: Whether the Heliographic Stonyhurst radio button is checked.
        :type checked: bool
        """
        if checked:
            self.coord_x_edit.setToolTip("Heliographic Stonyhurst Longitude of the model center in deg")
            self.coord_y_edit.setToolTip("Heliographic Stonyhurst Latitude of the model center in deg")
            self.coord_label.setText("Center Coords in deg")
            self.coord_x_label.setText("lon:")
            self.coord_y_label.setText("lat:")
            if coords_center is None:
                obstime = Time(self.model_time_edit.dateTime().toPyDateTime())
                # observer = get_earth(obstime)
                coords_center = self.coords_center.transform_to(
                    HeliographicStonyhurst(obstime=obstime))
            self.coord_x_edit.setText(f'{coords_center.lon.to(u.deg).value}')
            self.coord_y_edit.setText(f'{coords_center.lat.to(u.deg).value}')
            self.update_command_display()

    def update_coords_center(self, revert=False):
        if revert:
            self.coords_center = self.coords_center_orig
        else:
            self.coords_center = self._coords_center

    @property
    def _coords_center(self):
        time = Time(self.model_time_edit.dateTime().toPyDateTime())
        coords = [float(self.coord_x_edit.text()), float(self.coord_y_edit.text())]
        observer = get_earth(time)
        if self.hpc_radio_button.isChecked():
            coords_center = SkyCoord(coords[0] * u.arcsec, coords[1] * u.arcsec, obstime=time, observer=observer,
                                     rsun=696 * u.Mm, frame='helioprojective')
        elif self.hgc_radio_button.isChecked():
            coords_center = SkyCoord(lon=coords[0] * u.deg, lat=coords[1] * u.deg, obstime=time, observer=observer,
                                     radius=696 * u.Mm,
                                     frame='heliographic_carrington')
        elif self.hgs_radio_button.isChecked():
            coords_center = SkyCoord(lon=coords[0] * u.deg, lat=coords[1] * u.deg, obstime=time, observer=observer,
                                     radius=696 * u.Mm,
                                     frame='heliographic_stonyhurst')
        return coords_center

    def get_command(self):
        """
        Constructs the command based on the current UI settings.

        Returns
        -------
        list
            The command as a list of strings.
        """
        import astropy.time
        import astropy.units as u

        command = ['gx-fov2box']
        time = astropy.time.Time(self.model_time_edit.dateTime().toPyDateTime())
        command += ['--time', time.to_datetime().strftime('%Y-%m-%dT%H:%M:%S')]
        command += ['--coords', self.coord_x_edit.text(), self.coord_y_edit.text()]
        if self.hpc_radio_button.isChecked():
            command += ['--hpc']
        elif self.hgc_radio_button.isChecked():
            command += ['--hgc']
        else:
            command += ['--hgs']
        if self.proj_top_radio.isChecked():
            command += ['--top']
        else:
            command += ['--cea']

        command += ['--box-dims', self.grid_x_edit.text(), self.grid_y_edit.text(), self.grid_z_edit.text()]
        command += ['--dx-km', f'{float(self.res_edit.text()):.3f}']
        command += ['--pad-frac', f'{float(self.padding_size_edit.text()) / 100:.2f}']
        command += ['--data-dir', self.sdo_data_edit.text()]
        command += ['--gxmodel-dir', self.gx_model_edit.text()]
        if self.external_box_edit.text() != '':
            command += ['--entry-box', self.external_box_edit.text()]

        command += ['--euv' if self.download_aia_euv.isChecked() else '--no-euv']
        command += ['--uv' if self.download_aia_uv.isChecked() else '--no-uv']

        if self.save_empty_box.isChecked():
            command += ['--save-empty-box']
        if self.save_potential_box.isChecked():
            command += ['--save-potential']
        if self.save_bounds_box.isChecked():
            command += ['--save-bounds']
        if self.save_nas_box.isChecked():
            command += ['--save-nas']
        if self.save_gen_box.isChecked():
            command += ['--save-gen']
        if self.save_chr_box.isChecked():
            command += ['--save-chr']

        if self.empty_box_only_box.isChecked():
            command += ['--empty-box-only']
        if self.stop_after_potential_box.isChecked():
            command += ['--potential-only']
        if self.nlfff_only_box.isChecked():
            command += ['--nlfff-only']
        if self.generic_only_box.isChecked():
            command += ['--generic-only']

        if self.skip_nlfff_extrapolation.isChecked():
            command += ['--use-potential']
        if self.center_vox_box.isChecked():
            command += ['--center-vox']

        jump_action = self._get_jump_action()
        if jump_action == 'potential':
            command += ['--jump2potential']
        elif jump_action == 'nlfff':
            command += ['--jump2nlfff']
        elif jump_action == 'lines':
            command += ['--jump2lines']
        elif jump_action == 'chromo':
            command += ['--jump2chromo']

        if self.disambig_sfq_radio.isChecked():
            command += ['--sfq']

        if self.info_only_box is not None and self.info_only_box.isChecked():
            command += ['--info']

        return command

    def execute_command(self):
        """
        Executes the constructed command.
        """
        if self._gxbox_proc is not None and self._gxbox_proc.poll() is None:
            QMessageBox.warning(self, "GXbox Running", "A GXbox process is already running.")
            return

        command = self.get_command()
        self._last_model_path = None
        self.send_to_viewer_button.setEnabled(False)
        try:
            self._gxbox_proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            self._proc_partial_line = ""
            if self._gxbox_proc.stdout is not None:
                os.set_blocking(self._gxbox_proc.stdout.fileno(), False)
            self.execute_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_log_edit.append("Command started: " + " ".join(command))
            self._proc_timer.start()
        except Exception as e:
            QMessageBox.critical(self, "Execution Error", f"Failed to start command: {e}")
            self.status_log_edit.append("Command failed to start")

    def _drain_process_output(self):
        if self._gxbox_proc is None or self._gxbox_proc.stdout is None:
            return
        stdout = self._gxbox_proc.stdout
        fd = stdout.fileno()
        chunks = []
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            chunk = stdout.read()
            if not chunk:
                break
            chunks.append(chunk)
        if not chunks:
            return
        text = self._proc_partial_line + "".join(chunks).replace("\r\n", "\n").replace("\r", "\n")
        if text.endswith("\n"):
            complete_lines = text.split("\n")[:-1]
            self._proc_partial_line = ""
        else:
            parts = text.split("\n")
            complete_lines = parts[:-1]
            self._proc_partial_line = parts[-1]
        for line in complete_lines:
            if line.strip():
                self.status_log_edit.append(line)

    def stop_command(self):
        """
        Stops the running GXbox process if any.
        """
        if self._gxbox_proc is None or self._gxbox_proc.poll() is not None:
            self.status_log_edit.append("No running command to stop")
            self.stop_button.setEnabled(False)
            self.execute_button.setEnabled(True)
            return

        self.status_log_edit.append("Stopping command...")
        try:
            self._gxbox_proc.terminate()
            self._gxbox_proc.wait(timeout=5)
            self._drain_process_output()
            self.status_log_edit.append("Command stopped")
        except subprocess.TimeoutExpired:
            self._gxbox_proc.kill()
            self._drain_process_output()
            self.status_log_edit.append("Command killed")
        finally:
            self._gxbox_proc = None
            self.stop_button.setEnabled(False)
            self.execute_button.setEnabled(True)
            self._proc_timer.stop()

    def _check_gxbox_process(self):
        if self._gxbox_proc is None:
            self._proc_timer.stop()
            return

        self._drain_process_output()
        if self._gxbox_proc.poll() is None:
            return

        self._drain_process_output()
        if self._proc_partial_line.strip():
            self.status_log_edit.append(self._proc_partial_line)
            self._proc_partial_line = ""
        exit_code = self._gxbox_proc.returncode
        if exit_code == 0:
            self.status_log_edit.append("Command finished successfully")
            self._update_last_model_path()
        else:
            self.status_log_edit.append(f"Command exited with code {exit_code}")
        self._gxbox_proc = None
        self.stop_button.setEnabled(False)
        self.execute_button.setEnabled(True)
        self._proc_timer.stop()

    def save_command(self):
        """
        Saves the current command.
        """
        # Placeholder for saving command
        self.status_log_edit.append("Command saved")

    def refresh_command(self):
        """
        Refreshes the current session.
        """
        # Placeholder for refreshing command
        self.status_log_edit.append("Command refreshed")

    def clear_command(self):
        """
        Clears the status log.
        """
        # Placeholder for clearing command
        self.status_log_edit.clear()

    def clear_console(self):
        """
        Clears the console panel.
        """
        self.status_log_edit.clear()
        self._last_model_path = None
        self.send_to_viewer_button.setEnabled(False)

    def copy_console(self):
        """
        Copies the full console text to clipboard.
        """
        QApplication.clipboard().setText(self.status_log_edit.toPlainText())

    def save_console(self):
        """
        Saves the console output to a text file.
        """
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Console Output",
            str(Path.cwd() / "pyampp_console.txt"),
            "Text Files (*.txt);;Log Files (*.log);;All Files (*)",
        )
        if not file_name:
            return
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(self.status_log_edit.toPlainText())
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Could not save console output:\n{exc}")

    def _update_last_model_path(self):
        text = self.status_log_edit.toPlainText()
        candidates = re.findall(r"^- (.+\.h5)\s*$", text, flags=re.MULTILINE)
        for raw in reversed(candidates):
            p = Path(raw).expanduser()
            if p.exists():
                self._last_model_path = str(p)
                self.send_to_viewer_button.setEnabled(True)
                return
        root = Path(self.gx_model_edit.text()).expanduser()
        if not root.exists():
            return
        newest = None
        newest_mtime = -1.0
        for p in root.rglob("*.h5"):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest = p
        if newest is not None:
            self._last_model_path = str(newest)
            self.send_to_viewer_button.setEnabled(True)

    def send_to_gxbox_view(self):
        if not self._last_model_path:
            QMessageBox.information(self, "No Model", "No generated model was found to send.")
            return
        model_path = Path(self._last_model_path).expanduser()
        if not model_path.exists():
            QMessageBox.warning(self, "Missing Model", f"Model file not found:\n{model_path}")
            return
        try:
            start_dir = model_path.parent
            subprocess.Popen([
                "gxbox-view",
                "--pick",
                "--dir",
                str(start_dir),
                "--h5",
                str(model_path),
            ])
            self.status_log_edit.append(f"Launched gxbox-view with: {model_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Launch Failed", f"Could not launch gxbox-view:\n{exc}")


@app.command()
def main(
        debug: bool = typer.Option(
            False,
            "--debug",
            help="Enable debug mode with an interactive IPython session."
        )
):
    """
    Entry point for the PyAmppGUI application.

    This function initializes the PyQt application, sets up and displays the main
    GUI window for the Solar Data Model. It pre-configures some GUI elements with default
    values for model time and coordinates. Default values are set programmatically
    before the event loop starts.

    :param debug: Enable debug mode with an interactive IPython session, defaults to False
    :type debug: bool, optional
    :raises SystemExit: Exits the application loop when the GUI is closed
    :return: None
    :rtype: NoneType

    Examples
    --------
    .. code-block:: bash

        gxampp
    """

    app_qt = QApplication([])
    pyampp = PyAmppGUI()
    pyampp.model_time_edit.setDateTime(QDateTime(2024, 5, 12, 0, 0))
    pyampp.coord_x_edit.setText('0')
    pyampp.coord_y_edit.setText('0')
    pyampp.update_coords_center()
    pyampp.update_command_display()

    if debug:
        # Start an interactive IPython session for debugging
        import IPython
        IPython.embed()

        # If any matplotlib plots are created, show them
        import matplotlib.pyplot as plt
        plt.show()
    sys.exit(app_qt.exec_())

if __name__ == '__main__':
    app()
