import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QComboBox,
                             QRadioButton,
                             QCheckBox, QGridLayout, QGroupBox, QVBoxLayout, QHBoxLayout, QDateTimeEdit,
                             QCalendarWidget, QTextEdit, QMessageBox,
                             QFileDialog)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QSize, QDateTime, Qt
from PyQt5 import uic

from pyampp.util.config import *
import pyampp
from pathlib import Path
import argparse
from pyampp.gxbox.boxutils import validate_number
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from sunpy.coordinates import get_earth, HeliographicStonyhurst, HeliographicCarrington, Helioprojective
import numpy as np


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

    def read_external_box(self):
        """
        Reads the external box path based on the user input.
        """
        import pickle

        boxfile = self.external_box_edit.text()
        with open(boxfile, 'rb') as f:
            boxdata = pickle.load(f)
            map_bottom = boxdata['map_bottom']
            self.model_time_orig = map_bottom.date
            nx, ny, nz = boxdata['b3d']['nlfff']['bx'].shape
            box_res = map_bottom.rsun_meters.to(u.Mm) * ((map_bottom.scale[0] * 1. * u.pix).to(u.rad) / u.rad)
            center = map_bottom.center.transform_to(
                HeliographicStonyhurst(obstime=self.model_time_orig))
        self.model_time_edit.setDateTime(QDateTime(self.model_time_orig.to_datetime()))
        self.hgs_radio_button.setChecked(True)
        self.coord_x_edit.setTextL(f'{center.lon.to(u.deg).value}')
        self.coord_y_edit.setTextL(f'{center.lat.to(u.deg).value}')
        self.grid_x_edit.setTextL(f'{nx}')
        self.grid_y_edit.setTextL(f'{ny}')
        self.grid_z_edit.setTextL(f'{nz}')
        self.res_edit.setTextL(f'{box_res.to(u.km).value}')
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
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File", os.getcwd(), "gxbox Files (*.gxbox)")
        # file_name = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        if file_name:
            self.external_box_edit.setText(file_name)
            self.read_external_box()

    def add_model_configuration_section(self):
        self.jump_to_action_combo.addItems(['none', 'potential', 'NLFFF', 'lines', 'chromo'])
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

    def add_options_section(self):
        """
        Adds the options section to the main layout.
        """
        pass

    def add_cmd_display(self):
        """
        Adds the command display section to the main layout.
        """
        pass

    def add_cmd_buttons(self):
        """
        Adds the command buttons to the main layout.
        """
        self.execute_button.clicked.connect(self.execute_command)
        self.save_button.clicked.connect(self.save_command)
        self.clear_button_refresh.clicked.connect(self.refresh_command)
        self.clear_button_clear.clicked.connect(self.clear_command)

    def add_status_log(self):
        """
        Adds the status log section to the main layout.
        """
        pass

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
        print(self.coords_center_orig,self.coords_center)
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
            self.coord_x_edit.setTextL(f'{coords_center.Tx.to(u.arcsec).value}')
            self.coord_y_edit.setTextL(f'{coords_center.Ty.to(u.arcsec).value}')
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
                obstime = Time(self.model_time_edit.dateTime().toPyDateTime())
                observer = get_earth(obstime)
                coords_center = self.coords_center.transform_to(HeliographicCarrington(obstime=obstime,observer=observer))
            self.coord_x_edit.setTextL(f'{coords_center.lon.to(u.deg).value}')
            self.coord_y_edit.setTextL(f'{coords_center.lat.to(u.deg).value}')
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
                coords_center = self.coords_center.transform_to(HeliographicStonyhurst(obstime=obstime))
            self.coord_x_edit.setTextL(f'{coords_center.lon.to(u.deg).value}')
            self.coord_y_edit.setTextL(f'{coords_center.lat.to(u.deg).value}')
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
        if self.hpc_radio_button.isChecked():
            observer = get_earth(time)
            coords_center = SkyCoord(coords[0] * u.arcsec, coords[1] * u.arcsec, obstime=time, observer=observer,
                                     rsun=696 * u.Mm, frame='helioprojective')
        elif self.hgc_radio_button.isChecked():
            coords_center = SkyCoord(lon=coords[0] * u.deg, lat=coords[1] * u.deg, obstime=time,
                                     radius=696 * u.Mm,
                                     frame='heliographic_carrington')
        elif self.hgs_radio_button.isChecked():
            coords_center = SkyCoord(lon=coords[0] * u.deg, lat=coords[1] * u.deg, obstime=time,
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
        command = ['python', os.path.join(base_dir, 'gxbox', 'gxbox_factory.py')]
        time = astropy.time.Time(self.model_time_edit.dateTime().toPyDateTime())
        command += ['--time', time.to_datetime().strftime('%Y-%m-%dT%H:%M:%S')]

        if self.hpc_radio_button.isChecked():
            command += ['--coords', self.coord_x_edit.text(), self.coord_y_edit.text(), '--hpc']
        elif self.hgc_radio_button.isChecked():
            command += ['--coords', self.coord_x_edit.text(), self.coord_y_edit.text(), '--hgc']
        else:
            command += ['--coords', self.coord_x_edit.text(), self.coord_y_edit.text(), '--hgs']

        command += ['--box_dims', self.grid_x_edit.text(), self.grid_y_edit.text(), self.grid_z_edit.text()]
        command += ['--box_res', f'{((float(self.res_edit.text()) * u.km).to(u.Mm)).value:.3f}']
        command += ['--pad_frac', f'{float(self.padding_size_edit.text()) / 100:.2f}']
        command += ['--data_dir', self.sdo_data_edit.text()]
        command += ['--gxmodel_dir', self.gx_model_edit.text()]
        if self.external_box_edit.text() != '':
            command += ['--external_box', self.external_box_edit.text()]
        # print(command)
        return command

    def execute_command(self):
        """
        Executes the constructed command.
        """
        self.status_log_edit.append("Command executed")
        import subprocess
        command = self.get_command()
        subprocess.run(command, check=True)

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


def main():
    """
    Entry point for the PyAmppGUI application.

    This function initializes the PyQt application, sets up and displays the main GUI window for the Solar Data Model.
    It pre-configures some of the GUI elements with default values for the model time and coordinates.

    No parameters are taken directly by this function. All configurations are done within the GUI or passed through the
    global environment.

    Examples
    --------
    To run the GUI application, execute the script from the command line in the project directory:

    .. code-block:: bash

        python pyAMPP/pyampp/gxbox/gxampp.py

    This command initializes the PyQt application loop and opens the main window of the PyAmppGUI, where all interactions
    occur. Default values for date and coordinates are set programmatically before the event loop starts.
    """
    parser = argparse.ArgumentParser(description="Run GxBox with specified parameters.")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with interactive session.')
    args = parser.parse_args()

    app = QApplication([])
    pyampp = PyAmppGUI()
    pyampp.model_time_edit.setDateTime(QDateTime(2014, 11, 1, 16, 40))
    pyampp.coord_x_edit.setText('-632')
    pyampp.coord_y_edit.setText('-135')
    pyampp.update_coords_center()
    pyampp.update_command_display()
    if args.debug:
        # Start an interactive IPython session for debugging
        import IPython
        IPython.embed()
        import matplotlib.pyplot as plt
        plt.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
