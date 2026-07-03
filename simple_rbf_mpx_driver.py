"""Simple RBF Driver MPxNode for Maya.

Load as a Python plug-in, then open the UI:

    import maya.cmds as cmds
    cmds.loadPlugin(r"C:\path\to\simple_rbf_mpx_driver.py")

    import simple_rbf_mpx_driver
    simple_rbf_mpx_driver.show()

This version is a real DG node. Arbitrary numeric driver attributes are
connected to driverInput[], and output[] is connected to driven attributes.
It evaluates during Maya DG refresh instead of relying on scriptJob.
"""

from __future__ import annotations

import json
import math
import os
import traceback

import maya.api.OpenMaya as om
import maya.cmds as cmds

try:
    from maya import OpenMayaUI as omui
    from shiboken2 import wrapInstance
    from PySide2 import QtWidgets
except Exception:
    from maya import OpenMayaUI as omui
    from shiboken6 import wrapInstance
    from PySide6 import QtWidgets


NODE_NAME = "simpleRbfMpxDriver"
NODE_ID = om.MTypeId(0x0013F7A1)
DATA_ATTR = "rbfData"
WINDOW_OBJECT_NAME = "simpleRbfMpxDriverUI"


def maya_useNewAPI():
    """Tell Maya this Python plug-in uses maya.api.OpenMaya API 2.0."""
    pass


def _maya_main_window():
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget)


def _as_float(value, fallback=0.0):
    try:
        return float(value)
    except Exception:
        return fallback


def distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def gaussian(distance_value, radius):
    width = radius if radius > 1.0e-8 else 1.0
    x = distance_value / width
    return math.exp(-(x * x))


def normalize(values):
    total = sum(values)
    if abs(total) < 1.0e-12:
        return [0.0 for _ in values]
    return [value / total for value in values]


def distance_matrix(poses):
    return [[distance(a, b) for b in poses] for a in poses]


def mean_off_diagonal(matrix):
    values = []
    for row_index, row in enumerate(matrix):
        for col_index in range(row_index + 1, len(row)):
            values.append(row[col_index])
    return sum(values) / float(len(values)) if values else 1.0


def activation_row(driver_values, poses, radius):
    return normalize([gaussian(distance(driver_values, pose), radius) for pose in poses])


def activation_matrix(poses, radius):
    return [activation_row(pose, poses, radius) for pose in poses]


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def lu_decompose(matrix, epsilon=0.0):
    size = len(matrix)
    lu = [list(row) for row in matrix]
    perm = list(range(size))

    if epsilon:
        for index in range(size):
            lu[index][index] += epsilon

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(lu[row][column]))
        if abs(lu[pivot_row][column]) < 1.0e-12:
            raise ValueError("Activation matrix is singular.")

        if pivot_row != column:
            lu[column], lu[pivot_row] = lu[pivot_row], lu[column]
            perm[column], perm[pivot_row] = perm[pivot_row], perm[column]

        for row in range(column + 1, size):
            lu[row][column] /= lu[column][column]
            for update_column in range(column + 1, size):
                lu[row][update_column] -= lu[row][column] * lu[column][update_column]

    return lu, perm


def lu_solve(lu, perm, rhs):
    size = len(lu)
    result = [float(rhs[perm[index]]) for index in range(size)]

    for row in range(size):
        for column in range(row):
            result[row] -= lu[row][column] * result[column]

    for row in range(size - 1, -1, -1):
        for column in range(row + 1, size):
            result[row] -= lu[row][column] * result[column]
        if abs(lu[row][row]) < 1.0e-12:
            raise ValueError("LU back substitution failed.")
        result[row] /= lu[row][row]

    return result


def solve_weights(poses, values):
    if len(poses) < 2:
        raise ValueError("At least two poses are required.")

    distances = distance_matrix(poses)
    radius = mean_off_diagonal(distances)
    phi = activation_matrix(poses, radius)

    errors = []
    for epsilon in (0.0, 1.0e-10, 1.0e-8, 1.0e-6):
        try:
            lu, perm = lu_decompose(phi, epsilon)
            columns = [lu_solve(lu, perm, column) for column in transpose(values)]
            return transpose(columns), radius, epsilon
        except Exception as exc:
            errors.append(str(exc))

    raise ValueError("Could not solve RBF weights: {0}".format("; ".join(errors)))


def evaluate_outputs(driver_values, poses, weights, radius):
    if not poses or not weights:
        return []

    phi = activation_row(driver_values, poses, radius)
    output_count = len(weights[0])
    outputs = [0.0] * output_count

    for pose_index, influence in enumerate(phi):
        for output_index in range(output_count):
            outputs[output_index] += weights[pose_index][output_index] * influence

    return outputs


def selected_channel_attrs():
    channel_box = "mainChannelBox"
    attrs = cmds.channelBox(channel_box, query=True, selectedMainAttributes=True) or []
    objects = cmds.ls(selection=True) or []
    result = []

    for obj in objects:
        for attr in attrs:
            plug = obj + "." + attr
            if cmds.objExists(plug):
                result.append(plug)

    return result


def _lines(text):
    normalized = text.replace(",", "\n").replace(";", "\n")
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def _unique(items):
    return list(dict.fromkeys(items))


def _numeric_attr_value(attr):
    value = cmds.getAttr(attr)
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            raise ValueError("{0} is compound data. Pick scalar child attrs.".format(attr))
        raise ValueError("{0} is not a scalar numeric attr.".format(attr))
    return float(value)


def _disconnect_destination(destination):
    sources = cmds.listConnections(destination, source=True, destination=False, plugs=True) or []
    for source in sources:
        try:
            cmds.disconnectAttr(source, destination)
        except Exception:
            pass


class SimpleRbfMpxNode(om.MPxNode):
    driverInput = om.MObject()
    allowNegativeWeights = om.MObject()
    rbfData = om.MObject()
    output = om.MObject()

    @staticmethod
    def creator():
        return SimpleRbfMpxNode()

    @staticmethod
    def initialize():
        numeric_attr = om.MFnNumericAttribute()
        typed_attr = om.MFnTypedAttribute()

        SimpleRbfMpxNode.driverInput = numeric_attr.create(
            "driverInput", "din", om.MFnNumericData.kDouble, 0.0
        )
        numeric_attr.array = True
        numeric_attr.usesArrayDataBuilder = True
        numeric_attr.storable = False
        numeric_attr.keyable = True
        numeric_attr.readable = False
        numeric_attr.writable = True
        SimpleRbfMpxNode.addAttribute(SimpleRbfMpxNode.driverInput)

        SimpleRbfMpxNode.allowNegativeWeights = numeric_attr.create(
            "allowNegativeWeights", "anw", om.MFnNumericData.kBoolean, False
        )
        numeric_attr.storable = True
        numeric_attr.keyable = True
        numeric_attr.readable = True
        numeric_attr.writable = True
        SimpleRbfMpxNode.addAttribute(SimpleRbfMpxNode.allowNegativeWeights)

        SimpleRbfMpxNode.rbfData = typed_attr.create(
            "rbfData", "rbfd", om.MFnData.kString
        )
        typed_attr.storable = True
        typed_attr.keyable = False
        typed_attr.hidden = True
        SimpleRbfMpxNode.addAttribute(SimpleRbfMpxNode.rbfData)

        SimpleRbfMpxNode.output = numeric_attr.create(
            "output", "out", om.MFnNumericData.kDouble, 0.0
        )
        numeric_attr.array = True
        numeric_attr.usesArrayDataBuilder = True
        numeric_attr.storable = False
        numeric_attr.keyable = False
        numeric_attr.readable = True
        numeric_attr.writable = False
        SimpleRbfMpxNode.addAttribute(SimpleRbfMpxNode.output)

        SimpleRbfMpxNode.attributeAffects(SimpleRbfMpxNode.driverInput, SimpleRbfMpxNode.output)
        SimpleRbfMpxNode.attributeAffects(SimpleRbfMpxNode.allowNegativeWeights, SimpleRbfMpxNode.output)
        SimpleRbfMpxNode.attributeAffects(SimpleRbfMpxNode.rbfData, SimpleRbfMpxNode.output)

    def compute(self, plug, data_block):
        is_output = plug.attribute() == SimpleRbfMpxNode.output
        is_output_element = plug.isElement and plug.array().attribute() == SimpleRbfMpxNode.output
        if not is_output and not is_output_element:
            return None

        try:
            data_handle = data_block.inputValue(SimpleRbfMpxNode.rbfData)
            raw = data_handle.asString() or "{}"
            data = json.loads(raw)
            allow_negative = data_block.inputValue(SimpleRbfMpxNode.allowNegativeWeights).asBool()

            poses = data.get("poses", [])
            weights = data.get("weights", [])
            radius = _as_float(data.get("radius", 1.0), 1.0)
            driver_count = int(data.get("driver_count", 0))
            output_count = int(data.get("output_count", 0))

            driver_values = [0.0] * driver_count
            input_handle = data_block.inputArrayValue(SimpleRbfMpxNode.driverInput)
            for logical_index in range(driver_count):
                try:
                    input_handle.jumpToLogicalElement(logical_index)
                    driver_values[logical_index] = input_handle.inputValue().asDouble()
                except RuntimeError:
                    driver_values[logical_index] = 0.0

            outputs = evaluate_outputs(driver_values, poses, weights, radius)
            if not allow_negative:
                outputs = [max(0.0, value) for value in outputs]
            while len(outputs) < output_count:
                outputs.append(0.0)

            output_handle = data_block.outputArrayValue(SimpleRbfMpxNode.output)
            builder = output_handle.builder()
            for index, value in enumerate(outputs[:output_count]):
                element = builder.addElement(index)
                element.setDouble(float(value))
            output_handle.set(builder)
            output_handle.setAllClean()
        except Exception:
            traceback.print_exc()
            raise


class SimpleRbfMpxUI(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SimpleRbfMpxUI, self).__init__(parent or _maya_main_window())
        self.setObjectName(WINDOW_OBJECT_NAME)
        self.setWindowTitle("Simple RBF MPx Driver")
        self.setMinimumWidth(620)

        self.node = None
        self.poses = []
        self._build()

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.name_field = QtWidgets.QLineEdit("simpleRbfMpxDriver1")
        self.driver_field = QtWidgets.QPlainTextEdit()
        self.driver_field.setPlaceholderText("pCone1.rotateX\nctrl.customDriver")
        self.driven_field = QtWidgets.QPlainTextEdit()
        self.driven_field.setPlaceholderText("pCone1.a\npCone1.b")
        self.allow_negative_box = QtWidgets.QCheckBox()
        self.allow_negative_box.setChecked(False)

        driver_buttons = QtWidgets.QHBoxLayout()
        use_driver_channels = QtWidgets.QPushButton("Use Selected Channels")
        use_driver_channels.clicked.connect(self.pick_driver_attrs)
        add_common_rotate = QtWidgets.QPushButton("Add Rotate XYZ")
        add_common_rotate.clicked.connect(self.add_rotate_attrs)
        driver_buttons.addWidget(use_driver_channels)
        driver_buttons.addWidget(add_common_rotate)

        driven_buttons = QtWidgets.QHBoxLayout()
        use_driven_channels = QtWidgets.QPushButton("Use Selected Channels")
        use_driven_channels.clicked.connect(self.pick_driven_attrs)
        driven_buttons.addWidget(use_driven_channels)

        form.addRow("Name", self.name_field)
        form.addRow("Driver Attributes", self.driver_field)
        form.addRow("", driver_buttons)
        form.addRow("Driven Attributes", self.driven_field)
        form.addRow("", driven_buttons)
        form.addRow("Allow Negative Weights", self.allow_negative_box)
        layout.addLayout(form)

        self.pose_list = QtWidgets.QListWidget()
        layout.addWidget(self.pose_list)

        pose_buttons = QtWidgets.QHBoxLayout()
        add_pose = QtWidgets.QPushButton("Add Pose")
        add_pose.clicked.connect(self.add_pose)
        remove_pose = QtWidgets.QPushButton("Remove Pose")
        remove_pose.clicked.connect(self.remove_pose)
        clear_pose = QtWidgets.QPushButton("Clear")
        clear_pose.clicked.connect(self.clear_poses)
        pose_buttons.addWidget(add_pose)
        pose_buttons.addWidget(remove_pose)
        pose_buttons.addWidget(clear_pose)
        layout.addLayout(pose_buttons)

        solve_buttons = QtWidgets.QHBoxLayout()
        create = QtWidgets.QPushButton("Create / Solve Node")
        create.clicked.connect(self.create_or_solve)
        load = QtWidgets.QPushButton("Load Selected Node")
        load.clicked.connect(self.load_selected_node)
        solve_buttons.addWidget(create)
        solve_buttons.addWidget(load)
        layout.addLayout(solve_buttons)

        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)

    def driver_attrs(self):
        return _unique(_lines(self.driver_field.toPlainText()))

    def driven_attrs(self):
        return _unique(_lines(self.driven_field.toPlainText()))

    def pick_driver_attrs(self):
        attrs = selected_channel_attrs()
        if attrs:
            self.driver_field.setPlainText("\n".join(attrs))

    def pick_driven_attrs(self):
        attrs = selected_channel_attrs()
        if attrs:
            self.driven_field.setPlainText("\n".join(attrs))

    def add_rotate_attrs(self):
        selection = cmds.ls(selection=True) or []
        attrs = self.driver_attrs()
        for obj in selection:
            for channel in ("rotateX", "rotateY", "rotateZ"):
                plug = obj + "." + channel
                if cmds.objExists(plug):
                    attrs.append(plug)
        self.driver_field.setPlainText("\n".join(_unique(attrs)))

    def _validate_attrs(self):
        driver_attrs = self.driver_attrs()
        driven_attrs = self.driven_attrs()

        if not driver_attrs:
            raise ValueError("Add at least one driver attribute.")
        if not driven_attrs:
            raise ValueError("Add at least one driven attribute.")

        invalid = [attr for attr in driver_attrs + driven_attrs if not cmds.objExists(attr)]
        if invalid:
            raise ValueError("Invalid attributes: {0}".format(", ".join(invalid)))

        overlap = sorted(set(driver_attrs).intersection(driven_attrs))
        if overlap:
            raise ValueError("Driver and driven attributes cannot be the same: {0}".format(", ".join(overlap)))

        for attr in driver_attrs + driven_attrs:
            _numeric_attr_value(attr)

        return driver_attrs, driven_attrs

    def add_pose(self):
        try:
            driver_attrs, driven_attrs = self._validate_attrs()
            driver_values = [_numeric_attr_value(attr) for attr in driver_attrs]
            driven_values = [_numeric_attr_value(attr) for attr in driven_attrs]
            self.poses.append({"driver": driver_values, "values": driven_values})
            self.refresh_pose_list()
        except Exception as exc:
            cmds.warning(str(exc))

    def remove_pose(self):
        row = self.pose_list.currentRow()
        if 0 <= row < len(self.poses):
            self.poses.pop(row)
            self.refresh_pose_list()

    def clear_poses(self):
        self.poses = []
        self.refresh_pose_list()

    def refresh_pose_list(self):
        self.pose_list.clear()
        for index, pose in enumerate(self.poses):
            driver_text = ", ".join("{0:.3f}".format(value) for value in pose["driver"])
            value_text = ", ".join("{0:.3f}".format(value) for value in pose["values"])
            self.pose_list.addItem("#{0}: [{1}] -> [{2}]".format(index, driver_text, value_text))

    def create_or_solve(self):
        try:
            driver_attrs, driven_attrs = self._validate_attrs()
            if len(self.poses) < 2:
                raise ValueError("Add at least two poses.")

            driver_count = len(driver_attrs)
            output_count = len(driven_attrs)
            for pose in self.poses:
                if len(pose["driver"]) != driver_count or len(pose["values"]) != output_count:
                    raise ValueError("Attribute count changed after recording poses. Clear and re-add poses.")

            poses = [pose["driver"] for pose in self.poses]
            values = [pose["values"] for pose in self.poses]
            weights, radius, epsilon = solve_weights(poses, values)

            if self.node and cmds.objExists(self.node):
                node = self.node
            else:
                node = cmds.createNode(NODE_NAME, name=self.name_field.text().strip() or NODE_NAME)
                self.node = node

            allow_negative = self.allow_negative_box.isChecked()
            payload = {
                "version": 1,
                "driver_attrs": driver_attrs,
                "driven_attrs": driven_attrs,
                "driver_count": driver_count,
                "output_count": output_count,
                "poses": poses,
                "pose_values": values,
                "weights": weights,
                "radius": radius,
                "regularization": epsilon,
                "kernel": "gaussian",
                "phi": "normalized",
                "allow_negative_weights": allow_negative,
            }
            cmds.setAttr(node + "." + DATA_ATTR, json.dumps(payload, indent=2), type="string")
            cmds.setAttr(node + ".allowNegativeWeights", allow_negative)

            for index, attr in enumerate(driver_attrs):
                destination = "{0}.driverInput[{1}]".format(node, index)
                if not cmds.isConnected(attr, destination):
                    cmds.connectAttr(attr, destination, force=True)

            for index, attr in enumerate(driven_attrs):
                source = "{0}.output[{1}]".format(node, index)
                _disconnect_destination(attr)
                cmds.connectAttr(source, attr, force=True)

            cmds.dgdirty(node)
            self.status.setText(
                "Solved {0} poses, radius {1:.6f}, node {2}".format(len(poses), radius, node)
            )
        except Exception as exc:
            cmds.warning(str(exc))

    def load_selected_node(self):
        selection = cmds.ls(selection=True) or []
        for node in selection:
            if cmds.nodeType(node) == NODE_NAME:
                raw = cmds.getAttr(node + "." + DATA_ATTR) or "{}"
                payload = json.loads(raw)
                self.node = node
                self.name_field.setText(node)
                self.driver_field.setPlainText("\n".join(payload.get("driver_attrs", [])))
                self.driven_field.setPlainText("\n".join(payload.get("driven_attrs", [])))
                if cmds.attributeQuery("allowNegativeWeights", node=node, exists=True):
                    self.allow_negative_box.setChecked(cmds.getAttr(node + ".allowNegativeWeights"))
                else:
                    self.allow_negative_box.setChecked(payload.get("allow_negative_weights", False))
                self.poses = []
                for driver, values in zip(payload.get("poses", []), payload.get("pose_values", [])):
                    self.poses.append({"driver": driver, "values": values})
                self.refresh_pose_list()
                self.status.setText("Loaded {0}".format(node))
                return
        cmds.warning("Select a {0} node.".format(NODE_NAME))


def show():
    load_this_plugin()
    for widget in QtWidgets.QApplication.allWidgets():
        if widget.objectName() == WINDOW_OBJECT_NAME:
            widget.close()
            widget.deleteLater()
    dialog = SimpleRbfMpxUI()
    dialog.show()
    return dialog


def initializePlugin(plugin):
    plugin_fn = om.MFnPlugin(plugin, "MingHan(Martin) Lee", "1.0.0", "Any")
    plugin_fn.registerNode(
        NODE_NAME,
        NODE_ID,
        SimpleRbfMpxNode.creator,
        SimpleRbfMpxNode.initialize,
        om.MPxNode.kDependNode,
    )


def uninitializePlugin(plugin):
    plugin_fn = om.MFnPlugin(plugin)
    plugin_fn.deregisterNode(NODE_ID)


def load_this_plugin():
    path = os.path.abspath(__file__)
    if NODE_NAME not in (cmds.allNodeTypes() or []):
        cmds.loadPlugin(path)
