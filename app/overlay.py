"""Minimal fullscreen transparent overlay for cursor feedback."""

from __future__ import annotations

import time
from dataclasses import dataclass

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QWidget


@dataclass
class _ClickPulse:
    center_global: QPoint
    until: float


class OverlayWindow(QWidget):
    """
    Always-on-top, transparent overlay that does not steal mouse input when possible.

    Draws a simple cursor ring and optional click pulse at global screen coordinates.

    Cross-thread updates: emit `cursor_update_requested` / `click_effect_requested` from worker
    threads (e.g. automation); they queue to the GUI thread reliably via Qt signals.
    """

    cursor_update_requested = pyqtSignal(int, int)
    click_effect_requested = pyqtSignal(int, int)
    visibility_for_capture = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cursor_global = QPoint(0, 0)
        self._pulses: list[_ClickPulse] = []
        self._tick = QTimer(self)
        self._tick.setInterval(33)
        self._tick.timeout.connect(self._on_tick)

        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        app = QApplication.instance()
        if app is not None:
            geo = self._union_screen_geometry(app)
            self.setGeometry(geo)

        self._tick.start()

        self.cursor_update_requested.connect(
            self.update_cursor,
            Qt.ConnectionType.QueuedConnection,
        )
        self.click_effect_requested.connect(
            self.show_click_effect,
            Qt.ConnectionType.QueuedConnection,
        )
        self.visibility_for_capture.connect(
            self._set_hidden_for_capture,
            Qt.ConnectionType.QueuedConnection,
        )

    @pyqtSlot(bool)
    def _set_hidden_for_capture(self, hide: bool) -> None:
        """Hide the overlay so it is not included in mss screenshots (main thread)."""
        self.setVisible(not hide)

    @staticmethod
    def _union_screen_geometry(app: QApplication) -> QRect:
        from PyQt6.QtGui import QGuiApplication

        screens = QGuiApplication.screens()
        if not screens:
            ps = app.primaryScreen()
            return ps.geometry() if ps is not None else QRect(0, 0, 1280, 800)
        rect = screens[0].geometry()
        for s in screens[1:]:
            rect = rect.united(s.geometry())
        return rect

    @pyqtSlot(int, int)
    def update_cursor(self, x: int, y: int) -> None:
        """Move the visual cursor indicator to global coordinates."""
        self._cursor_global = QPoint(int(x), int(y))
        self.update()

    @pyqtSlot(int, int)
    def show_click_effect(self, x: int, y: int) -> None:
        """Brief pulse at global coordinates."""
        self._pulses.append(_ClickPulse(QPoint(int(x), int(y)), time.monotonic() + 0.45))
        self.update()

    def _on_tick(self) -> None:
        now = time.monotonic()
        before = len(self._pulses)
        self._pulses = [p for p in self._pulses if p.until > now]
        if len(self._pulses) != before:
            self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        local_cursor = self.mapFromGlobal(self._cursor_global)
        pen = QPen(QColor(80, 200, 255, 220))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(QColor(80, 200, 255, 40))
        r = 14
        painter.drawEllipse(local_cursor, r, r)

        now = time.monotonic()
        for pulse in self._pulses:
            p = self.mapFromGlobal(pulse.center_global)
            fade = max(0.0, (pulse.until - now) / 0.45)
            alpha = int(180 * fade)
            pen2 = QPen(QColor(255, 200, 80, alpha))
            pen2.setWidth(4)
            painter.setPen(pen2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pr = int(10 + (1.0 - fade) * 22)
            painter.drawEllipse(p, pr, pr)
