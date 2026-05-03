"""Side-panel UI for entering a task prompt."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class SidePanel(QWidget):
    """
    Simple control strip with prompt input, submit, stop, and status.

    Emits `prompt_submitted` with the trimmed text; does not block the event loop.
    """

    prompt_submitted = pyqtSignal(str)

    def __init__(
        self,
        on_submit: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_submit = on_submit
        self._on_stop: Callable[[], None] | None = None

        self.setWindowTitle("AutomationVisionAgent")
        self.setMinimumWidth(360)
        self.setMinimumHeight(420)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._status = QLabel("Ready.")
        self._status.setWordWrap(True)
        self._status.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._input = QTextEdit()
        self._input.setPlaceholderText("Describe the task for the agent…")
        self._input.setAcceptRichText(False)

        self._submit = QPushButton("Submit")
        self._submit.clicked.connect(self._handle_submit)

        self._stop = QPushButton("Stop")
        self._stop.setEnabled(False)
        self._stop.clicked.connect(self._handle_stop)

        row = QHBoxLayout()
        row.addWidget(self._submit, alignment=Qt.AlignmentFlag.AlignLeft)
        row.addWidget(self._stop, alignment=Qt.AlignmentFlag.AlignLeft)
        row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Task prompt"))
        layout.addWidget(self._input, stretch=1)
        layout.addLayout(row)
        layout.addWidget(QLabel("Status"))
        layout.addWidget(self._status, stretch=0)

        if self._on_submit is not None:
            self.prompt_submitted.connect(self._on_submit)

    def set_stop_handler(self, handler: Callable[[], None]) -> None:
        """Wire emergency stop (Stop button + Escape when focused)."""
        self._on_stop = handler

    def set_status(self, text: str) -> None:
        """Update the status label."""
        self._status.setText(text)

    def set_running(self, running: bool) -> None:
        """Disable submit while a background run is in progress; enable Stop."""
        self._submit.setEnabled(not running)
        self._stop.setEnabled(running)
        self._input.setReadOnly(running)

    def _handle_submit(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            self.set_status("Enter a prompt before submitting.")
            return
        self.prompt_submitted.emit(text)

    def _handle_stop(self) -> None:
        if self._on_stop is not None:
            self._on_stop()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is not None and event.key() == Qt.Key.Key_Escape:
            self._handle_stop()
        super().keyPressEvent(event)
